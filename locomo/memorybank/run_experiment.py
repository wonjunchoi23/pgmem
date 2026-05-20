"""
MemoryBank Batch Experiment Runner — LoComo (QA-Only Variant)

Runs MemoryBank memory-augmented LLM agent on the LoComo dataset,
processing multiple samples in parallel via batched LLM calls.

Batch unit: sample (each sample is independent).

Per-batch flow:
  Phase 1 — Memory Construction (batched by session_idx):
    For each session_idx across all active samples in lock-step:
      1. Store all turns in that session (embedding only, no LLM)
      2. [batch LLM] Generate session event summaries (one call per sample)
      3. [batch LLM] Generate session personality summaries (one call per sample)
         → event summary also added to FAISS as searchable document

  Phase 1 End (batch across all samples):
    4. [batch LLM] Synthesize global event summaries
    5. [batch LLM] Synthesize global personality portraits
    6. Apply Ebbinghaus forgetting curve per sample (no LLM)

  Phase 2 — QA Answering (batched by temperature group × QA_BATCH_SIZE):
    For each QA pair (across all samples):
      7. Retrieve memories by cosine similarity (no strength update)
      8. Build QA prompt (category-aware, deterministic cat-5 ordering)
    Then for each temperature group in QA_BATCH_SIZE chunks:
      9. [batch LLM] Answer QA
      10. Write retrieval log entry

  Phase 3 — Cleanup (per sample):
    Save memory snapshot (if SAVE_MEMORY_SNAPSHOTS)
    Clear memory
    Save results + checkpoint

Usage:
    python run_experiment.py \\
        --start-sample 0 --end-sample 4 \\
        --batch-size 4 \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --tensor-parallel 1 --gpu-memory 0.5 \\
        --config config_0
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore

from load_dataset import (
    load_locomo_dataset,
    Sample,
    LoCoMoSession,
    Turn,
    QAPair,
)

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"memorybank_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers
    )
    return logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path) -> Tuple[Set[str], Optional[int]]:
    if not checkpoint_file.exists():
        return set(), None
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        if "completed_sample_ids" in data:
            return {str(x) for x in data["completed_sample_ids"]}, None
        # Backward compat: old index-based format
        return set(), data.get("last_completed_sample_index")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set(), None


def save_checkpoint(
    checkpoint_file: Path,
    completed_sample_ids: Set[str],
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
):
    data = {
        "completed_sample_ids": sorted(completed_sample_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name": config_name,
            "model": model_path,
            "start_sample": start_sample,
            "end_sample": end_sample,
        },
    }
    _atomic_write(checkpoint_file, data)


def load_existing_results(results_file: Path) -> List[Dict]:
    if not results_file.exists():
        return []
    try:
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results from {results_file}")
        return results
    except Exception as e:
        logger.warning(f"Failed to load existing results ({e}). Starting fresh.")
        return []


def save_results(results_file: Path, results: List[Dict]):
    _atomic_write(results_file, results)


def _atomic_write(path: Path, data):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# =============================================================================
# RETRIEVAL LOGGING
# =============================================================================

def write_retrieval_log(log_path: Path, entry: Dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _build_context_breakdown(
    retrieved_items: List[Dict],
    event_summary_length: int,
    user_portrait_length: int,
) -> Dict[str, object]:
    dialogue_count = sum(
        1 for item in retrieved_items
        if item.get("memory_subtype") == "dialogue_snippet"
    )
    summary_count = sum(
        1 for item in retrieved_items
        if item.get("memory_subtype") == "daily_summary"
    )

    return {
        "memory_type": ["dialogue_snippet", "daily_summary"],
        "num_retrieved": [dialogue_count, summary_count],
        "prompt_context_type": [
            "dialogue_snippet",
            "daily_summary",
            "global_event_summary",
            "user_portrait",
        ],
        "num_prompt_context": [
            dialogue_count,
            summary_count,
            1 if event_summary_length > 0 else 0,
            1 if user_portrait_length > 0 else 0,
        ],
        "retrieved_item_total": len(retrieved_items),
    }


def build_retrieval_log_entry(
    phase: str,
    sample_id: str,
    session_id: Optional[int],
    dia_id: Optional[str],
    query: str,
    retrieved_items: List[Dict],
    total_memories: int,
    event_summary_length: int,
    user_portrait_length: int,
    memo_dates: str = "",
    prompt_block_count: int = 0,
    type_counts: Optional[Dict] = None,
) -> Dict:
    breakdown = _build_context_breakdown(
        retrieved_items,
        event_summary_length,
        user_portrait_length,
    )
    scores = [item["score"] for item in retrieved_items if "score" in item]
    return {
        "phase": phase,
        "sample_id": sample_id,
        "session_id": session_id,
        "dia_id": dia_id,
        "query": query,
        "memory_type": breakdown["memory_type"],
        "num_retrieved": breakdown["num_retrieved"],
        "retrieved_items": retrieved_items,
        "retrieval_scores": scores,
        "module_specific": {
            "module": "memorybank_batch",
            "total_memories": total_memories,
            "event_summary_length": event_summary_length,
            "user_portrait_length": user_portrait_length,
            "retrieved_item_total": breakdown["retrieved_item_total"],
            "prompt_context_type": breakdown["prompt_context_type"],
            "num_prompt_context": breakdown["num_prompt_context"],
            "memo_dates": memo_dates,
            "prompt_block_count": prompt_block_count,
            "type_counts": type_counts or {},
        },
    }


# =============================================================================
# HELPERS
# =============================================================================

def _format_dialogue(turns: List[Turn]) -> str:
    return "\n".join(f"[{turn.speaker}]: {turn.text}" for turn in turns)


# =============================================================================
# BATCHED RUNNER
# =============================================================================

class BatchedMemoryBankRunner:
    """
    Runs MemoryBank on multiple LoComo samples simultaneously via batched LLM calls.

    Phase 1 batches session-level LLM calls (event + personality summaries) across
    all samples at the same session index. Phase 2 batches QA calls across all
    samples grouped by temperature.
    """

    def __init__(self, llm_client, model_path: str, shared_embedding_model, config_metadata: Dict):
        self.llm_client = llm_client
        self.model_path = model_path
        self.shared_embedding_model = shared_embedding_model
        self.config_metadata = config_metadata

    def run_batch(
        self,
        samples: List[Sample],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
    ) -> List[Dict]:
        from agent import MemoryBankAgent, LLMCallLogger

        agents = [
            MemoryBankAgent(
                self.llm_client,
                self.model_path,
                embedding_model=self.shared_embedding_model,
            )
            for _ in samples
        ]

        for agent, prompt_log_dir in zip(agents, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                agent.set_llm_logger(LLMCallLogger(prompt_log_dir))

        phase1_data, history_buffers = self._run_phase1_batched(samples, agents)
        qa_results_list, phase2_stats = self._run_phase2_batched(
            samples, agents, retrieval_log_paths, history_buffers
        )

        results = []
        for i, (sample, agent) in enumerate(zip(samples, agents)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"sample_{sample.sample_id}"
                agent.save_memory_snapshot(snapshot_dir)
                logger.info(f"Memory snapshot saved: sample {sample.sample_id}")

            # Collect token stats before clear_memory() resets counters
            summary_tokens = agent.get_and_reset_summary_tokens()
            agent.clear_memory()
            logger.info(f"Memory cleared: sample {sample.sample_id}")

            p2 = phase2_stats[i]
            all_call_types = {
                **summary_tokens,
                "call_5_qa": {
                    "input": p2["qa_input"],
                    "output": p2["qa_output"],
                    "llm_calls": p2["num_qa_calls"],
                },
            }
            token_statistics = {
                **all_call_types,
                "total_input":     sum(v["input"]     for v in all_call_types.values()),
                "total_output":    sum(v["output"]    for v in all_call_types.values()),
                "total_llm_calls": sum(v["llm_calls"] for v in all_call_types.values()),
            }

            results.append({
                "sample_id": sample.sample_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": phase1_data[i]["memory_stats"],
                "qa_results": qa_results_list[i],
                "token_statistics": token_statistics,
                "phase1_statistics": phase1_data[i]["internal_stats"],
                "memory_snapshot_path": (
                    f"memory_snapshots/sample_{sample.sample_id}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            })
        return results

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_batched(
        self,
        samples: List[Sample],
        agents,
    ) -> Tuple[List[Dict], List[List[List[Turn]]]]:
        """
        Process all sessions in lockstep by session_idx.

        For each session_idx:
          - Store all turns via embedding (no LLM)
          - Batch LLM: session event summaries for all active samples
          - Batch LLM: session personality summaries for all active samples
          - Apply combined daily summary results

        After all sessions:
          - Batch LLM: global event synthesis
          - Batch LLM: global personality synthesis
          - Apply combined global summary results
          - Apply forgetting per sample

        Returns:
            (phase1_data, history_buffers)
            history_buffers[i] = list of per-session turn lists, for sample i
        """
        max_sessions = max(len(s.sessions) for s in samples)
        # history_buffers[sample_idx] = [session_0_turns, session_1_turns, ...]
        history_buffers: List[List[List[Turn]]] = [[] for _ in samples]

        for session_idx in tqdm(range(max_sessions), desc="Phase1 sessions"):
            # Collect samples that have this session
            active = [
                (i, samples[i], agents[i], samples[i].sessions[session_idx])
                for i in range(len(samples))
                if session_idx < len(samples[i].sessions)
            ]

            # ---- Store turns (embedding only, no LLM) + accumulate history ----
            for i, sample, agent, session in active:
                for turn in session.turns:
                    content = f"[{turn.speaker}]: {turn.text}"
                    agent.add_memory(
                        content=content,
                        session_id=session.session_id,
                        date_str=session.date_time,
                        timestamp=turn.dia_id,
                    )
                history_buffers[i].append(list(session.turns))

            # ---- Build session prompts ----
            event_prompts = []
            personality_prompts = []
            for i, sample, agent, session in active:
                dialogue_text = _format_dialogue(session.turns)
                ep, pp = agent.memory_system.build_daily_summary_prompts(
                    dialogue_text=dialogue_text,
                    speaker_a=sample.speaker_a,
                    speaker_b=sample.speaker_b,
                )
                event_prompts.append(ep)
                personality_prompts.append(pp)

            # ---- Batch LLM: session event summaries ----
            event_texts, event_usages, _ = self._batch_generate_with_retry(
                prompts=event_prompts,
                system_prompt=None,
                max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                temperature=cfg.SUMMARIZE_TEMPERATURE,
                guided_json=None,
            )
            for j, (i, sample, agent, session) in enumerate(active):
                agent.accumulate_summary_tokens(
                    event_usages[j]["prompt_tokens"],
                    event_usages[j]["completion_tokens"],
                    1,
                    call_type="call_1_daily_event",
                )
                if agent._llm_logger is not None:
                    agent._llm_logger.log(
                        "call_1_daily_event", None, event_prompts[j], event_texts[j]
                    )
                logger.debug(
                    f"Session {session.session_id} event summary "
                    f"(sample {sample.sample_id}): {len(event_texts[j])} chars"
                )

            # ---- Batch LLM: session personality summaries ----
            personality_texts, personality_usages, _ = self._batch_generate_with_retry(
                prompts=personality_prompts,
                system_prompt=None,
                max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                temperature=cfg.SUMMARIZE_TEMPERATURE,
                guided_json=None,
            )
            for j, (i, sample, agent, session) in enumerate(active):
                agent.accumulate_summary_tokens(
                    personality_usages[j]["prompt_tokens"],
                    personality_usages[j]["completion_tokens"],
                    1,
                    call_type="call_2_daily_personality",
                )
                if agent._llm_logger is not None:
                    agent._llm_logger.log(
                        "call_2_daily_personality",
                        None,
                        personality_prompts[j],
                        personality_texts[j],
                    )
                # Combined apply after both batches are complete
                agent.memory_system.apply_daily_summary_results(
                    session_id=session.session_id,
                    date_str=session.date_time,
                    event_summary=event_texts[j],
                    personality_summary=personality_texts[j],
                )

        # ---- Batch LLM: global synthesis ----
        global_prompts_list = [
            agent.memory_system.build_global_summary_prompts() for agent in agents
        ]
        active_global = [
            (i, prompts)
            for i, prompts in enumerate(global_prompts_list)
            if prompts is not None
        ]

        if active_global:
            g_indices = [i for i, _ in active_global]
            g_event_prompts = [prompts[0] for _, prompts in active_global]
            g_personality_prompts = [prompts[1] for _, prompts in active_global]

            # Global event
            g_event_texts, g_event_usages, _ = self._batch_generate_with_retry(
                prompts=g_event_prompts,
                system_prompt=None,
                max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                temperature=cfg.SUMMARIZE_TEMPERATURE,
                guided_json=None,
            )
            for j, i in enumerate(g_indices):
                agents[i].accumulate_summary_tokens(
                    g_event_usages[j]["prompt_tokens"],
                    g_event_usages[j]["completion_tokens"],
                    1,
                    call_type="call_3_global_event",
                )
                if agents[i]._llm_logger is not None:
                    agents[i]._llm_logger.log(
                        "call_3_global_event", None, g_event_prompts[j], g_event_texts[j]
                    )

            # Global personality
            g_personality_texts, g_personality_usages, _ = self._batch_generate_with_retry(
                prompts=g_personality_prompts,
                system_prompt=None,
                max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                temperature=cfg.SUMMARIZE_TEMPERATURE,
                guided_json=None,
            )
            for j, i in enumerate(g_indices):
                agents[i].accumulate_summary_tokens(
                    g_personality_usages[j]["prompt_tokens"],
                    g_personality_usages[j]["completion_tokens"],
                    1,
                    call_type="call_4_global_personality",
                )
                if agents[i]._llm_logger is not None:
                    agents[i]._llm_logger.log(
                        "call_4_global_personality",
                        None,
                        g_personality_prompts[j],
                        g_personality_texts[j],
                    )
                # Combined apply after both global batches are complete
                agents[i].memory_system.apply_global_summary_results(
                    event_summary=g_event_texts[j],
                    user_portrait=g_personality_texts[j],
                )

        # ---- Apply forgetting (no LLM) ----
        for sample, agent in zip(samples, agents):
            last_date_time = sample.sessions[-1].date_time
            agent.apply_forgetting(last_date_time)
            logger.info(
                f"Sample {sample.sample_id}: forgetting applied (now={last_date_time}), "
                f"memory count post-forgetting: {agent.get_memory_count()}"
            )

        # ---- Collect memory stats and internal stats after Phase 1 ----
        phase1_data = []
        for agent in agents:
            phase1_data.append({
                "memory_stats": agent.get_memory_stats(),
                "internal_stats": agent.get_and_reset_internal_stats(),
            })

        return phase1_data, history_buffers

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        samples: List[Sample],
        agents,
        retrieval_log_paths: List[Path],
        history_buffers: List[List[List[Turn]]],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        from agent import QA_SCHEMA

        # ---- Collect all QA jobs (retrieval + prompt build) ----
        qa_jobs = []
        for sample_idx, (sample, agent) in enumerate(zip(samples, agents)):
            # Build flat turn list for the most recent HISTORY_SESSION_WINDOW sessions
            recent_sessions = history_buffers[sample_idx][-cfg.HISTORY_SESSION_WINDOW:]
            recent_turns = [turn for session_turns in recent_sessions for turn in session_turns]

            for qa_idx, qa in enumerate(sample.qa):
                retrieval_result = agent.retrieve_memory(
                    qa.question,
                    k=cfg.RETRIEVE_K,
                    update_strength=False,
                )

                log_entry = build_retrieval_log_entry(
                    phase="qa",
                    sample_id=sample.sample_id,
                    session_id=None,
                    dia_id=None,
                    query=qa.question,
                    retrieved_items=retrieval_result.items,
                    total_memories=retrieval_result.total_memories,
                    event_summary_length=len(agent.get_event_summary()),
                    user_portrait_length=len(agent.get_user_portrait()),
                    memo_dates=retrieval_result.memo_dates,
                    prompt_block_count=retrieval_result.prompt_block_count,
                    type_counts=retrieval_result.type_counts,
                )
                write_retrieval_log(retrieval_log_paths[sample_idx], log_entry)

                choice_seed = f"{sample.sample_id}::{qa_idx}::{qa.question}"
                prompt, temperature = agent.build_qa_prompt(
                    question=qa.question,
                    retrieved_memory=retrieval_result.formatted,
                    memo_dates=retrieval_result.memo_dates,
                    category=qa.category,
                    adversarial_answer=qa.adversarial_answer or "",
                    choice_order_seed=choice_seed,
                    history=recent_turns,
                )

                retrieved_metadata = [
                    {
                        "memory_subtype": item.get("memory_subtype"),
                        "source_label": item.get("source_label", ""),
                        "content_preview": item.get("content_preview", ""),
                        "score": item.get("score"),
                    }
                    for item in retrieval_result.items
                ]

                qa_jobs.append({
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "sample_id": sample.sample_id,
                    "qa": qa,
                    "retrieved_memory": retrieval_result.formatted,
                    "memo_dates": retrieval_result.memo_dates,
                    "history": recent_turns,
                    "retrieved_metadata": retrieved_metadata,
                    "prompt": prompt,
                    "temperature": temperature,
                    "choice_seed": choice_seed,
                })

        qa_results_per_sample = [[] for _ in samples]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0}
            for _ in samples
        ]

        if not qa_jobs:
            return qa_results_per_sample, list(phase2_stats)

        # ---- Group by temperature and batch ----
        jobs_by_temperature: Dict[float, List[Dict]] = {}
        for job in qa_jobs:
            jobs_by_temperature.setdefault(job["temperature"], []).append(job)

        all_outputs = []
        for temperature, jobs in jobs_by_temperature.items():
            for chunk_start in tqdm(
                range(0, len(jobs), cfg.QA_BATCH_SIZE),
                desc=f"Phase2 QA temp={temperature}",
            ):
                chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
                chunk_prompts = [job["prompt"] for job in chunk]

                def retry_single(local_idx: int, _prompt: str):
                    job = chunk[local_idx]
                    agent = agents[job["sample_idx"]]
                    answer, token_info, prompt_used = agent.answer_qa(
                        question=job["qa"].question,
                        retrieved_memory=job["retrieved_memory"],
                        memo_dates=job["memo_dates"],
                        category=job["qa"].category,
                        adversarial_answer=job["qa"].adversarial_answer or "",
                        choice_order_seed=job["choice_seed"],
                        history=job["history"],
                    )
                    result = {"answer": answer}
                    usage = {
                        "prompt_tokens": token_info.get("input", 0),
                        "completion_tokens": token_info.get("output", 0),
                    }
                    return result, usage, True, prompt_used

                chunk_results, chunk_usages, chunk_logged = self._batch_generate_with_retry(
                    prompts=chunk_prompts,
                    system_prompt=None,
                    max_tokens=cfg.MAX_TOKENS,
                    temperature=temperature,
                    guided_json=QA_SCHEMA,
                    sequential_retry_fn=retry_single,
                )

                for job, result, usage, already_logged in zip(
                    chunk, chunk_results, chunk_usages, chunk_logged
                ):
                    all_outputs.append((job, result, usage, already_logged))

        all_outputs.sort(key=lambda item: (item[0]["sample_idx"], item[0]["qa_idx"]))

        for job, result, usage, already_logged in all_outputs:
            sample_idx = job["sample_idx"]

            if not already_logged and agents[sample_idx]._llm_logger is not None:
                agents[sample_idx]._llm_logger.log(
                    "call_5_qa", None, job["prompt"], result
                )

            phase2_stats[sample_idx]["qa_input"] += usage["prompt_tokens"]
            phase2_stats[sample_idx]["qa_output"] += usage["completion_tokens"]
            phase2_stats[sample_idx]["num_qa_calls"] += 1

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            qa_results_per_sample[sample_idx].append({
                "question": job["qa"].question,
                "category": job["qa"].category,
                "generated_answer": answer,
                "ground_truth_answer": job["qa"].final_answer,
                "evidence": job["qa"].evidence,
                "retrieved_memories": job["retrieved_metadata"],
                "qa_tokens": {
                    "input": usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "model": self.model_path,
                },
            })

        return qa_results_per_sample, list(phase2_stats)

    # ------------------------------------------------------------------
    # Batch LLM helper
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        guided_json=None,
        sequential_retry_fn=None,
    ) -> Tuple[List, List[Dict], List[bool]]:
        """
        Call generate_batch_raw and handle failures.

        For non-JSON calls (guided_json=None): returns raw text strings.
        For JSON calls: parses each item and retries failed items sequentially.

        Returns:
            (results, usages, already_logged)
            - results: list of strings (no guided_json) or dicts (with guided_json)
            - usages:  list of {"prompt_tokens": int, "completion_tokens": int}
            - already_logged: list of bool (True if sequential fallback already logged)
        """
        if not prompts:
            return [], [], []

        try:
            texts, usages = self.llm_client.generate_batch_raw(
                prompts=prompts,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                guided_json=guided_json,
                return_usage=True,
            )
        except Exception as e:
            if sequential_retry_fn is None:
                raise
            logger.warning(
                f"Batch generation failed; falling back to sequential for whole chunk: {e}"
            )
            results, usages, already_logged = [], [], []
            for idx, prompt in enumerate(prompts):
                result, usage, logged, _ = sequential_retry_fn(idx, prompt)
                results.append(result)
                usages.append(usage)
                already_logged.append(logged)
            return results, usages, already_logged

        already_logged = [False] * len(prompts)

        # No JSON parsing needed for raw text calls
        if guided_json is None:
            return texts, usages, already_logged

        from llm_client import _parse_json_response

        parsed = []
        retry_indices = []
        for idx, text in enumerate(texts):
            try:
                parsed.append(_parse_json_response(text) if isinstance(text, str) else text)
            except (json.JSONDecodeError, ValueError):
                parsed.append(None)
                retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
            if sequential_retry_fn is not None:
                result, usage, logged, _ = sequential_retry_fn(idx, prompts[idx])
                parsed[idx] = result
                usages[idx] = usage
                already_logged[idx] = logged
                continue
            try:
                retry_result = self.llm_client.generate(
                    prompt=prompts[idx],
                    system_prompt=system_prompt,
                    guided_json=guided_json,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                if isinstance(retry_result, dict):
                    usage_info = retry_result.pop("_usage", {})
                    usages[idx] = {
                        "prompt_tokens": usage_info.get("prompt_tokens", 0),
                        "completion_tokens": usage_info.get("completion_tokens", 0),
                    }
                    parsed[idx] = retry_result
                else:
                    parsed[idx] = {}
            except Exception as e:
                logger.error(f"Sequential retry for batch item {idx} failed: {e}")
                parsed[idx] = {}

        return parsed, usages, already_logged


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def create_llm_client(
    model_path: str,
    tensor_parallel: int,
    gpu_memory: float,
    max_model_len: Optional[int] = None,
):
    from llm_client import create_llm_client as _create

    engine = cfg.LLM_ENGINE
    if engine == "vllm":
        vllm_kwargs = dict(
            engine="vllm",
            model_path=model_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_memory,
            download_dir=None,
        )
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len
        return _create(**vllm_kwargs)
    if engine == "together":
        return _create(engine="together", **cfg.TOGETHER_CONFIG)
    if engine == "openai":
        return _create(engine="openai", **cfg.OPENAI_CONFIG)
    raise ValueError(f"Unknown LLM engine: {engine}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ---- Step 1: parse --config first to load the right module ----
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_0")
    pre_args, _ = pre_parser.parse_known_args()
    config_name = Path(pre_args.config).stem

    import importlib.util

    config_file = _MODULE_DIR / f"{config_name}.py"
    if not config_file.exists():
        print(f"[error] Config file not found: {config_file}")
        return 1
    spec = importlib.util.spec_from_file_location(config_name, config_file)
    global cfg
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    sys.modules["config"] = cfg

    from agent import MemoryBankAgent  # noqa: F401

    # ---- Step 2: full argument parsing ----
    parser = argparse.ArgumentParser(
        description="MemoryBank Batch Experiment on LoComo (QA-Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-sample", type=int, required=True,
                        help="First sample index (inclusive)")
    parser.add_argument("--end-sample", type=int, required=True,
                        help="Last sample index (inclusive)")
    parser.add_argument("--model", type=str,
                        default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override vLLM max_model_len")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE,
                        help="Number of samples to process in parallel")
    parser.add_argument("--config", type=str, default="config_0",
                        help="Config file name (without .py)")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_sample > args.end_sample:
        parser.error("--start-sample must be <= --end-sample")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    cfg.ensure_directories(args.model, args.start_sample, args.end_sample, config_name)

    sample_dir = cfg.get_sample_dir(
        args.model, args.start_sample, args.end_sample, config_name
    )
    global logger
    logger = setup_logging(sample_dir / "logs")

    results_file = cfg.get_results_file(
        args.model, args.start_sample, args.end_sample, config_name
    )
    checkpoint_file = cfg.get_checkpoint_file(
        args.model, args.start_sample, args.end_sample, config_name
    )
    retrieval_log_dir = cfg.get_retrieval_log_dir(
        args.model, args.start_sample, args.end_sample, config_name
    )
    snapshots_dir = cfg.get_memory_snapshots_dir(
        args.model, args.start_sample, args.end_sample, config_name
    )
    prompt_log_dir = cfg.get_prompt_log_dir(
        args.model, args.start_sample, args.end_sample, config_name
    )

    logger.info("=" * 60)
    logger.info("MemoryBank Batch Experiment — LoComo (QA-Only)")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Samples         : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Forgetting div  : {cfg.FORGETTING_DIVISOR}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    logger.info("Loading dataset...")
    samples = load_locomo_dataset(cfg.DATASET_PATH)

    if args.end_sample >= len(samples):
        logger.error(
            f"end_sample={args.end_sample} out of range "
            f"(dataset has {len(samples)} samples)"
        )
        return 1

    target_samples = samples[args.start_sample: args.end_sample + 1]

    # ---- Checkpoint resume ----
    completed_sample_ids: Set[str] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_sample_ids, last_completed_index = load_checkpoint(checkpoint_file)
        if last_completed_index is not None:
            # Convert old index-based checkpoint to id-based
            completed_sample_ids.update(
                str(target_samples[i].sample_id)
                for i in range(min(last_completed_index + 1, len(target_samples)))
            )
        if completed_sample_ids:
            logger.info(
                f"Resuming: {len(completed_sample_ids)} samples already completed"
            )

    pending_samples = [
        s for s in target_samples if str(s.sample_id) not in completed_sample_ids
    ]
    if not pending_samples:
        logger.info("All samples already completed.")
        return 0

    # ---- Load shared embedding model once ----
    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer

    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    # ---- Init LLM ----
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    config_metadata = {
        "config_name":       config_name,
        "model":             args.model,
        "embedding_model":   cfg.EMBEDDING_MODEL,
        "temperature":       cfg.TEMPERATURE,
        "temperature_c5":    cfg.TEMPERATURE_C5,
        "max_tokens":        cfg.MAX_TOKENS,
        "sample_range":      [args.start_sample, args.end_sample],
        "retrieve_k":        cfg.RETRIEVE_K,
        "forgetting_divisor": cfg.FORGETTING_DIVISOR,
    }

    runner = BatchedMemoryBankRunner(
        llm_client=llm_client,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
        config_metadata=config_metadata,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# MemoryBank Batch  |  LoComo (QA-Only)")
    print(f"# Model      : {cfg.extract_model_name(args.model)}")
    print(
        f"# Samples [{args.start_sample}, {args.end_sample}]  "
        f"({len(pending_samples)} to process)"
    )
    print(f"# Batch size : {args.batch_size}")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_samples) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_samples[
            batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size
        ]
        sample_ids = [sample.sample_id for sample in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: samples {sample_ids}")

        batch_retrieval_log_paths = [
            retrieval_log_dir / f"sample_{sample.sample_id}_retrieval_log.jsonl"
            for sample in batch
        ]
        batch_prompt_log_dirs = [
            prompt_log_dir / f"sample_{sample.sample_id}"
            for sample in batch
        ]

        try:
            batch_results = runner.run_batch(
                samples=batch,
                retrieval_log_paths=batch_retrieval_log_paths,
                snapshots_dir=snapshots_dir,
                prompt_log_dirs=batch_prompt_log_dirs,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed with hard error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        for result in batch_results:
            results.append(result)
            save_results(results_file, results)
            if cfg.ENABLE_CHECKPOINTING:
                completed_sample_ids.add(str(result["sample_id"]))
                save_checkpoint(
                    checkpoint_file=checkpoint_file,
                    completed_sample_ids=completed_sample_ids,
                    model_path=args.model,
                    start_sample=args.start_sample,
                    end_sample=args.end_sample,
                    config_name=config_name,
                )
            logger.info(
                f"Sample {result['sample_id']} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total LLM calls: "
                f"{result['token_statistics']['total_llm_calls']}"
            )

    print(f"\n{'#' * 60}")
    print(f"# Batch experiment complete!  Results -> {results_file}")
    print(f"{'#' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
