"""
LD-Agent Batch Experiment Runner (LoComo)

Changes from previous version:
  - import time removed; all timing tracking removed
  - avg_timing() removed
  - timing_statistics removed from result
  - token_statistics restructured to per-call-type breakdown:
      call_1_speaker1_persona, call_2_speaker2_persona, call_3_summarization, call_4_qa
  - api_calls → llm_calls throughout
  - config_metadata added to result
  - memory_at_qa_start added to result
  - EventMemory uses numpy+SentenceTransformer (shared encoder/tokenizer across agents)

Per-batch flow:
  Phase 1 — Memory Construction (batched across samples):
    Turn slots iterated in lock-step:
      1. Flush batch (samples needing session-boundary flush) → call_3_summarization
      2. STM append (all active, no LLM)
      3. Persona batch (all active) → call_1_speaker1_persona or call_2_speaker2_persona

  Final flush — Commit remaining STM to LTM before QA (STM retained):
    → call_3_summarization

  Phase 2 — QA Answering (batched by temperature group):
    → call_4_qa

Usage:
CUDA_VISIBLE_DEVICES=0 nohup python run_experiment.py \\
    --start-sample 0 --end-sample 9 \\
    --model Qwen/Qwen3-1.7B \\
    --tensor-parallel 1 --gpu-memory 0.11 \\
    --max-model-len 3000 \\
    --config config_0 \\
    > nohup/nohup_1.7b_sample_0_9.out 2>&1 &
"""

import argparse
import json
import logging
import os
import sys
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

from load_dataset import LoCoMoSession, QAPair, Sample, Turn, load_locomo_dataset
from ldagent_module import LDAgentModule, LLMCallLogger, _count_traits
from generator import format_memories_for_prompt, format_context_for_prompt, QA_RESPONSE_SCHEMA
from event_memory import SUMMARY_SCHEMA
from personas import TRAIT_SYS_PROMPT, TRAIT_SCHEMA

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"ldagent_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
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
        return set(), data.get("last_completed_sample_index")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set(), None


def save_checkpoint(
    checkpoint_file:      Path,
    completed_sample_ids: Set[str],
    model_path:           str,
    start_sample:         int,
    end_sample:           int,
    config_name:          str = "config",
):
    data = {
        "completed_sample_ids": sorted(completed_sample_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name":  config_name,
            "model":        model_path,
            "start_sample": start_sample,
            "end_sample":   end_sample,
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


# =============================================================================
# HELPERS
# =============================================================================

def flatten_sample_turns(sample: Sample) -> List[Dict]:
    """
    Flatten all sessions/turns into a single ordered list with Unix timestamps.
    """
    from dateutil import parser as dtparser

    flat: List[Dict] = []
    for session in sample.sessions:
        try:
            ts_base = dtparser.parse(session.date_time).timestamp()
        except Exception:
            logger.warning(
                f"Could not parse date_time {session.date_time!r} for sample "
                f"{sample.sample_id}; using 0.0"
            )
            ts_base = 0.0

        for turn_idx, turn in enumerate(session.turns):
            flat.append({
                "session_id":  session.session_id,
                "session_num": session.session_id,
                "date_time":   session.date_time,
                "dia_id":      turn.dia_id,
                "speaker":     turn.speaker,
                "text":        turn.text,
                "timestamp":   ts_base + turn_idx * cfg.SECONDS_PER_TURN,
                "sample_id":   sample.sample_id,
                "speaker_a":   sample.speaker_a,
            })
    return flat


# =============================================================================
# BATCHED RUNNER
# =============================================================================

class BatchedLDAgentRunner:
    def __init__(
        self,
        llm_client,
        model_path:      str,
        config_metadata: Dict,
        encoder=None,
        lemma_tokenizer=None,
    ):
        self.llm_client      = llm_client
        self.model_path      = model_path
        self.config_metadata = config_metadata
        self.encoder         = encoder
        self.lemma_tokenizer = lemma_tokenizer

    # =========================================================================
    # run_batch — top-level entry point for one batch of samples
    # =========================================================================

    def run_batch(
        self,
        samples:             List[Sample],
        retrieval_log_paths: List[Path],
        snapshots_dir:       Path,
        prompt_log_dirs:     List[Path],
    ) -> List[Dict]:

        # ── Initialise one agent per sample ──────────────────────────────────
        agents = [
            LDAgentModule(
                llm_client=self.llm_client,
                config=cfg,
                logger=logger,
                sample_id=sample.sample_id,
                speaker_a=sample.speaker_a,
                speaker_b=sample.speaker_b,
                encoder=self.encoder,
                lemma_tokenizer=self.lemma_tokenizer,
            )
            for sample in samples
        ]

        for agent, prompt_log_dir in zip(agents, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                agent.set_llm_logger(LLMCallLogger(prompt_log_dir))

        # ── Phase 1: memory construction ─────────────────────────────────────
        self._run_phase1_batched(samples, agents)

        # ── Final flush before QA (STM retained) ─────────────────────────────
        self._run_final_flush_batched(samples, agents)

        # ── Memory stats at QA start ──────────────────────────────────────────
        memory_stats_list = [agent.get_memory_stats() for agent in agents]

        # ── Collect Phase 1 token counts ──────────────────────────────────────
        phase1_tokens = [agent.get_and_reset_token_counts_by_type() for agent in agents]

        # ── Phase 2: QA ──────────────────────────────────────────────────────
        qa_results_list, phase2_stats = self._run_phase2_batched(
            samples, agents, retrieval_log_paths
        )

        # ── Compile per-sample results ────────────────────────────────────────
        results = []
        for i, (sample, agent) in enumerate(zip(samples, agents)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snap_dir = snapshots_dir / f"sample_{sample.sample_id}"
                agent.save_snapshot(snap_dir)
                logger.info(f"Memory snapshot saved: sample {sample.sample_id}")

            agent.clear()
            logger.info(f"Memory cleared: sample {sample.sample_id}")

            p1 = phase1_tokens[i]
            p2 = phase2_stats[i]

            c1 = p1["call_1_speaker1_persona"]
            c2 = p1["call_2_speaker2_persona"]
            c3 = p1["call_3_summarization"]
            c4 = {"input": p2["qa_input"], "output": p2["qa_output"], "llm_calls": p2["num_qa_calls"]}

            total_input     = c1["input"]  + c2["input"]  + c3["input"]  + c4["input"]
            total_output    = c1["output"] + c2["output"] + c3["output"] + c4["output"]
            total_llm_calls = c1["llm_calls"] + c2["llm_calls"] + c3["llm_calls"] + c4["llm_calls"]

            results.append({
                "sample_id":      sample.sample_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": memory_stats_list[i],
                "qa_results":     qa_results_list[i],
                "token_statistics": {
                    "call_1_speaker1_persona": c1,
                    "call_2_speaker2_persona": c2,
                    "call_3_summarization":    c3,
                    "call_4_qa":               c4,
                    "total_input":             total_input,
                    "total_output":            total_output,
                    "total_llm_calls":         total_llm_calls,
                },
                "memory_snapshot_path": (
                    f"memory_snapshots/sample_{sample.sample_id}"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            })

        return results

    # =========================================================================
    # Phase 1 — batched across samples
    # =========================================================================

    def _run_phase1_batched(
        self, samples: List[Sample], agents: List[LDAgentModule]
    ) -> None:
        """
        Process all turns in lock-step across active samples.
        Token counts are accumulated directly on each agent.
        """
        turns_per_sample = [flatten_sample_turns(sample) for sample in samples]
        max_turns = max((len(t) for t in turns_per_sample), default=0)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns", ncols=100):
            active = []
            for i, sample in enumerate(samples):
                if turn_idx < len(turns_per_sample[i]):
                    active.append((i, sample, agents[i], turns_per_sample[i][turn_idx]))
            self._process_turn_batch(active)

    def _process_turn_batch(
        self, active: List[Tuple[int, Sample, LDAgentModule, Dict]]
    ):
        """
        Process one turn slot for all active samples:
          1. Batch flush (samples needing session-boundary flush) → call_3_summarization
          2. Append turn to STM (all active, no LLM)
          3. Batch persona extraction (all active) → call_1 or call_2
        """
        if not active:
            return

        # ── Step 1: flush samples that need session-boundary flush ────────────
        flush_jobs = []
        for i_sample, sample, agent, turn in active:
            mb = agent.memory_bank
            if mb.should_flush(turn["timestamp"]):
                flush_ctx = mb.build_flush_context()
                flush_jobs.append((i_sample, sample, agent, turn, flush_ctx))

        if flush_jobs:
            flush_prompts = [job[4]["combined_prompt"] for job in flush_jobs]
            flush_results, flush_usages, _ = self._batch_generate_with_retry(
                prompts=flush_prompts,
                system_prompt=None,
                max_tokens=100,
                temperature=0.7,
                guided_json=SUMMARY_SCHEMA,
            )
            for (i_sample, sample, agent, turn, flush_ctx), result, usage in zip(
                flush_jobs, flush_results, flush_usages
            ):
                summary = result.get("summary", "") if isinstance(result, dict) else ""
                agent.memory_bank.apply_flush(summary, usage, flush_ctx, clear_stm=True)
                agent.accumulate_tokens(
                    "call_3_summarization",
                    usage.get("prompt_tokens",     0),
                    usage.get("completion_tokens", 0),
                )

        # ── Step 2: append turn to STM (no LLM call) ─────────────────────────
        for i_sample, sample, agent, turn in active:
            agent.memory_bank.append_turn_to_stm(
                speaker_name=turn["speaker"],
                text=        turn["text"],
                dia_id=      turn["dia_id"],
                timestamp=   turn["timestamp"],
                session_num= turn["session_num"],
                date_time=   turn["date_time"],
                sample_id=   turn["sample_id"],
            )

        # ── Step 3: persona extraction ────────────────────────────────────────
        persona_jobs = []
        for i_sample, sample, agent, turn in active:
            _, usr_p = agent.personas.build_trait_prompt(turn["text"])
            persona_jobs.append((i_sample, sample, agent, turn, usr_p))

        persona_prompts = [job[4] for job in persona_jobs]
        trait_results, trait_usages, _ = self._batch_generate_with_retry(
            prompts=persona_prompts,
            system_prompt=TRAIT_SYS_PROMPT,
            max_tokens=100,
            temperature=0.7,
            guided_json=TRAIT_SCHEMA,
        )
        for (i_sample, sample, agent, turn, usr_p), result, usage in zip(
            persona_jobs, trait_results, trait_usages
        ):
            per = agent.personas
            if turn["speaker"] == turn["speaker_a"]:
                per.apply_speaker1_trait_result(result, usage)
                agent.accumulate_tokens(
                    "call_1_speaker1_persona",
                    usage.get("prompt_tokens",     0),
                    usage.get("completion_tokens", 0),
                )
            else:
                per.apply_speaker2_trait_result(result, usage)
                agent.accumulate_tokens(
                    "call_2_speaker2_persona",
                    usage.get("prompt_tokens",     0),
                    usage.get("completion_tokens", 0),
                )

    # =========================================================================
    # Final flush before QA
    # =========================================================================

    def _run_final_flush_batched(
        self, samples: List[Sample], agents: List[LDAgentModule]
    ) -> None:
        """
        Commit any remaining STM content to LTM for each agent.
        STM is kept (clear_stm=False) so it remains available for QA context.
        Tokens accumulated directly on each agent as call_3_summarization.
        """
        flush_jobs = []
        for i, (sample, agent) in enumerate(zip(samples, agents)):
            mb = agent.memory_bank
            if mb.short_term_memory:
                flush_ctx = mb.build_flush_context()
                flush_jobs.append((i, sample, agent, flush_ctx))

        if not flush_jobs:
            return

        logger.info(
            f"Final flush: {len(flush_jobs)}/{len(samples)} samples have pending STM"
        )

        flush_prompts = [job[3]["combined_prompt"] for job in flush_jobs]
        flush_results, flush_usages, _ = self._batch_generate_with_retry(
            prompts=flush_prompts,
            system_prompt=None,
            max_tokens=100,
            temperature=0.7,
            guided_json=SUMMARY_SCHEMA,
        )
        for (i, sample, agent, flush_ctx), result, usage in zip(
            flush_jobs, flush_results, flush_usages
        ):
            summary = result.get("summary", "") if isinstance(result, dict) else ""
            agent.memory_bank.apply_flush(summary, usage, flush_ctx, clear_stm=False)
            agent.accumulate_tokens(
                "call_3_summarization",
                usage.get("prompt_tokens",     0),
                usage.get("completion_tokens", 0),
            )

    # =========================================================================
    # Phase 2 — batched QA
    # =========================================================================

    def _run_phase2_batched(
        self,
        samples:             List[Sample],
        agents:              List[LDAgentModule],
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """
        Run QA for all samples. Retrieval is sequential; LLM inference is batched
        per temperature group.

        Returns:
            qa_results_per_sample  — list[list[dict]] ordered by (sample, qa_idx)
            phase2_stats           — list of {"qa_input", "qa_output", "num_qa_calls"}
        """
        qa_jobs: List[Dict] = []

        for sample_idx, (sample, agent) in enumerate(zip(samples, agents)):
            mb  = agent.memory_bank
            per = agent.personas
            gen = agent.generator

            stm_context       = mb.get_stm_context()
            context_str       = format_context_for_prompt(stm_context)
            speaker1_traits, speaker2_traits = per.get_current_traits()
            stm_turns         = len(stm_context)
            trait_a_count     = _count_traits(speaker1_traits)
            trait_b_count     = _count_traits(speaker2_traits)

            for qa_idx, qa in enumerate(sample.qa):
                relevant_memories = mb.relevance_retrieve(
                    ori_query=qa.question,
                    n_results=cfg.RETRIEVE_K,
                    current_timestamp=mb.current_timestamp,
                )
                memories_str = format_memories_for_prompt(relevant_memories)

                # Build retrieval log entry
                log_retrieved_items = []
                for mem in relevant_memories:
                    dia_ids_str = mem.get("dia_ids", "")
                    dia_ids     = [d for d in dia_ids_str.split(",") if d] if dia_ids_str else []
                    log_retrieved_items.append({
                        "dia_ids":         dia_ids,
                        "content_preview": (mem.get("summary") or "")[:100],
                        "score":           mem.get("score", 0.0),
                        "source": {
                            "session_num": mem.get("session_num", 0),
                            "timestamp":   mem.get("timestamp",   0.0),
                            "date_time":   mem.get("date_time",   ""),
                        },
                    })

                module_specific = {
                    "ltm_entry_count":      mb.get_memory_count(),
                    "stm_context_turns":    stm_turns,
                    "speaker1_trait_count": trait_a_count,
                    "speaker2_trait_count": trait_b_count,
                }
                write_retrieval_log(
                    retrieval_log_paths[sample_idx],
                    {
                        "timestamp":     datetime.now().isoformat(),
                        "phase":         "qa",
                        "sample_id":     sample.sample_id,
                        "session_id":    None,
                        "dia_id":        None,
                        "query":         qa.question,
                        "memory_type":   ["ltm", "stm", "speaker1_trait", "speaker2_trait"],
                        "num_retrieved": [
                            len(log_retrieved_items),
                            stm_turns,
                            trait_a_count,
                            trait_b_count,
                        ],
                        "retrieved_items": log_retrieved_items,
                        "module_specific": module_specific,
                    },
                )

                choice_seed = f"{sample.sample_id}::{qa_idx}::{qa.question}"
                prompt, temperature = gen.build_qa_prompt_for_batch(
                    question=           qa.question,
                    category=           qa.category,
                    context=            context_str,
                    memories=           memories_str,
                    speaker1_traits=    speaker1_traits,
                    speaker2_traits=    speaker2_traits,
                    adversarial_answer= qa.adversarial_answer,
                    choice_order_seed=  choice_seed,
                )

                retrieved_memories_for_result = []
                for mem in relevant_memories:
                    dia_ids_str = mem.get("dia_ids", "")
                    dia_ids     = [d for d in dia_ids_str.split(",") if d] if dia_ids_str else []
                    summary     = mem.get("summary", "")
                    score       = mem.get("score", 0.0)
                    if dia_ids:
                        for did in dia_ids:
                            retrieved_memories_for_result.append({
                                "dia_id":          did,
                                "content_preview": summary[:100],
                                "score":           score,
                            })
                    else:
                        retrieved_memories_for_result.append({
                            "dia_id":          "",
                            "content_preview": summary[:100],
                            "score":           score,
                        })

                qa_jobs.append({
                    "sample_idx":         sample_idx,
                    "qa_idx":             qa_idx,
                    "sample_id":          sample.sample_id,
                    "qa":                 qa,
                    "prompt":             prompt,
                    "temperature":        temperature,
                    "choice_seed":        choice_seed,
                    "retrieved_memories": retrieved_memories_for_result,
                })

        # ── Initialise per-sample accumulators ────────────────────────────────
        qa_results_per_sample: List[List[Dict]] = [[] for _ in samples]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0}
            for _ in samples
        ]

        if not qa_jobs:
            return qa_results_per_sample, phase2_stats

        # ── Group by temperature and batch ────────────────────────────────────
        jobs_by_temp: Dict[float, List[Dict]] = {}
        for job in qa_jobs:
            jobs_by_temp.setdefault(job["temperature"], []).append(job)

        all_outputs: List[Tuple] = []

        for temperature, jobs in jobs_by_temp.items():
            for chunk_start in tqdm(
                range(0, len(jobs), cfg.QA_BATCH_SIZE),
                desc=f"Phase2 QA temp={temperature}",
                ncols=100,
            ):
                chunk         = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
                chunk_prompts = [job["prompt"] for job in chunk]

                def retry_single(local_idx: int, _prompt: str):
                    job      = chunk[local_idx]
                    agent    = agents[job["sample_idx"]]
                    qa_result = agent.get_qa_answer(
                        question=           job["qa"].question,
                        category=           job["qa"].category,
                        adversarial_answer= job["qa"].adversarial_answer,
                    )
                    return (
                        {"answer": qa_result.answer},
                        {
                            "prompt_tokens":     qa_result.token_info.get("input",  0),
                            "completion_tokens": qa_result.token_info.get("output", 0),
                        },
                        True,
                        "",
                    )

                chunk_results, chunk_usages, chunk_logged = self._batch_generate_with_retry(
                    prompts=            chunk_prompts,
                    system_prompt=      None,
                    max_tokens=         cfg.MAX_TOKENS,
                    temperature=        temperature,
                    guided_json=        QA_RESPONSE_SCHEMA,
                    sequential_retry_fn=retry_single,
                )

                for job, result, usage, already_logged in zip(
                    chunk, chunk_results, chunk_usages, chunk_logged
                ):
                    all_outputs.append((job, result, usage, already_logged))

        # ── Sort by (sample_idx, qa_idx) and write results ────────────────────
        all_outputs.sort(key=lambda item: (item[0]["sample_idx"], item[0]["qa_idx"]))

        for job, result, usage, already_logged in all_outputs:
            sample_idx = job["sample_idx"]

            if not already_logged and agents[sample_idx]._llm_logger is not None:
                agents[sample_idx]._llm_logger.log(
                    "call_4_qa", "", job["prompt"], result
                )

            phase2_stats[sample_idx]["qa_input"]    += usage["prompt_tokens"]
            phase2_stats[sample_idx]["qa_output"]   += usage["completion_tokens"]
            phase2_stats[sample_idx]["num_qa_calls"] += 1

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            qa_results_per_sample[sample_idx].append({
                "question":            job["qa"].question,
                "category":            job["qa"].category,
                "generated_answer":    answer,
                "ground_truth_answer": job["qa"].final_answer,
                "evidence":            job["qa"].evidence,
                "retrieved_memories":  job["retrieved_memories"],
                "qa_tokens": {
                    "input":  usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "model":  self.model_path,
                },
            })

        return qa_results_per_sample, phase2_stats

    # =========================================================================
    # Batch LLM helper
    # =========================================================================

    def _batch_generate_with_retry(
        self,
        prompts:              List[str],
        system_prompt:        Optional[str],
        max_tokens:           int,
        temperature:          float,
        guided_json=          None,
        sequential_retry_fn=  None,
    ) -> Tuple[List, List[Dict], List[bool]]:
        """
        Wrapper around generate_batch_raw() with JSON-parse retry.

        Returns (results, usages, already_logged).
        """
        if not prompts:
            return [], [], []

        try:
            texts, usages = self.llm_client.generate_batch_raw(
                prompts=       prompts,
                system_prompt= system_prompt,
                max_tokens=    max_tokens,
                temperature=   temperature,
                guided_json=   guided_json,
                return_usage=  True,
            )
        except Exception as e:
            if sequential_retry_fn is None:
                raise
            logger.warning(
                f"Batch generation failed; falling back to sequential: {e}"
            )
            results, usages_out, already_logged = [], [], []
            for idx, prompt in enumerate(prompts):
                result, usage, logged, _ = sequential_retry_fn(idx, prompt)
                results.append(result)
                usages_out.append(usage)
                already_logged.append(logged)
            return results, usages_out, already_logged

        already_logged = [False] * len(prompts)
        if guided_json is None:
            return texts, usages, already_logged

        from llm_client import _parse_json_response

        parsed     = []
        retry_idxs = []
        for idx, text in enumerate(texts):
            try:
                parsed.append(
                    _parse_json_response(text) if isinstance(text, str) else text
                )
            except (json.JSONDecodeError, ValueError):
                parsed.append(None)
                retry_idxs.append(idx)

        for idx in retry_idxs:
            logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
            if sequential_retry_fn is not None:
                result, usage, logged, _ = sequential_retry_fn(idx, prompts[idx])
                parsed[idx]         = result
                usages[idx]         = usage
                already_logged[idx] = logged
                continue
            try:
                retry_result = self.llm_client.generate(
                    prompt=        prompts[idx],
                    system_prompt= system_prompt,
                    guided_json=   guided_json,
                    temperature=   temperature,
                    max_tokens=    max_tokens,
                    json_retry=    cfg.JSON_RETRY,
                    return_usage=  True,
                )
                if isinstance(retry_result, dict):
                    usage_info  = retry_result.pop("_usage", {})
                    usages[idx] = {
                        "prompt_tokens":     usage_info.get("prompt_tokens",     0),
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
# LLM CLIENT FACTORY
# =============================================================================

def create_llm_client(model_path: str, tensor_parallel: int, gpu_memory: float,
                      max_model_len: Optional[int] = None):
    from llm_client import create_llm_client as _create

    engine = cfg.LLM_ENGINE
    if engine == "vllm":
        vllm_kwargs = dict(
            engine=                 "vllm",
            model_path=             model_path,
            tensor_parallel_size=   tensor_parallel,
            gpu_memory_utilization= gpu_memory,
            download_dir=           None,
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

    parser = argparse.ArgumentParser(
        description="LD-Agent Batch Experiment on LoComo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-sample",    type=int, required=True)
    parser.add_argument("--end-sample",      type=int, required=True)
    parser.add_argument("--model",           type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory",      type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len",   type=int, default=None)
    parser.add_argument("--batch-size",      type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config",          type=str, default="config_0")
    parser.add_argument("--max-tokens",      type=int, default=None,
                        help="Override MAX_TOKENS from config")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_sample > args.end_sample:
        parser.error("--start-sample must be <= --end-sample")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.max_tokens is not None:
        cfg.MAX_TOKENS = args.max_tokens

    cfg.ensure_directories(args.model, args.start_sample, args.end_sample, config_name)

    sample_dir        = cfg.get_sample_dir(args.model, args.start_sample, args.end_sample, config_name)
    results_file      = cfg.get_results_file(args.model, args.start_sample, args.end_sample, config_name)
    checkpoint_file   = cfg.get_checkpoint_file(args.model, args.start_sample, args.end_sample, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.start_sample, args.end_sample, config_name)
    snapshots_dir     = cfg.get_memory_snapshots_dir(args.model, args.start_sample, args.end_sample, config_name)
    prompt_log_dir    = cfg.get_prompt_log_dir(args.model, args.start_sample, args.end_sample, config_name)

    nohup_dir = _MODULE_DIR / "nohup"
    nohup_dir.mkdir(parents=True, exist_ok=True)

    global logger
    logger = setup_logging(sample_dir / "logs")

    logger.info("=" * 60)
    logger.info("LD-Agent Batch Experiment — LoComo")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Samples         : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  MAX_TOKENS      : {cfg.MAX_TOKENS}")
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

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    completed_sample_ids: Set[str] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_sample_ids, last_completed_index = load_checkpoint(checkpoint_file)
        if last_completed_index is not None:
            completed_sample_ids.update(
                str(target_samples[i].sample_id)
                for i in range(min(last_completed_index + 1, len(target_samples)))
            )
        if completed_sample_ids:
            logger.info(
                f"Resuming: {len(completed_sample_ids)} samples already completed"
            )

    pending_samples = [
        sample for sample in target_samples
        if str(sample.sample_id) not in completed_sample_ids
    ]
    if not pending_samples:
        logger.info("All samples already completed.")
        return 0

    # ── LLM client ────────────────────────────────────────────────────────────
    logger.info("Initialising LLM client...")
    try:
        llm_client = create_llm_client(
            args.model, args.tensor_parallel, args.gpu_memory,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        logger.error(f"Failed to initialise LLM client: {e}")
        return 1
    logger.info("LLM client ready.")

    # ── Shared encoder and lemma tokenizer (created once, reused across agents) ──
    logger.info("Loading shared encoder and lemma tokenizer...")
    import spacy
    from sentence_transformers import SentenceTransformer
    shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        shared_lemma_tokenizer = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        shared_lemma_tokenizer = spacy.load("en_core_web_sm")
    logger.info("Shared encoder and lemma tokenizer ready.")

    # ── Config metadata ───────────────────────────────────────────────────────
    config_metadata = {
        "config_name":               config_name,
        "model":                     args.model,
        "embedding_model":           "all-MiniLM-L6-v2",
        "temperature":               cfg.TEMPERATURE,
        "temperature_c5":            cfg.TEMPERATURE_C5,
        "max_tokens":                cfg.MAX_TOKENS,
        "sample_range":              [args.start_sample, args.end_sample],
        "retrieve_k":                cfg.RETRIEVE_K,
        "dist_threshold":            cfg.DIST_THRESHOLD,
        "relevance_memory_number":   cfg.RELEVANCE_MEMORY_NUMBER,
        "stm_flush_gap_seconds":     cfg.STM_FLUSH_GAP_SECONDS,
        "decay_temp":                cfg.DECAY_TEMP,
        "max_speaker1_personas":      cfg.MAX_SPEAKER_A_PERSONAS,
        "max_speaker2_personas":      cfg.MAX_SPEAKER_B_PERSONAS,
    }

    runner  = BatchedLDAgentRunner(
        llm_client=llm_client,
        model_path=args.model,
        config_metadata=config_metadata,
        encoder=shared_encoder,
        lemma_tokenizer=shared_lemma_tokenizer,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# LD-Agent Batch  |  LoComo")
    print(f"# Model   : {cfg.extract_model_name(args.model)}")
    print(f"# Samples [{args.start_sample}, {args.end_sample}]  "
          f"({len(pending_samples)} to process)")
    print(f"# Batch size : {args.batch_size}")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_samples) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        batch      = pending_samples[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
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
                samples=             batch,
                retrieval_log_paths= batch_retrieval_log_paths,
                snapshots_dir=       snapshots_dir,
                prompt_log_dirs=     batch_prompt_log_dirs,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        for result in batch_results:
            results.append(result)
            save_results(results_file, results)
            if cfg.ENABLE_CHECKPOINTING:
                completed_sample_ids.add(str(result["sample_id"]))
                save_checkpoint(
                    checkpoint_file=      checkpoint_file,
                    completed_sample_ids= completed_sample_ids,
                    model_path=           args.model,
                    start_sample=         args.start_sample,
                    end_sample=           args.end_sample,
                    config_name=          config_name,
                )
            ts = result["token_statistics"]
            logger.info(
                f"Sample {result['sample_id']} done.  "
                f"QA: {len(result['qa_results'])},  "
                f"Total LLM calls: {ts['total_llm_calls']}"
            )

    print(f"\n{'#' * 60}")
    print(f"# Batch experiment complete!  Results -> {results_file}")
    print(f"{'#' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
