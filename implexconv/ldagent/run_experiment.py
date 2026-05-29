"""
LD-Agent Experiment Runner (ImplexConv) — Batched Main Variant, QA-Only

Runs multiple sessions in parallel within a single GPU by batching together
LD-Agent's internal LLM calls. Sessions remain memory-isolated, but prompts
from the same execution stage are grouped into one batched vLLM call.

Batch execution flow:
  Phase 1 (turn-aligned across sessions):
    1. Boundary STM->LTM summarization if needed
    2. Append current user turn to STM
    3. Retrieve STM/LTM context per session
    4. Batch user-persona extraction
    5. Batch agent-persona extraction using GT response
    6. Store GT response in STM

  Phase 2 (after memory is frozen):
    1. Batch flush remaining STM to LTM
    2. Batch QA answer generation in QA_BATCH_SIZE chunks

Changes from v3:
  - import time removed; all timing tracking removed
  - _avg_timing() removed
  - token_statistics restructured to per-call-type breakdown:
      call_2_user_persona, call_3_agent_persona, call_4_summarization, call_5_qa
  - api_calls → llm_calls throughout
  - timing_statistics removed from result
  - config_metadata added to result
  - memory_at_qa_start added to result
  - response_input tracking removed (build_response_prompt_only no longer called)
  - timestamp removed from retrieval log entries and LLM call log entries
  - ChromaDB collection.count() → memory_bank.get_memory_count()
  - shared_encoder (SentenceTransformer) loaded once and shared across sessions
"""

import os
import re
import sys
import json
import tempfile
import logging
import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore
logger: logging.Logger = logging.getLogger(__name__)

from load_dataset import load_implexconv_dataset, Session, compute_virtual_seconds
from ldagent_module import LDAgentModule, LLMCallLogger
from generator import format_memories_for_prompt
from personas import TRAIT_SCHEMA
from event_memory import SUMMARY_GUIDED_JSON


# =============================================================================
# LOGGING
# =============================================================================


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        from datetime import datetime
        log_file = log_dir / f"ldagent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================


def load_checkpoint(checkpoint_file: Path) -> Set[int]:
    if not checkpoint_file.exists():
        return set()
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        if "completed_session_ids" in data:
            return set(data["completed_session_ids"])
        last = data.get("last_completed_session_index")
        if last is not None:
            return set(range(last + 1))
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
    return set()


def save_checkpoint(
    checkpoint_file: Path,
    completed_ids: Set[int],
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str,
):
    data = {
        "completed_session_ids": sorted(completed_ids),
        "config": {
            "model": model_path,
            "subset": subset,
            "start_session": start_session,
            "end_session": end_session,
            "config_name": config_name,
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
        logger.warning(f"Could not load results ({e}); starting fresh.")
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


def print_completion_banner(results_file: Path, total_sessions: int):
    banner = "#" * 60
    print(f"\n{banner}")
    print(f"# Experiment completed — {total_sessions} sessions processed")
    print(f"# Results: {results_file}")
    print(banner)
    print(banner)
    print()


# =============================================================================
# RETRIEVAL LOG
# =============================================================================


class RetrievalLogWriter:
    """Appends JSONL retrieval log entries to a per-session file."""

    def __init__(self, log_path: Path):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: Dict):
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _build_retrieval_log_entry(
    phase: str,
    session_id: int,
    conv_id: int,
    turn_id: int,
    retrieval_data: Dict,
) -> Dict:
    relevant_memories = retrieval_data.get("relevant_memories", [])

    retrieved_items = []
    for mem in relevant_memories:
        retrieved_items.append({
            "memory_id": str(mem.get("idx", "")),
            "content_preview": (mem.get("summary") or mem.get("dialog") or "")[:100],
            "score": mem.get("score", 0.0),
            "source_turn": {
                "session_id": mem.get("session_id", 0),
                "conv_id": mem.get("conv_id", 0),
                "virtual_seconds": mem.get("virtual_seconds", 0.0),
            },
        })

    module_specific = retrieval_data.get("module_specific", {})
    return {
        "phase": phase,
        "session_id": session_id,
        "conv_id": conv_id,
        "turn_id": turn_id,
        "query": retrieval_data.get("query", ""),
        "memory_type": ["ltm", "stm", "user_trait", "agent_trait"],
        "num_retrieved": [
            len(retrieved_items),
            module_specific.get("stm_context_turns", 0),
            module_specific.get("user_trait_count", 0),
            module_specific.get("agent_trait_count", 0),
        ],
        "retrieved_items": retrieved_items,
        "module_specific": module_specific,
    }


# =============================================================================
# HELPERS
# =============================================================================


def _normalize_yes_no_unknown(answer: str) -> str:
    a = answer.strip().lower()
    if a in ("yes", "no", "unknown"):
        return a
    if re.match(r"^yes\b", a):
        return "yes"
    if re.match(r"^no\b", a):
        return "no"
    return "unknown"


def _count_traits(traits_str: str) -> int:
    return len([t for t in traits_str.split("\n") if t.strip()]) if traits_str else 0


def _usage_dict(usage_info: Optional[Dict[str, int]]) -> Dict[str, int]:
    usage_info = usage_info or {}
    return {
        "prompt_tokens": usage_info.get("prompt_tokens", 0),
        "completion_tokens": usage_info.get("completion_tokens", 0),
    }


def _attach_usage(result: Any, usage: Dict[str, int]):
    if isinstance(result, dict):
        enriched = dict(result)
        enriched["_usage"] = usage
        return enriched
    return {"_usage": usage}


def _add_tokens(bucket: Dict, prompt_tokens: int, completion_tokens: int):
    """Accumulate tokens into a per-call-type bucket."""
    bucket["input"]     += prompt_tokens
    bucket["output"]    += completion_tokens
    bucket["llm_calls"] += 1


# =============================================================================
# BATCHED RUNNER
# =============================================================================


class BatchedLDAgentRunner:
    def __init__(
        self,
        llm_client,
        model_path: str,
        subset: str,
        start_session: int,
        end_session: int,
        config_name: str,
        config_metadata: Dict,
        shared_lemma_tokenizer=None,
        shared_encoder=None,
    ):
        self.llm_client = llm_client
        self.model_path = model_path
        self.subset = subset
        self.start_session = start_session
        self.end_session = end_session
        self.config_name = config_name
        self.config_metadata = config_metadata
        self.shared_lemma_tokenizer = shared_lemma_tokenizer
        self.shared_encoder = shared_encoder

        self.retrieval_log_dir = cfg.get_retrieval_log_dir(
            model_path, subset, start_session, end_session, config_name
        )
        self.memory_snapshots_dir = cfg.get_memory_snapshots_dir(
            model_path, subset, start_session, end_session, config_name
        )
        self.prompt_log_dir = cfg.get_prompt_log_dir(
            model_path, subset, start_session, end_session, config_name
        )

    # -------------------------------------------------------------------------
    # Batch generation
    # -------------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompt_items: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        guided_json=None,
    ) -> Tuple[List[Any], List[Dict[str, int]]]:
        if not prompt_items:
            return [], []

        inner = getattr(self.llm_client, "client", self.llm_client)

        if hasattr(inner, "llm") and hasattr(inner, "_build_prompt") and hasattr(inner, "_get_json_schema"):
            from llm_client import (
                SamplingParams,
                _make_guided_sampling_kwargs,
                _parse_json_response,
                _strip_think_tags,
            )

            json_schema = inner._get_json_schema(guided_json)
            full_prompts = [
                inner._build_prompt(
                    prompt=item["user_prompt"],
                    messages=None,
                    system_prompt=item.get("system_prompt"),
                    use_json_mode=guided_json is not None,
                    schema=json_schema,
                )
                for item in prompt_items
            ]

            sampling_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
            sampling_kwargs.update(_make_guided_sampling_kwargs(json_schema))
            sampling_params = SamplingParams(**sampling_kwargs)

            outputs = inner.llm.generate(full_prompts, sampling_params, use_tqdm=False)

            texts = []
            usages = []
            for output in outputs:
                text = ""
                if output.outputs:
                    text = _strip_think_tags(output.outputs[0].text.strip())
                texts.append(text)
                usages.append({
                    "prompt_tokens": len(output.prompt_token_ids) if hasattr(output, "prompt_token_ids") else 0,
                    "completion_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
                })

            if guided_json is None:
                return texts, usages

            parsed = []
            retry_indices = []
            for idx, text in enumerate(texts):
                try:
                    parsed.append(_parse_json_response(text))
                except (json.JSONDecodeError, ValueError):
                    parsed.append(None)
                    retry_indices.append(idx)

            for idx in retry_indices:
                logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
                try:
                    retry_result = self.llm_client.generate(
                        prompt=prompt_items[idx]["user_prompt"],
                        system_prompt=prompt_items[idx].get("system_prompt"),
                        guided_json=guided_json,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_retry=cfg.JSON_RETRY,
                        return_usage=True,
                    )
                    if isinstance(retry_result, dict):
                        usage_info = _usage_dict(retry_result.pop("_usage", {}))
                        usages[idx] = usage_info
                        parsed[idx] = retry_result
                    else:
                        parsed[idx] = {}
                except Exception as e:
                    logger.error(f"Sequential retry for batch item {idx} failed: {e}")
                    parsed[idx] = {}

            return parsed, usages

        results = []
        usages = []
        for item in prompt_items:
            result = self.llm_client.generate(
                prompt=item["user_prompt"],
                system_prompt=item.get("system_prompt"),
                guided_json=guided_json,
                temperature=temperature,
                max_tokens=max_tokens,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
            if guided_json is None:
                if isinstance(result, dict):
                    usages.append(_usage_dict(result.get("_usage", {})))
                    results.append(result.get("content", ""))
                else:
                    usages.append(_usage_dict(None))
                    results.append(str(result))
            else:
                if isinstance(result, dict):
                    usages.append(_usage_dict(result.pop("_usage", {})))
                    results.append(result)
                else:
                    usages.append(_usage_dict(None))
                    results.append({})
        return results, usages

    # -------------------------------------------------------------------------
    # Phase 1
    # -------------------------------------------------------------------------

    def _run_phase1_batched(
        self,
        sessions: List[Session],
        agents: List[LDAgentModule],
        log_writers: List[RetrievalLogWriter],
    ) -> List[Dict]:
        """
        Returns per-session Phase 1 token stats, keyed by call type:
          call_2_user_persona, call_3_agent_persona, call_4_summarization
        """
        phase1_stats = [
            {
                "call_2_user_persona":  {"input": 0, "output": 0, "llm_calls": 0},
                "call_3_agent_persona": {"input": 0, "output": 0, "llm_calls": 0},
                "call_4_summarization": {"input": 0, "output": 0, "llm_calls": 0},
            }
            for _ in sessions
        ]
        turn_pairs_list = [session.get_turn_pairs() for session in sessions]
        max_turns = max((len(pairs) for pairs in turn_pairs_list), default=0)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active = []
            for i, (session, agent, turn_pairs) in enumerate(zip(sessions, agents, turn_pairs_list)):
                if turn_idx >= len(turn_pairs):
                    continue
                user_turn, assistant_turn = turn_pairs[turn_idx]
                if user_turn is None:
                    continue
                active.append({
                    "session_idx": i,
                    "session": session,
                    "agent": agent,
                    "user_turn": user_turn,
                    "assistant_turn": assistant_turn,
                    "virtual_seconds": compute_virtual_seconds(
                        user_turn.conv_id,
                        user_turn.turn_id,
                        cfg.CONV_IDS_PER_DAY,
                        cfg.MINUTES_PER_TURN,
                    ),
                })

            if not active:
                continue

            # --- Boundary summarization ---
            boundary_batch = []
            for item in active:
                memory_bank = item["agent"].memory_bank
                summary_job = memory_bank.prepare_boundary_summary(
                    current_virtual_seconds=item["virtual_seconds"],
                    current_conv_id=item["user_turn"].conv_id,
                    current_session_id=item["session"].session_id,
                )
                if summary_job is not None:
                    boundary_batch.append({
                        "session_idx": item["session_idx"],
                        "agent": item["agent"],
                        "summary_job": summary_job,
                        "prompt_item": {
                            "system_prompt": summary_job["system_prompt"],
                            "user_prompt": summary_job["user_prompt"],
                        },
                    })

            if boundary_batch:
                results, usages = self._batch_generate_with_retry(
                    prompt_items=[entry["prompt_item"] for entry in boundary_batch],
                    max_tokens=100,
                    temperature=0.7,
                    guided_json=SUMMARY_GUIDED_JSON,
                )
                for entry, result, usage in zip(boundary_batch, results, usages):
                    agent = entry["agent"]
                    enriched = _attach_usage(result, usage)
                    if agent.memory_bank.llm_logger is not None:
                        agent.memory_bank.llm_logger.log(
                            "call_4_summarization",
                            entry["summary_job"]["system_prompt"],
                            entry["summary_job"]["user_prompt"],
                            enriched,
                        )
                    agent.memory_bank.set_last_summarize_usage(
                        usage["prompt_tokens"], usage["completion_tokens"]
                    )
                    summary_text = result.get("summary", "") if isinstance(result, dict) else ""
                    agent.memory_bank.apply_boundary_summary_result(entry["summary_job"], summary_text)
                    _add_tokens(
                        phase1_stats[entry["session_idx"]]["call_4_summarization"],
                        usage["prompt_tokens"], usage["completion_tokens"],
                    )

            # --- User persona extraction ---
            turn_jobs = []
            user_persona_batch = []
            for item in active:
                i = item["session_idx"]
                agent = item["agent"]
                session = item["session"]
                user_turn = item["user_turn"]
                memory_bank = agent.memory_bank
                personas = agent.personas

                context_memories = memory_bank.append_user_query(
                    query=user_turn.utterance,
                    current_virtual_seconds=item["virtual_seconds"],
                    current_conv_id=user_turn.conv_id,
                    current_session_id=user_turn.session_id,
                )
                sys_prompt, user_prompt = personas.build_user_trait_prompt(user_turn.utterance)

                turn_job = {
                    "session_idx": i,
                    "session": session,
                    "agent": agent,
                    "user_turn": user_turn,
                    "assistant_turn": item["assistant_turn"],
                    "virtual_seconds": item["virtual_seconds"],
                    "context_memories": context_memories,
                }
                turn_jobs.append(turn_job)
                user_persona_batch.append({
                    "turn_job": turn_job,
                    "prompt_item": {
                        "system_prompt": sys_prompt,
                        "user_prompt": user_prompt,
                    },
                })

            if user_persona_batch:
                results, usages = self._batch_generate_with_retry(
                    prompt_items=[entry["prompt_item"] for entry in user_persona_batch],
                    max_tokens=100,
                    temperature=0.7,
                    guided_json=TRAIT_SCHEMA,
                )
                for entry, result, usage in zip(user_persona_batch, results, usages):
                    turn_job = entry["turn_job"]
                    agent = turn_job["agent"]
                    enriched = _attach_usage(result, usage)
                    if agent.personas.llm_logger is not None:
                        agent.personas.llm_logger.log(
                            "call_2_user_persona",
                            entry["prompt_item"]["system_prompt"],
                            entry["prompt_item"]["user_prompt"],
                            enriched,
                        )
                    agent.personas.apply_user_trait_result(enriched)
                    _add_tokens(
                        phase1_stats[turn_job["session_idx"]]["call_2_user_persona"],
                        usage["prompt_tokens"], usage["completion_tokens"],
                    )

            # --- Agent persona extraction + retrieval logging ---
            agent_persona_batch = []
            for turn_job in turn_jobs:
                agent = turn_job["agent"]
                personas = agent.personas
                gt_response = ""
                if turn_job["assistant_turn"] is not None:
                    gt_response = turn_job["assistant_turn"].utterance

                user_traits, agent_traits = personas.get_current_traits()
                sys_prompt, user_prompt = personas.build_agent_trait_prompt(gt_response)
                agent_persona_batch.append({
                    "turn_job": turn_job,
                    "gt_response": gt_response,
                    "prompt_item": {
                        "system_prompt": sys_prompt,
                        "user_prompt": user_prompt,
                    },
                })

            if agent_persona_batch:
                results, usages = self._batch_generate_with_retry(
                    prompt_items=[entry["prompt_item"] for entry in agent_persona_batch],
                    max_tokens=100,
                    temperature=0.7,
                    guided_json=TRAIT_SCHEMA,
                )
                for entry, result, usage in zip(agent_persona_batch, results, usages):
                    turn_job = entry["turn_job"]
                    agent = turn_job["agent"]
                    enriched = _attach_usage(result, usage)
                    if agent.personas.llm_logger is not None:
                        agent.personas.llm_logger.log(
                            "call_3_agent_persona",
                            entry["prompt_item"]["system_prompt"],
                            entry["prompt_item"]["user_prompt"],
                            enriched,
                        )
                    agent.personas.apply_agent_trait_result(enriched)
                    _add_tokens(
                        phase1_stats[turn_job["session_idx"]]["call_3_agent_persona"],
                        usage["prompt_tokens"], usage["completion_tokens"],
                    )
                    agent.memory_bank.add_agent_response(
                        response=entry["gt_response"],
                        current_virtual_seconds=turn_job["virtual_seconds"],
                        current_session_id=turn_job["session"].session_id,
                    )

        return phase1_stats

    # -------------------------------------------------------------------------
    # Phase 2
    # -------------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        sessions: List[Session],
        agents: List[LDAgentModule],
        log_writers: List[RetrievalLogWriter],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """
        Returns (qa_results_per_session, phase2_stats).

        phase2_stats entries are keyed by call type:
          call_4_summarization (flush), call_5_qa
        """
        phase2_stats = [
            {
                "call_4_summarization": {"input": 0, "output": 0, "llm_calls": 0},
                "call_5_qa":            {"input": 0, "output": 0, "llm_calls": 0},
            }
            for _ in sessions
        ]

        # --- STM → LTM flush ---
        flush_batch = []
        for i, (session, agent) in enumerate(zip(sessions, agents)):
            flush_job = agent.memory_bank.prepare_flush_summary(current_session_id=session.session_id)
            if flush_job is None:
                continue
            flush_batch.append({
                "session_idx": i,
                "agent": agent,
                "flush_job": flush_job,
                "prompt_item": {
                    "system_prompt": flush_job["system_prompt"],
                    "user_prompt": flush_job["user_prompt"],
                },
            })

        if flush_batch:
            results, usages = self._batch_generate_with_retry(
                prompt_items=[entry["prompt_item"] for entry in flush_batch],
                max_tokens=100,
                temperature=0.7,
                guided_json=SUMMARY_GUIDED_JSON,
            )
            for entry, result, usage in zip(flush_batch, results, usages):
                agent = entry["agent"]
                enriched = _attach_usage(result, usage)
                if agent.memory_bank.llm_logger is not None:
                    agent.memory_bank.llm_logger.log(
                        "call_4_summarization",
                        entry["flush_job"]["system_prompt"],
                        entry["flush_job"]["user_prompt"],
                        enriched,
                    )
                agent.memory_bank.set_last_summarize_usage(
                    usage["prompt_tokens"], usage["completion_tokens"]
                )
                summary_text = result.get("summary", "") if isinstance(result, dict) else ""
                agent.memory_bank.apply_flush_summary_result(entry["flush_job"], summary_text)
                _add_tokens(
                    phase2_stats[entry["session_idx"]]["call_4_summarization"],
                    usage["prompt_tokens"], usage["completion_tokens"],
                )

        # --- QA preparation ---
        qa_jobs = []
        for i, (session, agent) in enumerate(zip(sessions, agents)):
            memory_bank = agent.memory_bank
            personas = agent.personas
            generator = agent.generator

            for qa_idx, qa in enumerate(session.qa):
                relevant_memories = memory_bank.relevance_retrieve(
                    ori_query=qa.question,
                    n_results=cfg.RETRIEVE_K,
                    current_virtual_seconds=memory_bank.current_virtual_seconds,
                )
                memories_str = format_memories_for_prompt(
                    relevant_memories, memory_bank.current_virtual_seconds
                )
                context_str = agent._format_stm_context_for_qa()

                user_traits, agent_traits = personas.get_current_traits()
                sys_prompt, user_prompt, _, schema = generator.build_qa_prompt(
                    question=qa.question,
                    memories=memories_str,
                    user_traits=user_traits,
                    agent_traits=agent_traits,
                    subset=self.subset,
                    context=context_str,
                )

                retrieved_metadata = [
                    {
                        "session_id": mem.get("session_id", 0),
                        "conv_id": mem.get("conv_id", 0),
                        "virtual_seconds": mem.get("virtual_seconds", 0.0),
                        "score": mem.get("score", 0.0),
                    }
                    for mem in relevant_memories
                ]
                retrieval_log_data = {
                    "query": qa.question,
                    "relevant_memories": relevant_memories,
                    "module_specific": {
                        "ltm_entry_count": memory_bank.get_memory_count(),
                        "stm_context_turns": len(memory_bank.short_term_memory),
                        "user_trait_count": _count_traits(user_traits),
                        "agent_trait_count": _count_traits(agent_traits),
                    },
                }
                log_writers[i].write(_build_retrieval_log_entry(
                    phase="qa",
                    session_id=session.session_id,
                    conv_id=-1,
                    turn_id=qa_idx,
                    retrieval_data=retrieval_log_data,
                ))

                qa_jobs.append({
                    "session_idx": i,
                    "session": session,
                    "agent": agent,
                    "qa": qa,
                    "qa_idx": qa_idx,
                    "retrieved_metadata": retrieved_metadata,
                    "prompt_item": {
                        "system_prompt": sys_prompt,
                        "user_prompt": user_prompt,
                    },
                    "schema": schema,
                })

        # --- QA batch generation ---
        qa_results_per_session: List[List[Dict]] = [[] for _ in sessions]

        for start in tqdm(range(0, len(qa_jobs), cfg.QA_BATCH_SIZE), desc="Phase2 QA chunks"):
            chunk = qa_jobs[start:start + cfg.QA_BATCH_SIZE]
            if not chunk:
                continue
            results, usages = self._batch_generate_with_retry(
                prompt_items=[entry["prompt_item"] for entry in chunk],
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=chunk[0]["schema"],
            )

            for entry, result, usage in zip(chunk, results, usages):
                agent = entry["agent"]
                enriched = _attach_usage(result, usage)
                if agent.generator.llm_logger is not None:
                    agent.generator.llm_logger.log(
                        "call_5_qa",
                        entry["prompt_item"]["system_prompt"],
                        entry["prompt_item"]["user_prompt"],
                        enriched,
                    )

                _add_tokens(
                    phase2_stats[entry["session_idx"]]["call_5_qa"],
                    usage["prompt_tokens"], usage["completion_tokens"],
                )

                answer = result.get("answer", "") if isinstance(result, dict) else ""
                if self.subset == "supportive":
                    answer = _normalize_yes_no_unknown(answer)

                qa_results_per_session[entry["session_idx"]].append({
                    "question": entry["qa"].question,
                    "generated_answer": answer,
                    "ground_truth_answer": entry["qa"].answer,
                    "retrieved_memories": entry["retrieved_metadata"],
                    "qa_tokens": {
                        "input": usage["prompt_tokens"],
                        "output": usage["completion_tokens"],
                        "model": self.model_path,
                    },
                })

        return qa_results_per_session, phase2_stats

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run_batch(self, sessions: List[Session]) -> List[Dict[str, Any]]:
        agents = []
        log_writers = []

        for session in sessions:
            sample_id = f"ldagent_{cfg.extract_model_name(self.model_path)}_{session.session_id}"
            agent = LDAgentModule(
                llm_client=self.llm_client,
                config=cfg,
                logger=logger,
                sample_id=sample_id,
                shared_lemma_tokenizer=self.shared_lemma_tokenizer,
                shared_encoder=self.shared_encoder,
            )
            if cfg.ENABLE_LLM_CALL_LOGGING:
                llm_log_dir = self.prompt_log_dir / f"session_{session.session_id}"
                agent.set_llm_logger(LLMCallLogger(llm_log_dir))
            agents.append(agent)
            log_writers.append(
                RetrievalLogWriter(
                    self.retrieval_log_dir / f"session_{session.session_id}_retrieval_log.jsonl"
                )
            )

        phase1_stats = self._run_phase1_batched(sessions, agents, log_writers)

        # Capture memory state after Phase 1, before QA
        memory_stats_list = [agent.get_memory_stats() for agent in agents]

        qa_results_list, phase2_stats = self._run_phase2_batched(sessions, agents, log_writers)

        results = []
        for i, (session, agent) in enumerate(zip(sessions, agents)):
            memory_snapshot_path = None
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snap_dir = self.memory_snapshots_dir / f"session_{session.session_id}"
                agent.save_snapshot(snap_dir)
                memory_snapshot_path = f"memory_snapshots/session_{session.session_id}"

            agent.clear()

            p1 = phase1_stats[i]
            p2 = phase2_stats[i]

            # Merge call_4_summarization from Phase 1 (boundary) and Phase 2 (flush)
            sum4 = {
                "input":     p1["call_4_summarization"]["input"]     + p2["call_4_summarization"]["input"],
                "output":    p1["call_4_summarization"]["output"]    + p2["call_4_summarization"]["output"],
                "llm_calls": p1["call_4_summarization"]["llm_calls"] + p2["call_4_summarization"]["llm_calls"],
            }

            total_input = (
                p1["call_2_user_persona"]["input"]
                + p1["call_3_agent_persona"]["input"]
                + sum4["input"]
                + p2["call_5_qa"]["input"]
            )
            total_output = (
                p1["call_2_user_persona"]["output"]
                + p1["call_3_agent_persona"]["output"]
                + sum4["output"]
                + p2["call_5_qa"]["output"]
            )
            total_llm_calls = (
                p1["call_2_user_persona"]["llm_calls"]
                + p1["call_3_agent_persona"]["llm_calls"]
                + sum4["llm_calls"]
                + p2["call_5_qa"]["llm_calls"]
            )

            results.append({
                "session_id":     session.session_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": memory_stats_list[i],
                "qa_results": qa_results_list[i],
                "token_statistics": {
                    "call_2_user_persona":  p1["call_2_user_persona"],
                    "call_3_agent_persona": p1["call_3_agent_persona"],
                    "call_4_summarization": sum4,
                    "call_5_qa":            p2["call_5_qa"],
                    "total_input":          total_input,
                    "total_output":         total_output,
                    "total_llm_calls":      total_llm_calls,
                },
                "memory_snapshot_path": memory_snapshot_path,
            })

        return results


# =============================================================================
# LLM CLIENT FACTORY
# =============================================================================


def create_llm_client(
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int = None,
):
    from llm_client import create_llm_client as _create

    if cfg.LLM_ENGINE == "vllm":
        kwargs = {
            "engine": "vllm",
            "model_path": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        return _create(**kwargs)
    if cfg.LLM_ENGINE == "openai":
        c = cfg.OPENAI_CONFIG
        return _create(engine="openai", model_name=c["model_name"], api_key=c.get("api_key"))
    if cfg.LLM_ENGINE == "together":
        c = cfg.TOGETHER_CONFIG
        return _create(engine="together", model_name=c["model_name"], api_key=c.get("api_key"))
    raise ValueError(f"Unknown LLM engine: {cfg.LLM_ENGINE}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_0")
    pre_args, _ = pre_parser.parse_known_args()

    global cfg
    cfg = importlib.import_module(Path(pre_args.config).stem)

    parser = argparse.ArgumentParser(description="Run LD-Agent Batch Experiment (ImplexConv)")
    parser.add_argument("--start-session", type=int, required=True)
    parser.add_argument("--end-session", type=int, required=True)
    parser.add_argument("--subset", type=str, required=True, choices=["opposed", "supportive"])
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    config_name = Path(args.config).stem
    cfg.ensure_directories(args.model, args.subset, args.start_session, args.end_session, config_name)

    session_dir = cfg.get_session_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file   = cfg.get_results_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.subset, args.start_session, args.end_session, config_name)

    logger.info("=" * 60)
    logger.info("LD-Agent Batch Experiment")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Subset          : {args.subset}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    dataset_path = cfg.DATASET_OPPOSED if args.subset == "opposed" else cfg.DATASET_SUPPORTIVE
    sessions = load_implexconv_dataset(dataset_path)
    if args.end_session >= len(sessions):
        logger.error(f"end_session={args.end_session} out of range (dataset has {len(sessions)} sessions)")
        return 1

    target_sessions = sessions[args.start_session:args.end_session + 1]
    completed_ids = load_checkpoint(checkpoint_file) if cfg.ENABLE_CHECKPOINTING else set()
    pending_sessions = [s for s in target_sessions if s.session_id not in completed_ids]

    if not pending_sessions:
        logger.info("All sessions already completed.")
        return 0

    logger.info("Loading shared spaCy lemma tokenizer...")
    import spacy
    try:
        shared_lemma_tokenizer = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        shared_lemma_tokenizer = spacy.load("en_core_web_sm")

    logger.info("Loading shared SentenceTransformer encoder (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Encoder ready.")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model,
        args.tensor_parallel,
        args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    config_metadata = {
        "config_name":             config_name,
        "model":                   args.model,
        "subset":                  args.subset,
        "embedding_model":         "sentence-transformers/all-MiniLM-L6-v2",
        "temperature":             cfg.TEMPERATURE,
        "max_tokens":              cfg.MAX_TOKENS,
        "session_range":           [args.start_session, args.end_session],
        "relevance_memory_number": cfg.RELEVANCE_MEMORY_NUMBER,
        "retrieve_k":              cfg.RETRIEVE_K,
        "dist_threshold":          cfg.DIST_THRESHOLD,
        "conv_ids_per_day":        cfg.CONV_IDS_PER_DAY,
        "minutes_per_turn":        cfg.MINUTES_PER_TURN,
        "finalize_every_n_convs":  cfg.FINALIZE_EVERY_N_CONVS,
        "decay_temp":              cfg.DECAY_TEMP,
        "max_user_personas":       cfg.MAX_USER_PERSONAS,
        "max_agent_personas":      cfg.MAX_AGENT_PERSONAS,
    }

    runner = BatchedLDAgentRunner(
        llm_client=llm_client,
        model_path=args.model,
        subset=args.subset,
        start_session=args.start_session,
        end_session=args.end_session,
        config_name=config_name,
        config_metadata=config_metadata,
        shared_lemma_tokenizer=shared_lemma_tokenizer,
        shared_encoder=shared_encoder,
    )

    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# LD-Agent Batch")
    print(f"# Model    : {cfg.extract_model_name(args.model)}")
    print(f"# Sessions : [{args.start_session}, {args.end_session}] ({len(pending_sessions)} to process)")
    print(f"# Batch    : {args.batch_size}")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_sessions[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        session_ids = [s.session_id for s in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: sessions {session_ids}")

        try:
            batch_results = runner.run_batch(batch)
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
                completed_ids.add(result["session_id"])
                save_checkpoint(
                    checkpoint_file,
                    completed_ids,
                    args.model,
                    args.subset,
                    args.start_session,
                    args.end_session,
                    config_name,
                )
            logger.info(f"Session {result['session_id']} complete.")

    logger.info("All sessions completed.")
    print_completion_banner(results_file, len(target_sessions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
