"""
MemoryBank Experiment Runner — ImplexConv (Batched Main Variant, QA-Only)

Runs multiple sessions together by batching only the LLM-heavy steps:
daily/global summarization and Phase 2 QA answering.

Memory construction itself remains sequential within each session:
  - store GT dialogue snippets for later QA retrieval
  - apply forgetting at conv_id boundaries

At boundary points, summarization prompts across sessions are collected and sent
through one batched vLLM call per call type. QA prompts are also answered in
QA_BATCH_SIZE chunks.
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
    load_implexconv_dataset,
    Session,
    Turn,
)


# =============================================================================
# LOGGING SETUP
# =============================================================================

class _ConsoleNoiseFilter(logging.Filter):
    """Hide high-frequency housekeeping logs from the terminal only."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.startswith("apply_forgetting(")
            or message.startswith("Daily summaries for conv_id ")
        )


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    console_handler = logging.StreamHandler()
    console_handler.addFilter(_ConsoleNoiseFilter())
    handlers = [console_handler]
    if cfg.LOG_TO_FILE:
        log_file = (
            log_dir
            / f"memorybank_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers
    )
    return logging.getLogger(__name__)


logger: logging.Logger = logging.getLogger(__name__)


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
        return set()
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
    config_name: str = "config",
):
    data = {
        "completed_session_ids": sorted(completed_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name": config_name,
            "model": model_path,
            "subset": subset,
            "start_session": start_session,
            "end_session": end_session,
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


def build_retrieval_log_entry(
    phase: str,
    session_id: int,
    conv_id: int,
    turn_id: int,
    query: str,
    retrieved_items: List[Dict],
    total_memories: int,
    total_daily_summaries: int,
    current_conv_id: int,
    user_portrait_length: int,
    memory_type_counts: Dict[str, int],
    prompt_memory_block_count: int,
    memo_dates: str,
    history_pairs: int = 0,
) -> Dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "session_id": session_id,
        "conv_id": conv_id,
        "turn_id": turn_id,
        "query": query,
        "memory_type": [
            "dialogue_snippet",
            "daily_summary",
            "personality",
            "history",
        ],
        "num_retrieved": [
            memory_type_counts.get("dialogue_snippet", 0),
            memory_type_counts.get("daily_summary", 0),
            1 if user_portrait_length > 0 else 0,
            history_pairs,
        ],
        "retrieved_items": retrieved_items,
        "retrieval_scores": [item["score"] for item in retrieved_items],
        "module_specific": {
            "module": "memorybank",
            "current_conv_id": current_conv_id,
            "total_memories": total_memories,
            "total_daily_summaries": total_daily_summaries,
            "user_portrait_length": user_portrait_length,
            "prompt_memory_block_count": prompt_memory_block_count,
            "memo_dates": memo_dates,
        },
    }


# =============================================================================
# HISTORY BUFFER
# =============================================================================

def _format_dialogue_for_summary(turns: List[Tuple[Turn, Optional[Turn]]]) -> str:
    lines = []
    for user_turn, assistant_turn in turns:
        if user_turn:
            lines.append(f"User: {user_turn.utterance}")
        if assistant_turn:
            lines.append(f"Assistant: {assistant_turn.utterance}")
    return "\n".join(lines)


def _get_history_for_conv_id(
    history_buffer: List[Tuple[Turn, Optional[Turn]]],
    current_conv_id: int,
    window: int,
) -> List[Tuple[Turn, Optional[Turn]]]:
    min_conv_id = current_conv_id - window + 1
    return [
        pair for pair in history_buffer
        if pair[0] is not None and pair[0].conv_id >= min_conv_id
    ]


def _format_dialogue_snippet_for_retrieval(
    user_turn: Turn,
    assistant_turn: Optional[Turn],
) -> str:
    lines = [f"[|User|]: {user_turn.utterance}"]
    if assistant_turn is not None:
        lines.append(f"[|AI|]: {assistant_turn.utterance}")
    return "\n".join(lines)


# =============================================================================
# BATCHED RUNNER
# =============================================================================

class BatchedMemoryBankRunner:
    def __init__(
        self,
        llm_client,
        subset: str,
        model_path: str,
        shared_embedding_model,
        config_metadata: Dict = None,
    ):
        self.llm_client = llm_client
        self.subset = subset
        self.model_path = model_path
        self.shared_embedding_model = shared_embedding_model
        self.config_metadata = config_metadata or {}

    def run_batch(
        self,
        sessions: List[Session],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Optional[Path]],
    ) -> List[Dict]:
        from agent import LLMCallLogger, MemoryBankAgent

        states = []
        for session, retrieval_log_path, prompt_log_dir in zip(
            sessions, retrieval_log_paths, prompt_log_dirs
        ):
            agent = MemoryBankAgent(
                self.llm_client,
                model_path=self.model_path,
                embedding_model=self.shared_embedding_model,
            )
            if cfg.ENABLE_LLM_CALL_LOGGING and prompt_log_dir is not None:
                agent.set_llm_logger(LLMCallLogger(prompt_log_dir))

            states.append({
                "session": session,
                "agent": agent,
                "turn_pairs": session.get_turn_pairs(),
                "retrieval_log_path": retrieval_log_path,
                "history_buffer": [],
                "conv_turn_pairs": [],
                "day_batch_turns": [],
                "prev_conv_id": None,
                # Per-call-type token accumulators
                "call_2_daily_event":        {"input": 0, "output": 0, "llm_calls": 0},
                "call_3_daily_personality":  {"input": 0, "output": 0, "llm_calls": 0},
                "call_4_global_event":       {"input": 0, "output": 0, "llm_calls": 0},
                "call_5_global_personality": {"input": 0, "output": 0, "llm_calls": 0},
                "call_6_qa":                 {"input": 0, "output": 0, "llm_calls": 0},
                # Phase 1 stats (populated after Phase 1 ends)
                "memory_at_qa_start": None,
                "phase1_statistics": None,
                "qa_results": [],
            })

        self._run_phase1_batched(states)
        self._run_final_global_summaries(states)
        self._collect_phase1_end_stats(states)
        self._run_phase2_batched(states)

        results = []
        for state in states:
            session = state["session"]
            agent = state["agent"]

            snapshot_rel_path = None
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"session_{session.session_id}"
                agent.save_memory_snapshot(snapshot_dir)
                snapshot_rel_path = f"memory_snapshots/session_{session.session_id}/"
                logger.info(f"Memory snapshot saved to {snapshot_dir}")

            call_types = [
                "call_2_daily_event",
                "call_3_daily_personality",
                "call_4_global_event",
                "call_5_global_personality",
                "call_6_qa",
            ]
            token_stats = {ct: dict(state[ct]) for ct in call_types}
            token_stats["total_input"] = sum(
                state[ct]["input"] for ct in call_types
            )
            token_stats["total_output"] = sum(
                state[ct]["output"] for ct in call_types
            )
            token_stats["total_llm_calls"] = sum(
                state[ct]["llm_calls"] for ct in call_types
            )

            result = {
                "session_id": session.session_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": state["memory_at_qa_start"],
                "qa_results": state["qa_results"],
                "token_statistics": token_stats,
                "phase1_statistics": state["phase1_statistics"],
                "memory_snapshot_path": snapshot_rel_path,
            }
            results.append(result)

            agent.clear_memory()
            logger.info(f"Memory cleared: session {session.session_id}")

        return results

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_batched(self, states: List[Dict]):
        if not states:
            return

        max_turns = max(len(state["turn_pairs"]) for state in states)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active_states = [
                state for state in states
                if turn_idx < len(state["turn_pairs"])
            ]
            self._handle_conv_boundaries(active_states, turn_idx)

            for state in active_states:
                self._process_turn_pair(state, state["turn_pairs"][turn_idx])

        self._flush_remaining_daily_summaries(states)

        for state in states:
            logger.info(
                f"Phase 1 done: session {state['session'].session_id}, "
                f"{len(state['turn_pairs'])} turns processed, "
                f"memory count: {state['agent'].get_memory_count()}"
            )

    def _handle_conv_boundaries(
        self,
        active_states: List[Dict],
        turn_idx: int,
    ):
        daily_jobs = []
        global_candidates = []

        for state in active_states:
            user_turn, _ = state["turn_pairs"][turn_idx]
            current_conv_id = user_turn.conv_id
            prev_conv_id = state["prev_conv_id"]

            if prev_conv_id is None or current_conv_id == prev_conv_id:
                continue

            state["agent"].apply_forgetting(current_conv_id)
            state["day_batch_turns"].extend(state["conv_turn_pairs"])
            state["conv_turn_pairs"] = []

            should_daily = (current_conv_id % cfg.CONVS_PER_DAY == 0)
            should_global = should_daily and (
                current_conv_id % cfg.GLOBAL_SUMMARY_INTERVAL == 0
            )

            if should_daily and state["day_batch_turns"]:
                daily_jobs.append({
                    "state": state,
                    "conv_id": prev_conv_id,
                    "dialogue_text": _format_dialogue_for_summary(
                        state["day_batch_turns"]
                    ),
                })
                state["day_batch_turns"] = []

            if should_global:
                global_candidates.append(state)

        self._run_daily_summary_jobs(daily_jobs)

        global_jobs = []
        for state in global_candidates:
            prompts = state["agent"].memory_system.build_global_summary_prompts()
            if prompts is None:
                continue
            event_prompt, personality_prompt = prompts
            global_jobs.append({
                "state": state,
                "event_prompt": event_prompt,
                "personality_prompt": personality_prompt,
            })

        self._run_global_summary_jobs(global_jobs)

    def _process_turn_pair(
        self,
        state: Dict,
        turn_pair: Tuple[Turn, Optional[Turn]],
    ):
        user_turn, assistant_turn = turn_pair
        current_conv_id = user_turn.conv_id
        agent = state["agent"]

        user_timestamp = (
            f"{state['session'].session_id:04d}_"
            f"{user_turn.conv_id:04d}_"
            f"{user_turn.turn_id:04d}"
        )
        agent.add_memory(
            _format_dialogue_snippet_for_retrieval(user_turn, assistant_turn),
            conv_id=current_conv_id,
            timestamp=user_timestamp,
        )

        state["history_buffer"].append((user_turn, assistant_turn))
        state["conv_turn_pairs"].append((user_turn, assistant_turn))
        state["prev_conv_id"] = current_conv_id

    def _flush_remaining_daily_summaries(self, states: List[Dict]):
        daily_jobs = []

        for state in states:
            state["day_batch_turns"].extend(state["conv_turn_pairs"])
            state["conv_turn_pairs"] = []
            if state["day_batch_turns"] and state["prev_conv_id"] is not None:
                daily_jobs.append({
                    "state": state,
                    "conv_id": state["prev_conv_id"],
                    "dialogue_text": _format_dialogue_for_summary(
                        state["day_batch_turns"]
                    ),
                })
                state["day_batch_turns"] = []

        self._run_daily_summary_jobs(daily_jobs)

    def _run_final_global_summaries(self, states: List[Dict]):
        global_jobs = []

        for state in states:
            prompts = state["agent"].memory_system.build_global_summary_prompts()
            if prompts is None:
                continue
            event_prompt, personality_prompt = prompts
            global_jobs.append({
                "state": state,
                "event_prompt": event_prompt,
                "personality_prompt": personality_prompt,
            })

        self._run_global_summary_jobs(global_jobs)

        for state in states:
            logger.info(
                f"Global summaries synthesized for session "
                f"{state['session'].session_id}. "
                f"Event: {len(state['agent'].get_event_summary())} chars, "
                f"Portrait: {len(state['agent'].get_user_portrait())} chars"
            )

    def _collect_phase1_end_stats(self, states: List[Dict]):
        """Collect memory stats and Phase 1 internal stats after Phase 1 completes."""
        for state in states:
            state["memory_at_qa_start"] = state["agent"].get_memory_stats()
            state["phase1_statistics"] = (
                state["agent"].get_and_reset_internal_stats()
            )

    def _run_daily_summary_jobs(self, daily_jobs: List[Dict]):
        if not daily_jobs:
            return

        event_jobs = []
        personality_jobs = []
        for job in daily_jobs:
            event_prompt, personality_prompt = job["state"][
                "agent"
            ].memory_system.build_daily_summary_prompts(job["dialogue_text"])
            event_jobs.append({
                "state": job["state"],
                "conv_id": job["conv_id"],
                "prompt": event_prompt,
            })
            personality_jobs.append({
                "state": job["state"],
                "conv_id": job["conv_id"],
                "prompt": personality_prompt,
            })

        event_texts, event_usages = self._generate_text_batch(
            prompts=[job["prompt"] for job in event_jobs],
            system_prompt=None,
            max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            temperature=cfg.SUMMARIZE_TEMPERATURE,
        )
        personality_texts, personality_usages = self._generate_text_batch(
            prompts=[job["prompt"] for job in personality_jobs],
            system_prompt=None,
            max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            temperature=cfg.SUMMARIZE_TEMPERATURE,
        )

        touched_states = {}
        event_results = {}
        personality_results = {}

        for job, text, usage in zip(event_jobs, event_texts, event_usages):
            state = job["state"]
            state["agent"].memory_system.log_summary_call(
                "call_2_daily_event", job["prompt"], text
            )
            state["agent"].accumulate_call_tokens(
                "call_2_daily_event",
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
            touched_states[id(state)] = state
            event_results[id(state)] = text

        for job, text, usage in zip(
            personality_jobs, personality_texts, personality_usages
        ):
            state = job["state"]
            state["agent"].memory_system.log_summary_call(
                "call_3_daily_personality", job["prompt"], text
            )
            state["agent"].accumulate_call_tokens(
                "call_3_daily_personality",
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
            touched_states[id(state)] = state
            personality_results[id(state)] = text

        for job in daily_jobs:
            state = job["state"]
            state["agent"].memory_system.apply_daily_summary_results(
                job["conv_id"],
                event_summary=event_results.get(id(state), ""),
                personality_summary=personality_results.get(id(state), ""),
            )

        for state in touched_states.values():
            self._consume_summary_tokens(state)

    def _run_global_summary_jobs(self, global_jobs: List[Dict]):
        if not global_jobs:
            return

        event_texts, event_usages = self._generate_text_batch(
            prompts=[job["event_prompt"] for job in global_jobs],
            system_prompt=None,
            max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            temperature=cfg.SUMMARIZE_TEMPERATURE,
        )
        personality_texts, personality_usages = self._generate_text_batch(
            prompts=[job["personality_prompt"] for job in global_jobs],
            system_prompt=None,
            max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            temperature=cfg.SUMMARIZE_TEMPERATURE,
        )

        touched_states = {}
        event_results = {}
        personality_results = {}

        for job, text, usage in zip(global_jobs, event_texts, event_usages):
            state = job["state"]
            state["agent"].memory_system.log_summary_call(
                "call_4_global_event", job["event_prompt"], text
            )
            state["agent"].accumulate_call_tokens(
                "call_4_global_event",
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
            touched_states[id(state)] = state
            event_results[id(state)] = text

        for job, text, usage in zip(
            global_jobs, personality_texts, personality_usages
        ):
            state = job["state"]
            state["agent"].memory_system.log_summary_call(
                "call_5_global_personality", job["personality_prompt"], text
            )
            state["agent"].accumulate_call_tokens(
                "call_5_global_personality",
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
            touched_states[id(state)] = state
            personality_results[id(state)] = text

        for job in global_jobs:
            state = job["state"]
            state["agent"].memory_system.apply_global_summary_results(
                event_summary=event_results.get(id(state), ""),
                user_portrait=personality_results.get(id(state), ""),
            )

        for state in touched_states.values():
            self._consume_summary_tokens(state)

    def _consume_summary_tokens(self, state: Dict):
        counts = state["agent"].get_and_reset_token_counts_by_type()
        for call_type, bucket in counts.items():
            if call_type in state:
                state[call_type]["input"]     += bucket["input"]
                state[call_type]["output"]    += bucket["output"]
                state[call_type]["llm_calls"] += bucket["llm_calls"]

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(self, states: List[Dict]):
        from agent import QA_SCHEMA_OPPOSED, QA_SCHEMA_SUPPORTIVE, QA_SYSTEM_PROMPT

        schema = (
            QA_SCHEMA_SUPPORTIVE
            if self.subset == "supportive"
            else QA_SCHEMA_OPPOSED
        )

        qa_jobs = []
        for state in states:
            agent = state["agent"]
            qa_conv_id = (
                state["prev_conv_id"] if state["prev_conv_id"] is not None else 0
            )
            qa_history = _get_history_for_conv_id(
                state["history_buffer"],
                qa_conv_id,
                cfg.HISTORY_CONV_WINDOW,
            )

            for qa in state["session"].qa:
                retrieval_result = agent.retrieve_memory(
                    qa.question,
                    current_conv_id=qa_conv_id,
                    update_strength=False,
                )

                prompt = agent.build_qa_prompt(
                    question=qa.question,
                    retrieved_memory=retrieval_result.formatted,
                    memo_dates=retrieval_result.memo_dates,
                    subset=self.subset,
                    history=qa_history,
                )

                log_entry = build_retrieval_log_entry(
                    phase="qa",
                    session_id=state["session"].session_id,
                    conv_id=-1,
                    turn_id=-1,
                    query=qa.question,
                    retrieved_items=retrieval_result.items,
                    total_memories=retrieval_result.total_memories,
                    total_daily_summaries=retrieval_result.total_daily_summaries,
                    current_conv_id=qa_conv_id,
                    user_portrait_length=len(agent.get_user_portrait()),
                    memory_type_counts=retrieval_result.type_counts,
                    prompt_memory_block_count=retrieval_result.prompt_block_count,
                    memo_dates=retrieval_result.memo_dates,
                    history_pairs=len(qa_history),
                )
                write_retrieval_log(state["retrieval_log_path"], log_entry)

                retrieved_metadata = []
                for item in retrieval_result.items:
                    meta = {
                        "memory_subtype": item.get("memory_subtype"),
                        "source_label": item.get("source_label"),
                        "score": item.get("score", 0.0),
                    }
                    meta.update(item.get("source_turn", {}))
                    if "source_summary_conv_id" in item:
                        meta["source_summary_conv_id"] = item[
                            "source_summary_conv_id"
                        ]
                    meta["score"] = item.get("score", 0.0)
                    retrieved_metadata.append(meta)

                qa_jobs.append({
                    "state": state,
                    "qa": qa,
                    "qa_conv_id": qa_conv_id,
                    "history": qa_history,
                    "retrieved_memory": retrieval_result.formatted,
                    "retrieved_memo_dates": retrieval_result.memo_dates,
                    "retrieved_metadata": retrieved_metadata,
                    "prompt": prompt,
                })

        for chunk_start in tqdm(
            range(0, len(qa_jobs), cfg.QA_BATCH_SIZE),
            desc="Phase2 QA chunks",
        ):
            chunk = qa_jobs[chunk_start: chunk_start + cfg.QA_BATCH_SIZE]
            self._run_qa_chunk(chunk, schema, QA_SYSTEM_PROMPT)

        for state in states:
            logger.info(
                f"Phase 2 done: session {state['session'].session_id}, "
                f"{len(state['qa_results'])} QA answers"
            )

    def _run_qa_chunk(self, chunk: List[Dict], schema: Dict, system_prompt: str):
        if not chunk:
            return

        try:
            results, usages = self._batch_generate_json_with_retry(
                prompts=[job["prompt"] for job in chunk],
                system_prompt=system_prompt,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=schema,
            )

            for job, result, usage in zip(chunk, results, usages):
                self._record_batched_qa_result(job, result, usage, system_prompt)
            return

        except Exception as e:
            logger.warning(
                f"Batch QA generation failed for {len(chunk)} prompts; "
                f"falling back to sequential QA. Error: {e}"
            )

        for job in chunk:
            agent = job["state"]["agent"]
            answer, qa_tokens, _ = agent.answer_qa(
                question=job["qa"].question,
                retrieved_memory=job["retrieved_memory"],
                memo_dates=job["retrieved_memo_dates"],
                subset=self.subset,
                history=job["history"],
            )
            self._record_sequential_qa_result(job, answer, qa_tokens)

    def _record_batched_qa_result(
        self,
        job: Dict,
        result: Dict,
        usage: Dict,
        system_prompt: str,
    ):
        state = job["state"]
        agent = state["agent"]

        if agent._llm_logger is not None:
            agent._llm_logger.log(
                call_type="call_6_qa",
                system_prompt=system_prompt,
                user_prompt=job["prompt"],
                output=result,
            )

        answer = ""
        if isinstance(result, dict):
            answer = agent.normalize_qa_answer(
                result.get("answer", ""),
                self.subset,
            )

        token_info = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "model": self.model_path,
        }

        state["call_6_qa"]["input"]     += token_info["input"]
        state["call_6_qa"]["output"]    += token_info["output"]
        state["call_6_qa"]["llm_calls"] += 1
        state["qa_results"].append({
            "question": job["qa"].question,
            "generated_answer": answer,
            "ground_truth_answer": job["qa"].answer,
            "retrieved_memories": job["retrieved_metadata"],
            "qa_tokens": token_info,
        })

    def _record_sequential_qa_result(
        self,
        job: Dict,
        answer: str,
        qa_tokens: Dict,
    ):
        state = job["state"]

        state["call_6_qa"]["input"]     += qa_tokens.get("input", 0)
        state["call_6_qa"]["output"]    += qa_tokens.get("output", 0)
        state["call_6_qa"]["llm_calls"] += 1
        state["qa_results"].append({
            "question": job["qa"].question,
            "generated_answer": answer,
            "ground_truth_answer": job["qa"].answer,
            "retrieved_memories": job["retrieved_metadata"],
            "qa_tokens": qa_tokens,
        })

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _generate_text_batch(
        self,
        prompts: List[str],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[List[str], List[Dict]]:
        if not prompts:
            return [], []

        try:
            return self.llm_client.generate_batch_raw(
                prompts=prompts,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                return_usage=True,
            )
        except Exception as e:
            logger.warning(
                f"Batch text generation failed for {len(prompts)} prompts; "
                f"falling back to sequential calls. Error: {e}"
            )

        texts = []
        usages = []
        for prompt in prompts:
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    return_usage=True,
                )
                if isinstance(result, dict):
                    texts.append(result.get("content", ""))
                    usage = result.get("_usage", {})
                else:
                    texts.append(str(result) if result is not None else "")
                    usage = {}
            except Exception as e:
                logger.error(f"Sequential text generation failed: {e}")
                texts.append("")
                usage = {}

            usages.append({
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            })

        return texts, usages

    def _batch_generate_json_with_retry(
        self,
        prompts: List[str],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        guided_json: Dict,
    ) -> Tuple[List[Dict], List[Dict]]:
        from llm_client import _parse_json_response

        texts, usages = self.llm_client.generate_batch_raw(
            prompts=prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            guided_json=guided_json,
            return_usage=True,
        )

        parsed_results = []
        retry_indices = []
        for idx, text in enumerate(texts):
            try:
                if isinstance(text, dict):
                    parsed_results.append(text)
                else:
                    parsed_results.append(_parse_json_response(text))
            except (json.JSONDecodeError, ValueError):
                parsed_results.append(None)
                retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(
                f"Batch QA item {idx} failed JSON parsing; retrying sequentially."
            )
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
                        "completion_tokens": usage_info.get(
                            "completion_tokens", 0
                        ),
                    }
                    parsed_results[idx] = retry_result
                else:
                    parsed_results[idx] = {}
                    usages[idx] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
            except Exception as e:
                logger.error(f"Sequential QA retry failed: {e}")
                parsed_results[idx] = {}
                usages[idx] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

        return parsed_results, usages


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
        description="MemoryBank Batch Experiment on ImplexConv (QA-Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-session", type=int, required=True)
    parser.add_argument("--end-session", type=int, required=True)
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        choices=["opposed", "supportive"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default=cfg.DEFAULT_VLLM_CONFIG["model_path"],
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"],
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"],
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.BATCH_SIZE,
        help=f"Sessions to process in parallel (default: {cfg.BATCH_SIZE})",
    )
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    cfg.ensure_directories(
        args.model,
        args.subset,
        args.start_session,
        args.end_session,
        config_name,
    )

    session_dir = cfg.get_session_dir(
        args.model,
        args.subset,
        args.start_session,
        args.end_session,
        config_name,
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file = cfg.get_results_file(
        args.model,
        args.subset,
        args.start_session,
        args.end_session,
        config_name,
    )
    checkpoint_file = cfg.get_checkpoint_file(
        args.model,
        args.subset,
        args.start_session,
        args.end_session,
        config_name,
    )
    retrieval_log_dir = cfg.get_retrieval_log_dir(
        args.model,
        args.subset,
        args.start_session,
        args.end_session,
        config_name,
    )
    snapshots_dir = cfg.get_memory_snapshots_dir(
        args.model,
        args.subset,
        args.start_session,
        args.end_session,
        config_name,
    )
    prompt_log_dir = cfg.get_prompt_log_dir(
        args.model,
        args.subset,
        args.start_session,
        args.end_session,
        config_name,
    )

    logger.info("=" * 60)
    logger.info("MemoryBank Batch Experiment (QA-Only)")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Subset          : {args.subset}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Forgetting div  : {cfg.FORGETTING_DIVISOR}")
    logger.info(f"  History window  : {cfg.HISTORY_CONV_WINDOW} conv_ids")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    logger.info("Loading dataset...")
    dataset_path = (
        cfg.DATASET_OPPOSED
        if args.subset == "opposed"
        else cfg.DATASET_SUPPORTIVE
    )
    sessions = load_implexconv_dataset(dataset_path, args.subset)

    if args.end_session >= len(sessions):
        logger.error(
            f"end_session={args.end_session} out of range "
            f"(dataset has {len(sessions)} sessions)"
        )
        return 1

    target_sessions = sessions[args.start_session: args.end_session + 1]

    completed_ids: Set[int] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_ids = load_checkpoint(checkpoint_file)
        if completed_ids:
            logger.info(
                f"Resuming: {len(completed_ids)} sessions already completed"
            )

    pending_sessions = [
        session for session in target_sessions
        if session.session_id not in completed_ids
    ]
    if not pending_sessions:
        logger.info("All sessions already completed.")
        return 0

    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

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
        "embedding_model":         cfg.EMBEDDING_MODEL,
        "temperature":             cfg.TEMPERATURE,
        "max_tokens":              cfg.MAX_TOKENS,
        "session_range":           [args.start_session, args.end_session],
        "retrieve_k":              cfg.RETRIEVE_K,
        "forgetting_divisor":      cfg.FORGETTING_DIVISOR,
        "convs_per_day":           cfg.CONVS_PER_DAY,
        "global_summary_interval": cfg.GLOBAL_SUMMARY_INTERVAL,
        "history_conv_window":     cfg.HISTORY_CONV_WINDOW,
        "summarize_temperature":   cfg.SUMMARIZE_TEMPERATURE,
        "summarize_max_tokens":    cfg.SUMMARIZE_MAX_TOKENS,
    }

    runner = BatchedMemoryBankRunner(
        llm_client=llm_client,
        subset=args.subset,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
        config_metadata=config_metadata,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# MemoryBank Batch  |  subset={args.subset}")
    print(f"# Model : {cfg.extract_model_name(args.model)}")
    print(
        f"# Sessions [{args.start_session}, {args.end_session}]  "
        f"({len(pending_sessions)} to process)"
    )
    print(f"# Batch size : {args.batch_size}")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_sessions[
            batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size
        ]
        session_ids = [session.session_id for session in batch]
        logger.info(
            f"Batch {batch_idx + 1}/{num_batches}: sessions {session_ids}"
        )

        batch_retrieval_log_paths = [
            retrieval_log_dir / f"session_{session.session_id}_retrieval_log.jsonl"
            for session in batch
        ]
        batch_prompt_log_dirs = [
            (
                prompt_log_dir / f"session_{session.session_id}"
                if cfg.ENABLE_LLM_CALL_LOGGING
                else None
            )
            for session in batch
        ]

        try:
            batch_results = runner.run_batch(
                sessions=batch,
                retrieval_log_paths=batch_retrieval_log_paths,
                snapshots_dir=snapshots_dir,
                prompt_log_dirs=batch_prompt_log_dirs,
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
                completed_ids.add(result["session_id"])
                save_checkpoint(
                    checkpoint_file,
                    completed_ids,
                    args.model,
                    args.subset,
                    args.start_session,
                    args.end_session,
                    config_name=config_name,
                )

            logger.info(
                f"Session {result['session_id']} done. "
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
