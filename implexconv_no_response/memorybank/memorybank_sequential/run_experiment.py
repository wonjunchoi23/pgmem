"""
MemoryBank Experiment Runner — ImplexConv (QA-Only Variant)

Response generation LLM calls are removed. Phase 1 still retrieves memories and
constructs the response prompt (for logging), but does NOT call the LLM.
Memory is built using GT agent responses as before. Only QA (Phase 2) is evaluated.

Per-session flow:
  Phase 1 — Memory Construction:
    For each (user_turn, assistant_turn) pair:
      1. Retrieve relevant memories (top-k cosine similarity)
      2. Build response prompt (NO LLM call) → log to call_1_response (output=null)
         → estimate input tokens via len(prompt) // CHARS_PER_TOKEN
      3. Write retrieval log entry (phase="prompt_construction")
      4. Store user turn and GT assistant turn (embedding only, no LLM)
      5. On conv_id boundary: apply forgetting + summarize daily events + personality

    On conv_id boundary ("new day"):
      6. apply_forgetting → summarize_daily (2 LLM calls: call_2, call_3)
      7. Every GLOBAL_SUMMARY_INTERVAL conv_ids: synthesize_global (2 LLM calls: call_4, call_5)

  Phase 1 End — Flush + Global Summary Synthesis:
    Flush remaining turns → summarize_daily → synthesize_global (always)

  Phase 2 — QA Answering (memory frozen, no new stores):
    For each QA pair:
      1. Retrieve using question directly (no keyword generation, update_strength=False)
      2. Answer QA with global summaries + retrieved memory → log to call_6_qa
         (timed: retrieval + inference)
      3. Write retrieval log entry (phase="qa")

  Phase 3 — Cleanup:
    Save memory snapshot (if SAVE_MEMORY_SNAPSHOTS)
    Clear memory
    Save results + checkpoint

Usage:
    python run_experiment.py \\
        --start-session 0 --end-session 4 \\
        --subset opposed \\
        --model Qwen/Qwen3-1.7B \\
        --tensor-parallel 1 --gpu-memory 0.5 \\
        --config config_0
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

# Suppress noisy logs before any imports
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

# cfg is loaded dynamically in main() based on --config argument.
cfg = None  # type: ignore

from load_dataset import (
    load_implexconv_dataset,
    Session,
    Turn,
    QAPair,
)

# Chars-per-token estimate for response prompt token counting (no LLM call).
_CHARS_PER_TOKEN = 4


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
        log_file = log_dir / f"memorybank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers
    )
    return logging.getLogger(__name__)


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path) -> Optional[int]:
    if not checkpoint_file.exists():
        return None
    try:
        with open(checkpoint_file) as f:
            return json.load(f).get("last_completed_session_index")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return None


def save_checkpoint(
    checkpoint_file: Path,
    session_index: int,
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config",
):
    data = {
        "last_completed_session_index": session_index,
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
    """Append one retrieval log entry (JSONL)."""
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
    current_conv_id: int,
    event_summary_length: int,
    user_portrait_length: int,
    history_pairs: int = 0,
) -> Dict:
    """
    Build a retrieval log entry for MemoryBank.

    module_specific fields:
    - current_conv_id: conv_id at time of retrieval (used for forgetting curve)
    - total_memories: total memory entries in the store
    - event_summary_length: chars in current event summary
    - user_portrait_length: chars in current user portrait

    Prompt snapshots are NOT stored here — they are logged separately in
    prompt_log/ (call_1_response for Phase 1, call_6_qa for Phase 2).
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "session_id": session_id,
        "conv_id": conv_id,
        "turn_id": turn_id,
        "query": query,
        "memory_type": ["retrieved", "history"],
        "num_retrieved": [len(retrieved_items), history_pairs],
        "retrieved_items": retrieved_items,
        "retrieval_scores": [item["score"] for item in retrieved_items],
        "module_specific": {
            "module": "memorybank",
            "current_conv_id": current_conv_id,
            "total_memories": total_memories,
            "event_summary_length": event_summary_length,
            "user_portrait_length": user_portrait_length,
        },
    }


# =============================================================================
# TIMING HELPER
# =============================================================================

def avg_timing(timings: List[Dict]) -> Dict:
    if not timings:
        return {"retrieval_time": 0.0, "inference_time": 0.0, "total_time": 0.0}
    n = len(timings)
    return {
        k: sum(t.get(k, 0.0) for t in timings) / n
        for k in ("retrieval_time", "inference_time", "total_time")
    }


# =============================================================================
# HISTORY BUFFER
# =============================================================================

def _format_dialogue_for_summary(turns: List[Tuple[Turn, Optional[Turn]]]) -> str:
    """Format turn pairs into dialogue text for summarization."""
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
    """Return all turn pairs from the last `window` conv_ids.

    Includes turns where user_turn.conv_id >= current_conv_id - window + 1,
    i.e. within [current_conv_id - window + 1, current_conv_id].
    """
    min_conv_id = current_conv_id - window + 1
    return [
        pair for pair in history_buffer
        if pair[0] is not None and pair[0].conv_id >= min_conv_id
    ]


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class MemoryBankRunner:
    """
    Runs the MemoryBank experiment on a single session (QA-only variant).

    Phase 1 — Memory Construction (no response generation LLM call):
        For each (user_turn, assistant_turn) pair:
        1. Retrieve memories (top-k cosine similarity)
        2. Build response prompt (no LLM call) → log to call_1_response (output=null)
        3. Write retrieval log (phase="prompt_construction")
        4. Store user turn and GT assistant turn (embedding only)
        5. On conv_id boundary: apply forgetting + summarize daily events + personality

    Phase 1 End — Flush + Global summary synthesis (always)

    Phase 2 — QA Answering (memory frozen, no strength updates):
        For each QA:
        1. Retrieve using question directly (no keyword generation, no strength update)
        2. Answer QA with global summaries → log to call_6_qa
        3. Write retrieval log (phase="qa")
    """

    def __init__(self, agent, subset: str, model_path: str):
        self.agent = agent
        self.subset = subset
        self.model_path = model_path

    def run_session(
        self,
        session: Session,
        retrieval_log_path: Path,
        snapshots_dir: Path,
        prompt_log_dir: Optional[Path] = None,
    ) -> Dict:
        """Run Phases 1, 1-end transition, 2, and 3 on a single session."""
        logger.info(f"{'='*60}")
        logger.info(
            f"Session {session.session_id}  "
            f"({len(session.get_turn_pairs())} turn pairs, "
            f"{len(session.qa)} QA)"
        )
        logger.info(f"{'='*60}")

        self.agent.clear_memory()

        # Set up per-session LLM call logger
        if cfg.ENABLE_LLM_CALL_LOGGING and prompt_log_dir is not None:
            from agent import LLMCallLogger
            session_log_dir = prompt_log_dir / f"session_{session.session_id}"
            self.agent.set_llm_logger(LLMCallLogger(session_log_dir))

        # ==============================================================
        # Phase 1: Memory Construction (no response generation)
        # ==============================================================
        phase1_total_response_input = 0
        total_internal_input = total_internal_output = 0
        num_internal_calls = 0

        turn_pairs = session.get_turn_pairs()

        # Accumulates all turn pairs; sliced by conv_id window at usage time.
        history_buffer: List[Tuple[Turn, Optional[Turn]]] = []

        # conv_id boundary tracking
        prev_conv_id = None
        conv_turn_pairs: List[Tuple[Turn, Optional[Turn]]] = []

        # Accumulates turns across CONVS_PER_DAY conv_ids; flushed when
        # daily summary fires (every cfg.CONVS_PER_DAY conv_ids).
        day_batch_turns: List[Tuple[Turn, Optional[Turn]]] = []

        for user_turn, assistant_turn in tqdm(
            turn_pairs, desc=f"S{session.session_id} Phase1"
        ):
            current_conv_id = user_turn.conv_id

            # ---- conv_id boundary detection ----
            if prev_conv_id is not None and current_conv_id != prev_conv_id:
                # Forget at every conv_id boundary (original MemoryBank behaviour).
                self.agent.apply_forgetting(current_conv_id)

                # Accumulate completed conv_id's turns into the day batch.
                day_batch_turns.extend(conv_turn_pairs)
                conv_turn_pairs = []

                # Daily summary: fire every CONVS_PER_DAY conv_ids (= 1 day).
                should_daily = (current_conv_id % cfg.CONVS_PER_DAY == 0)
                # Global synthesis: fire every GLOBAL_SUMMARY_INTERVAL conv_ids.
                should_global = should_daily and (
                    current_conv_id % cfg.GLOBAL_SUMMARY_INTERVAL == 0
                )

                if should_daily and day_batch_turns:
                    dialogue_text = _format_dialogue_for_summary(day_batch_turns)
                    self.agent.on_conv_boundary(prev_conv_id, dialogue_text)
                    day_batch_turns = []

                    summ_tokens = self.agent.get_and_reset_summary_tokens()
                    total_internal_input += summ_tokens["input"]
                    total_internal_output += summ_tokens["output"]
                    num_internal_calls += summ_tokens["api_calls"]

                if should_global:
                    self.agent.on_session_end()

                    synth_tokens = self.agent.get_and_reset_summary_tokens()
                    total_internal_input += synth_tokens["input"]
                    total_internal_output += synth_tokens["output"]
                    num_internal_calls += synth_tokens["api_calls"]

            prev_conv_id = current_conv_id

            # Step 1: Retrieve
            retrieval_result = self.agent.retrieve_memory(
                user_turn.utterance,
                current_conv_id=current_conv_id,
                update_strength=True,
            )

            # Step 2: Build response prompt (NO LLM call) + log call_1_response
            recent_history = _get_history_for_conv_id(
                history_buffer, current_conv_id, cfg.HISTORY_CONV_WINDOW
            )
            prompt_snap = self.agent.build_response_prompt(
                user_utterance=user_turn.utterance,
                retrieved_memory=retrieval_result.formatted,
                history=recent_history,
            )

            # Estimate input tokens for this prompt (no actual LLM call)
            phase1_total_response_input += len(prompt_snap) // _CHARS_PER_TOKEN

            # Step 3: Write retrieval log
            log_entry = build_retrieval_log_entry(
                phase="prompt_construction",
                session_id=session.session_id,
                conv_id=user_turn.conv_id,
                turn_id=user_turn.turn_id,
                query=user_turn.utterance,
                retrieved_items=retrieval_result.items,
                total_memories=retrieval_result.total_memories,
                current_conv_id=current_conv_id,
                event_summary_length=len(self.agent.get_event_summary()),
                user_portrait_length=len(self.agent.get_user_portrait()),
                history_pairs=len(recent_history),
            )
            write_retrieval_log(retrieval_log_path, log_entry)

            # Step 4: Store user turn in memory (embedding only)
            user_timestamp = (
                f"{session.session_id:04d}_"
                f"{user_turn.conv_id:04d}_"
                f"{user_turn.turn_id:04d}"
            )
            self.agent.add_memory(
                user_turn.to_message(),  # "User: ..."
                conv_id=current_conv_id,
                timestamp=user_timestamp,
            )

            # Step 5: Store GT assistant turn in memory
            if assistant_turn:
                asst_timestamp = (
                    f"{session.session_id:04d}_"
                    f"{assistant_turn.conv_id:04d}_"
                    f"{assistant_turn.turn_id:04d}"
                )
                self.agent.add_memory(
                    assistant_turn.to_message(),  # "Assistant: ..."
                    conv_id=current_conv_id,
                    timestamp=asst_timestamp,
                )

            # Update history buffer and conv tracking
            history_buffer.append((user_turn, assistant_turn))
            conv_turn_pairs.append((user_turn, assistant_turn))

        # ---- Flush remaining turns at end of Phase 1 ----
        day_batch_turns.extend(conv_turn_pairs)
        if day_batch_turns and prev_conv_id is not None:
            dialogue_text = _format_dialogue_for_summary(day_batch_turns)
            self.agent.on_conv_boundary(prev_conv_id, dialogue_text)

            summ_tokens = self.agent.get_and_reset_summary_tokens()
            total_internal_input += summ_tokens["input"]
            total_internal_output += summ_tokens["output"]
            num_internal_calls += summ_tokens["api_calls"]

        logger.info(
            f"Phase 1 done: {len(turn_pairs)} turns processed, "
            f"memory count: {self.agent.get_memory_count()}"
        )

        # ==============================================================
        # Phase 1 End: Global Summary Synthesis (always)
        # ==============================================================
        self.agent.on_session_end()

        synth_tokens = self.agent.get_and_reset_summary_tokens()
        total_internal_input += synth_tokens["input"]
        total_internal_output += synth_tokens["output"]
        num_internal_calls += synth_tokens["api_calls"]

        logger.info(
            f"Global summaries synthesized. "
            f"Event: {len(self.agent.get_event_summary())} chars, "
            f"Portrait: {len(self.agent.get_user_portrait())} chars"
        )

        # ==============================================================
        # Phase 2: QA Answering (memory frozen, no strength updates)
        # ==============================================================
        qa_results: List[Dict] = []
        phase2_timings: List[Dict] = []

        total_qa_input = total_qa_output = 0
        num_qa_calls = 0

        # Use the final conv_id for forgetting curve in QA retrieval
        qa_conv_id = prev_conv_id if prev_conv_id is not None else 0

        # History for QA: last HISTORY_CONV_WINDOW conv_ids from Phase 1
        qa_history = _get_history_for_conv_id(
            history_buffer, qa_conv_id, cfg.HISTORY_CONV_WINDOW
        )

        for qa in tqdm(session.qa, desc=f"S{session.session_id} Phase2"):
            # Step 1: Retrieve using question directly (no keyword generation)
            # update_strength=False: memory is frozen during QA
            t0 = time.time()
            retrieval_result = self.agent.retrieve_memory(
                qa.question,
                current_conv_id=qa_conv_id,
                update_strength=False,
            )
            t_after_retrieval = time.time()

            # Step 2: Answer QA (timed)
            answer, qa_tokens, prompt_snap = self.agent.answer_qa(
                qa.question,
                retrieved_memory=retrieval_result.formatted,
                subset=self.subset,
                history=qa_history,
            )
            t_after_inference = time.time()

            phase2_timings.append({
                "retrieval_time": t_after_retrieval - t0,
                "inference_time": t_after_inference - t_after_retrieval,
                "total_time": t_after_inference - t0,
            })

            # Step 3: Write retrieval log
            log_entry = build_retrieval_log_entry(
                phase="qa",
                session_id=session.session_id,
                conv_id=-1,
                turn_id=-1,
                query=qa.question,
                retrieved_items=retrieval_result.items,
                total_memories=retrieval_result.total_memories,
                current_conv_id=qa_conv_id,
                event_summary_length=len(self.agent.get_event_summary()),
                user_portrait_length=len(self.agent.get_user_portrait()),
                history_pairs=len(qa_history),
            )
            write_retrieval_log(retrieval_log_path, log_entry)

            # Count only successful QA calls (answer is non-empty or valid label)
            if answer:
                num_qa_calls += 1
            total_qa_input += qa_tokens.get("input", 0)
            total_qa_output += qa_tokens.get("output", 0)

            # Build retrieved_memories metadata
            retrieved_metadata = []
            for item in retrieval_result.items:
                meta = dict(item.get("source_turn", {}))
                meta["score"] = item.get("score", 0.0)
                retrieved_metadata.append(meta)

            qa_results.append({
                "question": qa.question,
                "generated_answer": answer,
                "ground_truth_answer": qa.answer,
                "retrieved_memories": retrieved_metadata,
                "qa_tokens": qa_tokens,
            })

        logger.info(f"Phase 2 done: {len(qa_results)} QA answers")

        # ==============================================================
        # Phase 3: Save snapshot, aggregate stats
        # ==============================================================
        snapshot_rel_path = None
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snapshot_dir = snapshots_dir / f"session_{session.session_id}"
            self.agent.save_memory_snapshot(snapshot_dir)
            snapshot_rel_path = (
                f"memory_snapshots/session_{session.session_id}/"
            )
            logger.info(f"Memory snapshot saved to {snapshot_dir}")

        # Aggregate token statistics (QA-only variant schema)
        token_stats = {
            "total_response_input": phase1_total_response_input,  # estimated, no LLM call
            "total_qa_input": total_qa_input,
            "total_qa_output": total_qa_output,
            "total_input": (
                phase1_total_response_input
                + total_qa_input
                + total_internal_input
            ),
            "total_output": (
                total_qa_output
                + total_internal_output
            ),
            "num_qa_api_calls": num_qa_calls,
            "num_total_api_calls": num_qa_calls + num_internal_calls,
        }

        timing_stats = {
            "phase2_qa_avg": avg_timing(phase2_timings),
        }

        return {
            "session_id": session.session_id,
            "qa_results": qa_results,
            "timing_statistics": timing_stats,
            "token_statistics": token_stats,
            "memory_snapshot_path": snapshot_rel_path,
        }


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
    elif engine == "together":
        return _create(engine="together", **cfg.TOGETHER_CONFIG)
    elif engine == "openai":
        return _create(engine="openai", **cfg.OPENAI_CONFIG)
    raise ValueError(f"Unknown LLM engine: {engine}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ---- Step 1: parse --config first so we can load the right module ----
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

    # Inject into sys.modules so agent.py / memory_bank.py see it
    sys.modules["config"] = cfg

    # Import agent AFTER config is injected
    from agent import MemoryBankAgent  # noqa: E402

    # ---- Step 2: full argument parsing ----
    parser = argparse.ArgumentParser(
        description="MemoryBank Experiment on ImplexConv (QA-Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-session", type=int, required=True,
        help="First session index (inclusive)",
    )
    parser.add_argument(
        "--end-session", type=int, required=True,
        help="Last session index (inclusive)",
    )
    parser.add_argument(
        "--subset", type=str, required=True,
        choices=["opposed", "supportive"],
        help="Dataset subset to use",
    )
    parser.add_argument(
        "--model", type=str,
        default=cfg.DEFAULT_VLLM_CONFIG["model_path"],
    )
    parser.add_argument(
        "--tensor-parallel", type=int,
        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"],
    )
    parser.add_argument(
        "--gpu-memory", type=float,
        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"],
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Override vLLM max_model_len",
    )
    parser.add_argument(
        "--config", type=str, default="config_0",
        help="Config file name (without .py). Used to load settings "
             "and as output directory prefix.",
    )
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")

    cfg.ensure_directories(
        args.model, args.subset,
        args.start_session, args.end_session, config_name,
    )

    session_dir = cfg.get_session_dir(
        args.model, args.subset,
        args.start_session, args.end_session, config_name,
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file = cfg.get_results_file(
        args.model, args.subset,
        args.start_session, args.end_session, config_name,
    )
    checkpoint_file = cfg.get_checkpoint_file(
        args.model, args.subset,
        args.start_session, args.end_session, config_name,
    )
    retrieval_log_dir = cfg.get_retrieval_log_dir(
        args.model, args.subset,
        args.start_session, args.end_session, config_name,
    )
    snapshots_dir = cfg.get_memory_snapshots_dir(
        args.model, args.subset,
        args.start_session, args.end_session, config_name,
    )
    prompt_log_dir = cfg.get_prompt_log_dir(
        args.model, args.subset,
        args.start_session, args.end_session, config_name,
    )

    logger.info("=" * 60)
    logger.info("MemoryBank Experiment (QA-Only Variant)")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Subset          : {args.subset}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Forgetting div  : {cfg.FORGETTING_DIVISOR}")
    logger.info(f"  History window  : {cfg.HISTORY_CONV_WINDOW} conv_ids")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    # Load dataset
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

    target_sessions = sessions[args.start_session : args.end_session + 1]

    # Checkpoint resume
    start_idx = 0
    if cfg.ENABLE_CHECKPOINTING:
        last = load_checkpoint(checkpoint_file)
        if last is not None:
            start_idx = last + 1
            logger.info(f"Resuming from target session index {start_idx}")

    # Init LLM
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    agent = MemoryBankAgent(llm_client, model_path=args.model)
    runner = MemoryBankRunner(
        agent, subset=args.subset, model_path=args.model
    )
    results = load_existing_results(results_file)

    print(f"\n{'#'*60}")
    print(f"# MemoryBank (QA-Only)  |  subset={args.subset}")
    print(f"# Model : {cfg.extract_model_name(args.model)}")
    print(
        f"# Sessions [{args.start_session}, {args.end_session}]  "
        f"({len(target_sessions) - start_idx} to process)"
    )
    print(f"{'#'*60}\n")

    for idx, session in enumerate(
        target_sessions[start_idx:], start=start_idx
    ):
        retrieval_log_path = (
            retrieval_log_dir
            / f"session_{session.session_id}_retrieval_log.jsonl"
        )

        try:
            result = runner.run_session(
                session, retrieval_log_path, snapshots_dir,
                prompt_log_dir=prompt_log_dir if cfg.ENABLE_LLM_CALL_LOGGING else None,
            )

            # Clear memory for next session
            agent.clear_memory()
            logger.info(
                f"Memory cleared after session {session.session_id}"
            )

            results.append(result)
            save_results(results_file, results)

            if cfg.ENABLE_CHECKPOINTING:
                save_checkpoint(
                    checkpoint_file, idx,
                    args.model, args.subset,
                    args.start_session, args.end_session,
                    config_name=config_name,
                )

            logger.info(
                f"Session {session.session_id} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total API calls: "
                f"{result['token_statistics']['num_total_api_calls']}"
            )

        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as e:
            logger.error(
                f"Error on session {session.session_id}: {e}"
            )
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'#'*60}")
    print(f"# Experiment complete!  Results -> {results_file}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
