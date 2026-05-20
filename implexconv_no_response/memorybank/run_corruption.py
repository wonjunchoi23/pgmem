"""
MemoryBank Memory Corruption Experiment Runner — ImplexConv (QA-Only)

Loads Phase 1 memory snapshots, randomly deletes X% of memory entries
(seed = 42 + session_id), then runs Phase 2 QA. Phase 1 is skipped entirely.

Corruption targets (Layer 1 + Layer 2):
  Layer 1:
    - entries (dialogue snippets)
    - retriever corpus + embeddings (aligned with entries)
  Layer 2:
    - daily_summary_entries
    - daily_summary_retriever corpus + embeddings (aligned with daily_summary_entries)
    - daily_event_summaries dict (by conv_id)
    - daily_personality_summaries dict (by conv_id)
  Unchanged:
    - global_event_summary, global_user_portrait (Layers 3-4)

Output directory (auto-derived):
  corrupt{rate_pct}_{config_name}_outputs_{model_name}_{subset}/

Usage:
  nohup python memorybank/run_corruption.py \\
      --model Qwen/Qwen3-1.7B \\
      --subset opposed \\
      --corrupt-rate 0.3 \\
      --start-session 0 --end-session 499 \\
      --tensor-parallel 1 --gpu-memory 0.9 \\
      --config config_0 > logs/corrupt30_memorybank_opposed.log 2>&1 &
"""

import importlib.util
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

import run_experiment as _re
from run_experiment import (
    BatchedMemoryBankRunner,
    create_llm_client,
    load_checkpoint,
    load_existing_results,
    save_checkpoint,
    save_results,
    setup_logging,
)
from load_dataset import Session, load_implexconv_dataset

cfg = None  # type: ignore
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# SNAPSHOT LOCATOR
# =============================================================================

def find_snapshot_dir(snapshot_base: Path, session_id: int) -> Path:
    for session_dir in sorted(snapshot_base.iterdir()):
        if not session_dir.name.startswith("session_"):
            continue
        snap_path = session_dir / "memory_snapshots" / f"session_{session_id}"
        if snap_path.exists():
            return snap_path
    raise FileNotFoundError(
        f"No snapshot found for session {session_id} under {snapshot_base}"
    )


# =============================================================================
# MEMORY CORRUPTION
# =============================================================================

def corrupt_memory_bank_system(
    memory_system, corrupt_rate: float, rng: random.Random
) -> Dict:
    """
    Delete corrupt_rate fraction of Layer 1 entries and Layer 2 daily summaries
    in-place.

    Layer 1: entries (dialogue snippets) + retriever corpus/embeddings.
    Layer 2: daily_summary_entries + daily_summary_retriever + source dicts
             (daily_event_summaries, daily_personality_summaries).
    Layers 3-4 (global summaries) are left intact.

    Returns corruption metadata dict.
    """
    # ---- Layer 1: dialogue snippet entries ----
    n_entries = len(memory_system.entries)
    n_delete_entries = round(n_entries * corrupt_rate)

    if n_delete_entries > 0:
        delete_entry_indices = sorted(
            rng.sample(range(n_entries), n_delete_entries)
        )
        memory_system.retriever.remove_by_indices(delete_entry_indices)
        for idx in sorted(delete_entry_indices, reverse=True):
            memory_system.entries.pop(idx)

    # ---- Layer 2: daily summary entries ----
    n_daily = len(memory_system.daily_summary_entries)
    n_delete_daily = round(n_daily * corrupt_rate)

    deleted_daily_conv_ids: Set[int] = set()
    if n_delete_daily > 0:
        delete_daily_indices = sorted(
            rng.sample(range(n_daily), n_delete_daily)
        )
        for idx in delete_daily_indices:
            deleted_daily_conv_ids.add(
                memory_system.daily_summary_entries[idx].conv_id
            )
        memory_system.daily_summary_retriever.remove_by_indices(delete_daily_indices)
        for idx in sorted(delete_daily_indices, reverse=True):
            memory_system.daily_summary_entries.pop(idx)
        for conv_id in deleted_daily_conv_ids:
            memory_system.daily_event_summaries.pop(conv_id, None)
            memory_system.daily_personality_summaries.pop(conv_id, None)

    return {
        "n_entries_before":  n_entries,
        "n_entries_deleted": n_delete_entries,
        "n_entries_after":   len(memory_system.entries),
        "n_daily_before":    n_daily,
        "n_daily_deleted":   n_delete_daily,
        "n_daily_after":     len(memory_system.daily_summary_entries),
    }


# =============================================================================
# CORRUPTION RUNNER
# =============================================================================

class CorruptionRunner(BatchedMemoryBankRunner):
    """
    Skips Phase 1; loads pre-built snapshots, corrupts them,
    then runs Phase 2 QA via inherited _run_phase2_batched().
    """

    def __init__(
        self,
        llm_client,
        subset: str,
        model_path: str,
        shared_embedding_model,
        snapshot_base: Path,
        corrupt_rate: float,
        output_dir: Path,
        start: int,
        end: int,
    ):
        super().__init__(llm_client, subset, model_path, shared_embedding_model)
        self.snapshot_base = Path(snapshot_base)
        self.corrupt_rate  = corrupt_rate
        self.output_dir    = Path(output_dir)

        session_subdir = self.output_dir / f"session_{start}_{end}"
        self.retrieval_log_dir    = session_subdir / "retrieval_logs"
        self.memory_snapshots_dir = session_subdir / "memory_snapshots"
        self.prompt_log_dir       = session_subdir / "prompt_log"

    def run_batch(self, sessions: List[Session]) -> List[Dict]:
        from agent import LLMCallLogger, MemoryBankAgent

        states = []
        corruption_stats_map: Dict[int, Dict] = {}

        for session in sessions:
            agent = MemoryBankAgent(
                self.llm_client,
                model_path=self.model_path,
                embedding_model=self.shared_embedding_model,
            )
            snap_dir = find_snapshot_dir(self.snapshot_base, session.session_id)
            agent.memory_system.load_snapshot(snap_dir)

            rng = random.Random(42 + session.session_id)
            corruption_stats_map[session.session_id] = corrupt_memory_bank_system(
                agent.memory_system, self.corrupt_rate, rng
            )

            retrieval_log_path = (
                self.retrieval_log_dir
                / f"session_{session.session_id}_retrieval_log.jsonl"
            )
            prompt_log_dir = self.prompt_log_dir / f"session_{session.session_id}"
            if cfg.ENABLE_LLM_CALL_LOGGING:
                prompt_log_dir.mkdir(parents=True, exist_ok=True)
                agent.set_llm_logger(LLMCallLogger(prompt_log_dir))

            turn_pairs = session.get_turn_pairs()
            prev_conv_id = (
                turn_pairs[-1][0].conv_id if turn_pairs else None
            )

            states.append({
                "session":           session,
                "agent":             agent,
                "turn_pairs":        turn_pairs,
                "retrieval_log_path": retrieval_log_path,
                "history_buffer":    list(turn_pairs),
                "conv_turn_pairs":   [],
                "day_batch_turns":   [],
                "prev_conv_id":      prev_conv_id,
                "call_2_daily_event":        {"input": 0, "output": 0, "llm_calls": 0},
                "call_3_daily_personality":  {"input": 0, "output": 0, "llm_calls": 0},
                "call_4_global_event":       {"input": 0, "output": 0, "llm_calls": 0},
                "call_5_global_personality": {"input": 0, "output": 0, "llm_calls": 0},
                "call_6_qa":                 {"input": 0, "output": 0, "llm_calls": 0},
                "memory_at_qa_start": None,
                "phase1_statistics":  None,
                "qa_results":         [],
            })

        # Phase 1 skipped; collect stats from loaded (corrupted) memory
        self._collect_phase1_end_stats(states)

        # Phase 2 QA (inherited)
        self._run_phase2_batched(states)

        # Assemble results
        results = []
        zero_call = {"input": 0, "output": 0, "llm_calls": 0}

        for state in states:
            session = state["session"]
            agent   = state["agent"]

            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snap_out = (
                    self.memory_snapshots_dir / f"session_{session.session_id}"
                )
                agent.save_memory_snapshot(snap_out)

            call_6 = state["call_6_qa"]
            token_stats = {
                "call_2_daily_event":        dict(zero_call),
                "call_3_daily_personality":  dict(zero_call),
                "call_4_global_event":       dict(zero_call),
                "call_5_global_personality": dict(zero_call),
                "call_6_qa": {
                    "input":     call_6["input"],
                    "output":    call_6["output"],
                    "llm_calls": call_6["llm_calls"],
                },
                "total_input":     call_6["input"],
                "total_output":    call_6["output"],
                "total_llm_calls": call_6["llm_calls"],
            }

            results.append({
                "session_id":            session.session_id,
                "qa_results":            state["qa_results"],
                "token_statistics":      token_stats,
                "memory_at_qa_start":    state["memory_at_qa_start"],
                "phase1_statistics":     state["phase1_statistics"],
                "corruption_statistics": corruption_stats_map[session.session_id],
                "memory_snapshot_path": (
                    f"memory_snapshots/session_{session.session_id}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            })

            agent.clear_memory()
            logger.info(f"Memory cleared: session {session.session_id}")

        return results


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

def ensure_output_dirs(output_dir: Path, start: int, end: int):
    session_subdir = output_dir / f"session_{start}_{end}"
    (session_subdir / "retrieval_logs").mkdir(parents=True, exist_ok=True)
    (session_subdir / "memory_snapshots").mkdir(parents=True, exist_ok=True)
    (session_subdir / "prompt_log").mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_0")
    pre_args, _ = pre_parser.parse_known_args()
    config_name = Path(pre_args.config).stem

    config_file = _MODULE_DIR / f"{config_name}.py"
    if not config_file.exists():
        print(f"[error] Config file not found: {config_file}")
        return 1

    spec = importlib.util.spec_from_file_location(config_name, config_file)
    global cfg
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    sys.modules["config"] = cfg  # agent.py + memory_bank.py do `import config as cfg`
    _re.cfg = cfg                # run_experiment's module-level functions use `cfg`

    parser = argparse.ArgumentParser(
        description="MemoryBank Memory Corruption Experiment on ImplexConv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-session",   type=int,   required=True)
    parser.add_argument("--end-session",     type=int,   required=True)
    parser.add_argument("--subset",          type=str,   required=True,
                        choices=["opposed", "supportive"])
    parser.add_argument("--corrupt-rate",    type=float, required=True,
                        help="Fraction of memories to delete, e.g. 0.3")
    parser.add_argument("--model",           type=str,
                        default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory",      type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len",   type=int,   default=None)
    parser.add_argument("--batch-size",      type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--config",          type=str,   default="config_0")
    args = parser.parse_args()

    if not 0.0 < args.corrupt_rate < 1.0:
        parser.error("--corrupt-rate must be strictly between 0.0 and 1.0")
    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    # Auto-derive paths
    model_name    = cfg.extract_model_name(args.model)
    rate_pct      = int(args.corrupt_rate * 100)
    snapshot_base = cfg.get_output_dir(args.model, args.subset, config_name)
    output_dir    = (
        cfg.BASE_OUTPUT_DIR
        / f"corrupt{rate_pct}_{config_name}_outputs_{model_name}_{args.subset}"
    )

    ensure_output_dirs(output_dir, args.start_session, args.end_session)
    session_subdir = output_dir / f"session_{args.start_session}_{args.end_session}"

    global logger
    logger = setup_logging(session_subdir / "logs")
    _re.logger = logger

    results_file = (
        session_subdir
        / f"results_{model_name}_{args.subset}_session_{args.start_session}_{args.end_session}.json"
    )
    checkpoint_file = (
        session_subdir
        / f"checkpoint_{model_name}_{args.subset}_session_{args.start_session}_{args.end_session}.json"
    )

    logger.info("=" * 60)
    logger.info("MemoryBank Memory Corruption Experiment")
    logger.info(f"  Config        : {config_name}")
    logger.info(f"  Subset        : {args.subset}")
    logger.info(f"  Model         : {args.model}")
    logger.info(f"  Corrupt rate  : {args.corrupt_rate} ({rate_pct}%)")
    logger.info(f"  Sessions      : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Snapshot base : {snapshot_base}")
    logger.info(f"  Output dir    : {output_dir}")
    logger.info(f"  Batch size    : {args.batch_size}")
    logger.info(f"  Embedding     : {cfg.EMBEDDING_MODEL}")
    logger.info("=" * 60)

    dataset_path = (
        cfg.DATASET_OPPOSED if args.subset == "opposed" else cfg.DATASET_SUPPORTIVE
    )
    sessions = load_implexconv_dataset(dataset_path, args.subset)

    if args.end_session >= len(sessions):
        logger.error(
            f"end_session={args.end_session} out of range ({len(sessions)} sessions)"
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
        s for s in target_sessions if s.session_id not in completed_ids
    ]
    if not pending_sessions:
        logger.info("All sessions already completed.")
        return 0

    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    runner = CorruptionRunner(
        llm_client=llm_client,
        subset=args.subset,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
        snapshot_base=snapshot_base,
        corrupt_rate=args.corrupt_rate,
        output_dir=output_dir,
        start=args.start_session,
        end=args.end_session,
    )

    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# MemoryBank Corruption  |  subset={args.subset}  |  rate={rate_pct}%")
    print(f"# Model    : {model_name}")
    print(
        f"# Sessions [{args.start_session}, {args.end_session}]"
        f"  ({len(pending_sessions)} to process)"
    )
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_sessions[
            batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size
        ]
        session_ids = [s.session_id for s in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: sessions {session_ids}")

        try:
            batch_results = runner.run_batch(batch)
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
                    checkpoint_file, completed_ids,
                    args.model, args.subset,
                    args.start_session, args.end_session,
                    config_name=config_name,
                )
            logger.info(
                f"Session {result['session_id']} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Corruption: {result['corruption_statistics']}"
            )

    print(f"\n{'#' * 60}")
    print(f"# Done. Results -> {results_file}")
    print(f"{'#' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
