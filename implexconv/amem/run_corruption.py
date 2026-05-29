"""
A-MEM Memory Corruption Experiment Runner — ImplexConv (QA-Only)

Loads Phase 1 memory snapshots, randomly deletes X% of memory notes
(seed = 42 + session_id), then runs Phase 2 QA. Phase 1 is skipped entirely.

Corruption targets:
  - memories dict (MemoryNote entries)
  - retriever.corpus  (text documents, insertion-order aligned)
  - retriever.embeddings (numpy array rows, insertion-order aligned)
  - MemoryNote.links remapped via multi-hop BFS over deleted nodes

Output directory (auto-derived):
  corrupt{rate_pct}_{config_name}_outputs_{model_name}_{subset}/

Usage:
  nohup python amem/run_corruption.py \\
      --model Qwen/Qwen3-1.7B \\
      --subset opposed \\
      --corrupt-rate 0.3 \\
      --start-session 0 --end-session 499 \\
      --tensor-parallel 1 --gpu-memory 0.9 \\
      --config config_0 > logs/corrupt30_amem_opposed.log 2>&1 &
"""

import importlib.util
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

import run_experiment as _re
from run_experiment import (
    BatchedAMEMRunner,
    LLMCallLogger,
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

def _resolve_links_bfs(
    start_links: List[int],
    old_links_map: Dict[int, List[int]],
    deleted_set: Set[int],
) -> List[int]:
    """Multi-hop BFS through deleted nodes; return surviving old-indices only."""
    result = []
    visited: Set[int] = set()
    queue = list(start_links)
    while queue:
        idx = queue.pop(0)
        if idx in visited:
            continue
        visited.add(idx)
        if idx not in deleted_set:
            result.append(idx)
        else:
            queue.extend(old_links_map.get(idx, []))
    return result


def corrupt_memory_system(memory_system, corrupt_rate: float, rng: random.Random) -> Dict:
    """
    Delete corrupt_rate fraction of memories in-place, then:
      - filters retriever.corpus and retriever.embeddings to surviving indices
      - remaps MemoryNote.links via multi-hop BFS over deleted nodes

    Returns corruption metadata dict.
    """
    memories = memory_system.memories   # Dict[str, MemoryNote], insertion-ordered
    retriever = memory_system.retriever  # SimpleEmbeddingRetriever

    uuids = list(memories.keys())
    n_total = len(uuids)
    n_delete = round(n_total * corrupt_rate)

    if n_delete == 0:
        return {"n_before": n_total, "n_deleted": 0, "n_after": n_total}

    delete_uuids = set(rng.sample(uuids, n_delete))
    deleted_old_idx = {i for i, u in enumerate(uuids) if u in delete_uuids}

    # Capture all links before any deletion (BFS needs full graph)
    old_links_map: Dict[int, List[int]] = {
        i: list(memories[uuids[i]].links) for i in range(n_total)
    }

    for u in delete_uuids:
        del memories[u]

    surviving_old_indices = [i for i, u in enumerate(uuids) if u not in delete_uuids]
    old_idx_to_new_idx = {old_i: new_i for new_i, old_i in enumerate(surviving_old_indices)}

    # Remap links for each surviving note
    for old_i, u in zip(surviving_old_indices, [uuids[i] for i in surviving_old_indices]):
        note = memories[u]
        resolved = _resolve_links_bfs(old_links_map[old_i], old_links_map, deleted_old_idx)
        note.links = list(dict.fromkeys(
            old_idx_to_new_idx[o] for o in resolved if o in old_idx_to_new_idx
        ))

    # Filter retriever corpus and embeddings (insertion-order aligned with uuids)
    retriever.corpus = [retriever.corpus[i] for i in surviving_old_indices]
    if retriever.embeddings is not None:
        retriever.embeddings = retriever.embeddings[surviving_old_indices]

    return {"n_before": n_total, "n_deleted": n_delete, "n_after": len(memories)}


# =============================================================================
# CORRUPTION RUNNER
# =============================================================================

class CorruptionRunner(BatchedAMEMRunner):
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
        from agent import BaseAgent

        # Step 1: load snapshot + corrupt per session
        agents = []
        corruption_stats_map: Dict[int, Dict] = {}
        for s in sessions:
            agent = BaseAgent(
                self.llm_client, self.model_path,
                embedding_model=self.shared_embedding_model,
            )
            snap_dir = find_snapshot_dir(self.snapshot_base, s.session_id)
            agent.memory_system.load_snapshot(snap_dir)
            rng = random.Random(42 + s.session_id)
            corruption_stats_map[s.session_id] = corrupt_memory_system(
                agent.memory_system, self.corrupt_rate, rng
            )
            agents.append(agent)

        # Step 2: LLM call loggers
        retrieval_log_paths = [
            self.retrieval_log_dir / f"session_{s.session_id}_retrieval_log.jsonl"
            for s in sessions
        ]
        prompt_log_dirs = [
            self.prompt_log_dir / f"session_{s.session_id}"
            for s in sessions
        ]
        for agent, pld in zip(agents, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                pld.mkdir(parents=True, exist_ok=True)
                agent.set_llm_logger(LLMCallLogger(pld))

        # Step 3: memory stats before QA (Phase 1 skipped)
        memory_stats_list = [agent.get_memory_stats() for agent in agents]

        # Step 4: Phase 2 (inherited)
        qa_results_list, phase2_stats = self._run_phase2_batched(
            sessions, agents, retrieval_log_paths
        )

        # Step 5: snapshots + result assembly
        results = []
        zero_call = {"input": 0, "output": 0, "llm_calls": 0}

        for i, (session, agent) in enumerate(zip(sessions, agents)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snap_out = self.memory_snapshots_dir / f"session_{session.session_id}"
                agent.save_memory_snapshot(snap_out)
            agent.clear_memory()

            p2 = phase2_stats[i]
            token_stats = {
                "call_2_note_construction": {**zero_call, "parse_fallback_count": 0},
                "call_3_evolution":         dict(zero_call),
                "call_4_qa": {
                    "input":     p2["qa_input"],
                    "output":    p2["qa_output"],
                    "llm_calls": p2["num_qa_llm_calls"],
                },
                "total_input":     p2["qa_input"],
                "total_output":    p2["qa_output"],
                "total_llm_calls": p2["num_qa_llm_calls"],
            }

            results.append({
                "session_id":            session.session_id,
                "qa_results":            qa_results_list[i],
                "token_statistics":      token_stats,
                "evolution_statistics":  {
                    "evo_triggered_count": 0,
                    "actions_taken": {"strengthen": 0, "update_neighbor": 0},
                },
                "memory_at_qa_start":    memory_stats_list[i],
                "corruption_statistics": corruption_stats_map[session.session_id],
                "memory_snapshot_path":  (
                    f"memory_snapshots/session_{session.session_id}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            })

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

    # Pre-parse --config to load the right config file before full arg parsing
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
    sys.modules["config"] = cfg  # agent.py does `import config as cfg` at module level
    _re.cfg = cfg                # run_experiment's module-level functions use `cfg`

    parser = argparse.ArgumentParser(
        description="A-MEM Memory Corruption Experiment on ImplexConv",
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
    logger.info("A-MEM Memory Corruption Experiment")
    logger.info(f"  Config        : {config_name}")
    logger.info(f"  Subset        : {args.subset}")
    logger.info(f"  Model         : {args.model}")
    logger.info(f"  Corrupt rate  : {args.corrupt_rate} ({rate_pct}%)")
    logger.info(f"  Sessions      : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Snapshot base : {snapshot_base}")
    logger.info(f"  Output dir    : {output_dir}")
    logger.info(f"  Batch size    : {args.batch_size}")
    logger.info("=" * 60)

    dataset_path = cfg.DATASET_OPPOSED if args.subset == "opposed" else cfg.DATASET_SUPPORTIVE
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
            logger.info(f"Resuming: {len(completed_ids)} sessions already completed")

    pending_sessions = [s for s in target_sessions if s.session_id not in completed_ids]
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
    print(f"# A-MEM Corruption  |  subset={args.subset}  |  rate={rate_pct}%")
    print(f"# Model    : {model_name}")
    print(f"# Sessions [{args.start_session}, {args.end_session}]  ({len(pending_sessions)} to process)")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_sessions[
            batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size
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
    import sys
    sys.exit(main())
