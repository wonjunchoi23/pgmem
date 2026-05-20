"""
LD-Agent Memory Corruption Experiment Runner

Loads Phase 1 memory snapshots, randomly deletes corrupt_rate fraction of
memory items (LTM, STM, user_traits, agent_traits), then runs Phase 2 QA
on the degraded memory state.

Phase 1 is skipped entirely. token_statistics call_2/call_3 are always zero.
snapshot_base and output_dir are auto-derived from --model/--config/--subset/--corrupt-rate.

Usage:
    CUDA_VISIBLE_DEVICES=0 nohup python run_corruption.py \
        --start-session 0 --end-session 499 \
        --subset opposed \
        --model Qwen/Qwen3-1.7B \
        --corrupt-rate 0.3 \
        --tensor-parallel 1 --gpu-memory 0.34 --max-model-len 30000 \
        --batch-size 25 --config config_0 \
        > nohup/nohup_corrupt30_1.7b_0_499.out 2>&1 &
"""

import os
import math
import random
import sys
import json
import logging
import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None
logger: logging.Logger = logging.getLogger(__name__)

import run_experiment as _re

from load_dataset import load_implexconv_dataset, Session
from ldagent_module import LDAgentModule, LLMCallLogger
from run_experiment import (
    BatchedLDAgentRunner,
    RetrievalLogWriter,
    load_checkpoint,
    save_checkpoint,
    load_existing_results,
    save_results,
    print_completion_banner,
    setup_logging,
    create_llm_client,
)


# =============================================================================
# SNAPSHOT LOOKUP
# =============================================================================


def find_snapshot_dir(snapshot_base: Path, session_id: int) -> Path:
    """Search across parallel session subdirs for the snapshot of a given session."""
    for session_dir in sorted(snapshot_base.iterdir()):
        if not session_dir.name.startswith("session_"):
            continue
        snap_path = session_dir / "memory_snapshots" / f"session_{session_id}"
        if snap_path.exists():
            return snap_path
    raise FileNotFoundError(
        f"Snapshot for session {session_id} not found in {snapshot_base}"
    )


# =============================================================================
# CORRUPTION
# =============================================================================


def _delete_fraction(items: list, corrupt_rate: float, rng: random.Random) -> list:
    """Return items with floor(N * corrupt_rate) entries randomly removed."""
    n = len(items)
    if n == 0:
        return items
    n_delete = math.floor(n * corrupt_rate)
    if n_delete == 0:
        return items
    to_delete = set(rng.sample(range(n), n_delete))
    return [x for i, x in enumerate(items) if i not in to_delete]


def corrupt_agent(
    agent: LDAgentModule,
    corrupt_rate: float,
    rng: random.Random,
) -> Dict:
    """
    Randomly delete corrupt_rate fraction of each memory structure in-place.
    LTM lists (metadata, documents, embeddings) are deleted at the same indices.
    Returns a dict describing the corruption applied, included in config_metadata.
    """
    mb  = agent.memory_bank
    per = agent.personas

    ltm_before          = len(mb._ltm_metadata)
    stm_before          = len(mb.short_term_memory)
    user_traits_before  = len(per.user_traits)
    agent_traits_before = len(per.agent_traits)

    # LTM: delete same indices across all three aligned lists
    if ltm_before > 0:
        n_delete  = math.floor(ltm_before * corrupt_rate)
        to_delete = set(rng.sample(range(ltm_before), n_delete))
        keep = [i for i in range(ltm_before) if i not in to_delete]
        mb._ltm_metadata   = [mb._ltm_metadata[i]   for i in keep]
        mb._ltm_documents  = [mb._ltm_documents[i]  for i in keep]
        mb._ltm_embeddings = [mb._ltm_embeddings[i] for i in keep]

    mb.short_term_memory = _delete_fraction(mb.short_term_memory, corrupt_rate, rng)
    per.user_traits      = _delete_fraction(per.user_traits,      corrupt_rate, rng)
    per.agent_traits     = _delete_fraction(per.agent_traits,     corrupt_rate, rng)

    ltm_after          = len(mb._ltm_metadata)
    stm_after          = len(mb.short_term_memory)
    user_traits_after  = len(per.user_traits)
    agent_traits_after = len(per.agent_traits)

    return {
        "corrupt_rate":    corrupt_rate,
        "random_seed":     42,
        "memories_before": ltm_before + stm_before + user_traits_before + agent_traits_before,
        "memories_after":  ltm_after  + stm_after  + user_traits_after  + agent_traits_after,
        "memories_deleted": (
            (ltm_before - ltm_after)
            + (stm_before - stm_after)
            + (user_traits_before - user_traits_after)
            + (agent_traits_before - agent_traits_after)
        ),
        "detail": {
            "ltm":          {"before": ltm_before,          "after": ltm_after},
            "stm":          {"before": stm_before,          "after": stm_after},
            "user_traits":  {"before": user_traits_before,  "after": user_traits_after},
            "agent_traits": {"before": agent_traits_before, "after": agent_traits_after},
        },
    }


# =============================================================================
# CORRUPTION RUNNER
# =============================================================================


class CorruptionRunner(BatchedLDAgentRunner):
    """
    Replaces Phase 1 with snapshot loading + corruption, then runs
    the inherited _run_phase2_batched().
    """

    def __init__(
        self,
        llm_client,
        model_path:    str,
        subset:        str,
        start_session: int,
        end_session:   int,
        config_name:   str,
        config_metadata:         Dict,
        snapshot_base:           Path,
        corrupt_rate:            float,
        output_dir:              Path,
        save_corrupted_snapshot: bool = False,
        shared_lemma_tokenizer=None,
        shared_encoder=None,
    ):
        super().__init__(
            llm_client=llm_client,
            model_path=model_path,
            subset=subset,
            start_session=start_session,
            end_session=end_session,
            config_name=config_name,
            config_metadata=config_metadata,
            shared_lemma_tokenizer=shared_lemma_tokenizer,
            shared_encoder=shared_encoder,
        )
        self.snapshot_base           = Path(snapshot_base)
        self.corrupt_rate            = corrupt_rate
        self.save_corrupted_snapshot = save_corrupted_snapshot

        # Override paths set by super().__init__() to redirect output to output_dir
        session_subdir = Path(output_dir) / f"session_{start_session}_{end_session}"
        self.retrieval_log_dir    = session_subdir / "retrieval_logs"
        self.memory_snapshots_dir = session_subdir / "memory_snapshots"
        self.prompt_log_dir       = session_subdir / "prompt_log"

    def run_batch(self, sessions: List[Session]) -> List[Dict[str, Any]]:
        agents           = []
        log_writers      = []
        corruption_metas = []

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

            # Load snapshot into the empty agent
            snapshot_dir = find_snapshot_dir(self.snapshot_base, session.session_id)
            agent.memory_bank.load_snapshot(snapshot_dir)
            agent.personas.load_snapshot(snapshot_dir)
            logger.info(
                f"Session {session.session_id}: snapshot loaded "
                f"(LTM={agent.memory_bank.get_memory_count()}, "
                f"STM={agent.memory_bank.get_short_term_count()}, "
                f"user_traits={agent.personas.get_user_trait_count()}, "
                f"agent_traits={agent.personas.get_agent_trait_count()})"
            )

            # Corrupt — seed is 42 + session_id for per-session reproducibility
            session_rng = random.Random(42 + session.session_id)
            meta = corrupt_agent(agent, self.corrupt_rate, session_rng)
            corruption_metas.append(meta)
            logger.info(
                f"Session {session.session_id}: corrupted "
                f"{meta['memories_deleted']}/{meta['memories_before']} items "
                f"(LTM {meta['detail']['ltm']['before']}→{meta['detail']['ltm']['after']}, "
                f"STM {meta['detail']['stm']['before']}→{meta['detail']['stm']['after']}, "
                f"traits "
                f"{meta['detail']['user_traits']['before'] + meta['detail']['agent_traits']['before']}"
                f"→{meta['detail']['user_traits']['after'] + meta['detail']['agent_traits']['after']})"
            )

            agents.append(agent)
            log_writers.append(
                RetrievalLogWriter(
                    self.retrieval_log_dir / f"session_{session.session_id}_retrieval_log.jsonl"
                )
            )

        # Capture memory state after corruption, before QA
        memory_stats_list = [agent.get_memory_stats() for agent in agents]

        # Phase 2 — inherited from BatchedLDAgentRunner
        qa_results_list, phase2_stats = self._run_phase2_batched(sessions, agents, log_writers)

        results = []
        for i, (session, agent) in enumerate(zip(sessions, agents)):
            memory_snapshot_path = None
            if self.save_corrupted_snapshot:
                snap_dir = self.memory_snapshots_dir / f"session_{session.session_id}"
                agent.save_snapshot(snap_dir)
                memory_snapshot_path = f"memory_snapshots/session_{session.session_id}"

            agent.clear()

            p2 = phase2_stats[i]
            total_input     = p2["call_4_summarization"]["input"]     + p2["call_5_qa"]["input"]
            total_output    = p2["call_4_summarization"]["output"]    + p2["call_5_qa"]["output"]
            total_llm_calls = p2["call_4_summarization"]["llm_calls"] + p2["call_5_qa"]["llm_calls"]

            results.append({
                "session_id":         session.session_id,
                "config_metadata":    {**self.config_metadata, "corruption": corruption_metas[i]},
                "memory_at_qa_start": memory_stats_list[i],
                "qa_results":         qa_results_list[i],
                "token_statistics": {
                    "call_2_user_persona":  {"input": 0, "output": 0, "llm_calls": 0},
                    "call_3_agent_persona": {"input": 0, "output": 0, "llm_calls": 0},
                    "call_4_summarization": p2["call_4_summarization"],
                    "call_5_qa":            p2["call_5_qa"],
                    "total_input":          total_input,
                    "total_output":         total_output,
                    "total_llm_calls":      total_llm_calls,
                },
                "memory_snapshot_path": memory_snapshot_path,
            })

        return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_0")
    pre_args, _ = pre_parser.parse_known_args()

    global cfg
    cfg = importlib.import_module(Path(pre_args.config).stem)

    # Inject cfg and logger into run_experiment's namespace so inherited methods see them
    _re.cfg = cfg

    parser = argparse.ArgumentParser(description="LD-Agent Memory Corruption Experiment")
    parser.add_argument("--corrupt-rate",    type=float, required=True,
                        help="Fraction of memory items to delete (0.0–1.0)")
    parser.add_argument("--start-session",   type=int, required=True)
    parser.add_argument("--end-session",     type=int, required=True)
    parser.add_argument("--subset",          type=str, required=True, choices=["opposed", "supportive"])
    parser.add_argument("--model",           type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory",      type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len",   type=int, default=None)
    parser.add_argument("--batch-size",      type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config",          type=str, default="config_0")
    parser.add_argument("--save-corrupted-snapshot", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.corrupt_rate <= 1.0:
        parser.error("--corrupt-rate must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")

    corrupt_rate_pct = int(args.corrupt_rate * 100)
    config_name      = Path(args.config).stem
    model_name       = cfg.extract_model_name(args.model)

    # Auto-derive snapshot_base and output_dir from model/config/subset/corrupt_rate
    snapshot_base = cfg.get_output_dir(args.model, args.subset, config_name)
    output_dir    = cfg.BASE_OUTPUT_DIR / f"corrupt{corrupt_rate_pct}_{config_name}_outputs_{model_name}_{args.subset}"
    session_subdir = output_dir / f"session_{args.start_session}_{args.end_session}"

    # Create output directories
    session_subdir.mkdir(parents=True, exist_ok=True)
    (session_subdir / "retrieval_logs").mkdir(exist_ok=True)
    (session_subdir / "prompt_log").mkdir(exist_ok=True)
    if args.save_corrupted_snapshot:
        (session_subdir / "memory_snapshots").mkdir(exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    global logger
    logger = setup_logging(session_subdir / "logs")
    _re.logger = logger

    results_file    = session_subdir / f"results_{model_name}_{args.subset}_session_{args.start_session}_{args.end_session}.json"
    checkpoint_file = session_subdir / f"checkpoint_{model_name}_{args.subset}_session_{args.start_session}_{args.end_session}.json"

    logger.info("=" * 60)
    logger.info("LD-Agent Memory Corruption Experiment")
    logger.info(f"  Config        : {config_name}")
    logger.info(f"  Subset        : {args.subset}")
    logger.info(f"  Model         : {args.model}")
    logger.info(f"  Sessions      : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Corrupt rate  : {args.corrupt_rate} ({corrupt_rate_pct}%)")
    logger.info(f"  Snapshot base : {snapshot_base}")
    logger.info(f"  Output dir    : {output_dir}")
    logger.info("=" * 60)

    dataset_path = cfg.DATASET_OPPOSED if args.subset == "opposed" else cfg.DATASET_SUPPORTIVE
    sessions = load_implexconv_dataset(dataset_path)
    if args.end_session >= len(sessions):
        logger.error(f"end_session={args.end_session} out of range ({len(sessions)} sessions)")
        return 1

    target_sessions  = sessions[args.start_session:args.end_session + 1]
    completed_ids    = load_checkpoint(checkpoint_file) if cfg.ENABLE_CHECKPOINTING else set()
    pending_sessions = [s for s in target_sessions if s.session_id not in completed_ids]

    if not pending_sessions:
        logger.info("All sessions already completed.")
        return 0

    logger.info("Loading shared spaCy lemma tokenizer...")
    import spacy
    try:
        shared_lemma_tokenizer = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        shared_lemma_tokenizer = spacy.load("en_core_web_sm")

    logger.info("Loading shared SentenceTransformer encoder (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Encoder ready.")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
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

    runner = CorruptionRunner(
        llm_client=llm_client,
        model_path=args.model,
        subset=args.subset,
        start_session=args.start_session,
        end_session=args.end_session,
        config_name=config_name,
        config_metadata=config_metadata,
        snapshot_base=snapshot_base,
        corrupt_rate=args.corrupt_rate,
        output_dir=output_dir,
        save_corrupted_snapshot=args.save_corrupted_snapshot,
        shared_lemma_tokenizer=shared_lemma_tokenizer,
        shared_encoder=shared_encoder,
    )

    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# LD-Agent Corruption — {corrupt_rate_pct}%")
    print(f"# Model    : {model_name}")
    print(f"# Sessions : [{args.start_session}, {args.end_session}] ({len(pending_sessions)} to process)")
    print(f"# Batch    : {args.batch_size}")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch       = pending_sessions[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        session_ids = [s.session_id for s in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: sessions {session_ids}")

        try:
            batch_results = runner.run_batch(batch)
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
                    checkpoint_file, completed_ids, args.model,
                    args.subset, args.start_session, args.end_session, config_name,
                )
            logger.info(f"Session {result['session_id']} complete.")

    logger.info("All sessions completed.")
    print_completion_banner(results_file, len(target_sessions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
