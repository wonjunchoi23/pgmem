"""
Theanine Experiment Runner — ImplexConv (QA-Only Variant)

Runs the Theanine memory-augmented LLM agent on the ImplexConv dataset.
Response generation LLM calls are removed (QA-only variant).

Per-session flow:

  Phase 1 — Memory Construction (no LLM calls per turn):
    Memory accumulates across conv_ids within a session (cleared between sessions).
    For each (user_turn, assistant_turn) pair:
      1. Detect conv_id boundary → finalize previous conv (memory construction)
      2. Retrieve timeline paths (cosine retrieval + path traversal)
      3. Write retrieval log entry
      4. Update current_dialogue accumulator with GT turns
    After all turns: finalize the last conv

  Phase 2 — QA Answering (memory frozen, no new construction):
    For each QA pair:
      1. Retrieve top-k memory node summaries (cosine similarity)
      2. Generate QA answer (timed, all questions)
      3. Write retrieval log entry

  Phase 3 — Cleanup:
    Save memory snapshot (if SAVE_MEMORY_SNAPSHOTS)
    Aggregate token statistics
    Save results + checkpoint
    Clear memory

Token tracking categories:
  call_3_summarization → finalize_conv summarization LLM call
  call_4_relation      → finalize_conv relation extraction LLM calls
  call_2_refinement    → refine_all (timeline refinement, Phase 2 only)
  call_5_qa            → answer_qa LLM call

Usage:
    python run_experiment.py \\
        --start-session 0 --end-session 4 \\
        --subset opposed \\
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
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Suppress noisy logs before any imports
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

# cfg is loaded dynamically in main() based on --config argument.
# All references to cfg inside functions are resolved at call time.
cfg = None  # type: ignore

from load_dataset import (
    load_implexconv_dataset,
    Session,
    Turn,
    QAPair,
)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"theanine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
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


def save_checkpoint(checkpoint_file: Path, session_index: int,
                    model_path: str, subset: str,
                    start_session: int, end_session: int,
                    config_name: str = "config"):
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
    use_timelines: Optional[List] = None,
    timeline_info: Optional[List[Dict]] = None,
    current_dialogue_turns: int = 0,
) -> Dict:
    """
    Build a retrieval log entry for Theanine.

    Theanine module_specific fields:
    - use_timelines: Sampled path tuples used for refinement
    - timeline_info: All retrieved nodes and their full path sets
    - num_paths_used: Number of paths refined and passed to Generator

    Both Phase 1 ("memory_construction") and Phase 2 ("qa") use the same
    timeline pipeline, so the log format is identical across all turns.
    """
    # Always compute seed/path_linked counts — zero when memory is empty
    if use_timelines:
        seed_ids = {item["memory_id"] for item in retrieved_items}
        path_node_ids = {
            elem
            for path in use_timelines
            for i, elem in enumerate(path)
            if i % 2 == 0   # even indices = node_id, odd = relation
        }
        num_path_linked = len(path_node_ids - seed_ids)
    else:
        num_path_linked = 0
    memory_type   = ["seed", "path_linked", "current_dialogue"]
    num_retrieved = [len(retrieved_items), num_path_linked, current_dialogue_turns]

    return {
        "phase": phase,
        "session_id": session_id,
        "conv_id": conv_id,
        "turn_id": turn_id,
        "query": query[:500],   # truncate very long queries
        "memory_type":   memory_type,
        "num_retrieved": num_retrieved,
        "retrieved_items": retrieved_items,
        "retrieval_scores": [item["score"] for item in retrieved_items],
        "module_specific": {
            "module": "theanine",
            "num_paths_used": len(use_timelines) if use_timelines else 0,
            "use_timelines": [list(p) for p in use_timelines] if use_timelines else [],
            "timeline_info": timeline_info or [],
        },
    }


# =============================================================================
# DIALOGUE HELPERS
# =============================================================================

def build_conv_dialogue(turns: List[Turn]) -> str:
    """Format a list of GT turns into a dialogue string for finalize_conv."""
    return "\n".join(t.to_message() for t in turns)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class TheanineRunner:
    """
    Runs the Theanine experiment on a single session.

    Memory is constructed at conv_id boundaries (finalize_conv).
    current_dialogue accumulates within each conv_id and resets when conv_id changes.

    Phase 1:
      For each turn pair, detect conv_id boundary → finalize prev conv → retrieve
      timeline (cosine + path traversal) → write retrieval log.  No LLM calls per turn.

    Phase 2:
      For each QA: timeline retrieval → refine (call_2_refinement) → generate answer (call_5_qa).
    """

    def __init__(self, module, subset: str, model_path: str, config_metadata: Dict):
        self.module          = module
        self.subset          = subset
        self.model_path      = model_path
        self.config_metadata = config_metadata

    def run_session(self, session: Session, retrieval_log_path: Path,
                    snapshots_dir: Path) -> Dict:
        """
        Run Phases 1–3 on a single session.
        Returns a session result dict matching the output schema.
        """
        logger.info(f"{'='*60}")
        logger.info(f"Session {session.session_id}  "
                    f"({len(session.get_turn_pairs())} turn pairs, "
                    f"{len(session.qa)} QA)")
        logger.info(f"{'='*60}")

        self.module.clear()

        # ==============================================================
        # Phase 1: Memory Construction (no response generation LLM call)
        # ==============================================================
        # Per-type token counters
        summ_input = summ_output = summ_calls = 0
        summ_fallback_count = 0
        rel_input = rel_output = rel_calls = 0

        # Conv-level state
        current_conv_id:    Optional[int]  = None
        current_day:        int            = -1   # virtual day index (conv_id // CONV_IDS_PER_DAY)
        current_dialogue:   str            = ""   # GT turn accumulator; resets at each day boundary
        finalize_turns:     List[Turn]     = []   # GT turns for the next finalize_conv batch (reset after finalize)
        pending_count:      int            = 0    # completed conv_ids not yet finalized

        turn_pairs = session.get_turn_pairs()

        for user_turn, assistant_turn in tqdm(turn_pairs,
                                              desc=f"S{session.session_id} Phase1"):

            # ── Conv boundary: accumulate N conv_ids, then finalize ──────────
            if current_conv_id is not None and user_turn.conv_id != current_conv_id:
                pending_count += 1  # one more conv_id has just completed

                if pending_count >= cfg.FINALIZE_EVERY_N_CONVS:
                    full_conv_dialogue = build_conv_dialogue(finalize_turns)
                    mem_tokens = self.module.finalize_conv(
                        current_conv_id, session.session_id, full_conv_dialogue,
                        turn_id_start=finalize_turns[0].turn_id,
                        turn_id_end=finalize_turns[-1].turn_id,
                        global_turn_id_start=finalize_turns[0].global_turn_id,
                        global_turn_id_end=finalize_turns[-1].global_turn_id,
                    )
                    summ_input          += mem_tokens["call_3_summarization"]["input"]
                    summ_output         += mem_tokens["call_3_summarization"]["output"]
                    summ_calls          += mem_tokens["call_3_summarization"]["llm_calls"]
                    summ_fallback_count += mem_tokens["call_3_summarization"]["parse_fallback_count"]
                    rel_input           += mem_tokens["call_4_relation"]["input"]
                    rel_output          += mem_tokens["call_4_relation"]["output"]
                    rel_calls           += mem_tokens["call_4_relation"]["llm_calls"]

                    # Reset only the finalize batch — current_dialogue keeps accumulating
                    finalize_turns = []
                    pending_count  = 0
                # else: keep accumulating — do not reset

            current_conv_id = user_turn.conv_id

            # ── Day boundary: reset current_dialogue at each new virtual day ──
            new_day = user_turn.conv_id // cfg.CONV_IDS_PER_DAY
            if new_day != current_day:
                current_dialogue = ""
                current_day = new_day

            # ── Step 1: Build query = current user turn only ─────────────────
            # Using only the current turn (not accumulated dialogue) gives a
            # more focused embedding query, improving retrieval relevance.
            user_line = user_turn.to_message()
            query_for_retrieval = user_line.strip()

            # ── Step 2: Retrieve timeline ────────────────────────────────────
            timelines, log_items = self.module.retrieve_for_response(query_for_retrieval)

            # ── Step 3: Write retrieval log ──────────────────────────────────
            dialogue_with_user = (current_dialogue + user_line + "\n").strip()
            dialogue_turns = sum(
                1 for l in dialogue_with_user.split("\n") if l.strip()
            )
            log_entry = build_retrieval_log_entry(
                phase="memory_construction",
                session_id=session.session_id,
                conv_id=user_turn.conv_id,
                turn_id=user_turn.turn_id,
                query=query_for_retrieval,
                retrieved_items=log_items,
                use_timelines=timelines.get("use_timeline", []),
                timeline_info=timelines.get("timeline", []),
                current_dialogue_turns=dialogue_turns,
            )
            write_retrieval_log(retrieval_log_path, log_entry)

            # ── Step 4: Update dialogue accumulator with GT turns ────────────
            # GT assistant turn used — not the generated response.
            gt_response = assistant_turn.utterance if assistant_turn else ""

            current_dialogue += user_line + "\n"
            if assistant_turn:
                current_dialogue += assistant_turn.to_message() + "\n"

            finalize_turns.append(user_turn)
            if assistant_turn:
                finalize_turns.append(assistant_turn)

        # ── Finalize remaining pending convs at end of Phase 1 ──────────────
        # Handles convs that didn't reach the FINALIZE_EVERY_N_CONVS threshold
        # (e.g. last 1 conv when N=2), or the very last conv of the session.
        if finalize_turns:
            full_conv_dialogue = build_conv_dialogue(finalize_turns)
            mem_tokens = self.module.finalize_conv(
                current_conv_id, session.session_id, full_conv_dialogue,
                turn_id_start=finalize_turns[0].turn_id,
                turn_id_end=finalize_turns[-1].turn_id,
                global_turn_id_start=finalize_turns[0].global_turn_id,
                global_turn_id_end=finalize_turns[-1].global_turn_id,
            )
            summ_input          += mem_tokens["call_3_summarization"]["input"]
            summ_output         += mem_tokens["call_3_summarization"]["output"]
            summ_calls          += mem_tokens["call_3_summarization"]["llm_calls"]
            summ_fallback_count += mem_tokens["call_3_summarization"]["parse_fallback_count"]
            rel_input           += mem_tokens["call_4_relation"]["input"]
            rel_output          += mem_tokens["call_4_relation"]["output"]
            rel_calls           += mem_tokens["call_4_relation"]["llm_calls"]

        # Save full session dialogue for Phase 2 QA context
        final_dialogue = current_dialogue

        logger.info(f"Phase 1 done: {len(turn_pairs)} turns, "
                    f"memory nodes: {self.module.get_memory_count()}")

        # Record memory state before QA begins (memory frozen from here)
        memory_at_qa_start = self.module.get_memory_stats()

        # ==============================================================
        # Phase 2: QA Answering (memory frozen, no new construction)
        # ==============================================================
        qa_results: List[Dict] = []

        refine_input = refine_output = refine_calls = 0
        qa_input = qa_output = qa_calls = 0

        # Append question as the last User turn so QA is treated as a
        # continuation of the full session dialogue.  All questions share the
        # same final_dialogue base — questions are independent of each other.
        for qa in tqdm(session.qa, desc=f"S{session.session_id} Phase2"):
            qa_dialogue = (final_dialogue + f"User: {qa.question}\n").strip()

            # Step 1: Timeline retrieval
            timelines, log_items = self.module.retrieve_for_response(qa.question)

            # Build retrieved_memories metadata for output schema
            retrieved_memories = [
                {
                    "session_id": item["source_turn"]["session_id"],
                    "conv_id":    item["source_turn"]["conv_id"],
                    "turn_id":    item["source_turn"]["turn_id_start"],
                }
                for item in log_items
            ]

            # Step 2: Refine timelines + generate QA answer
            answer, qa_tokens, refine_tokens, prompt_snap = self.module.answer_qa(
                qa.question, timelines, self.subset,
                current_dialogue=qa_dialogue,
            )

            refine_input += refine_tokens.get("input", 0)
            refine_output += refine_tokens.get("output", 0)
            refine_calls  += refine_tokens.get("llm_calls", 0)

            # Step 3: Write retrieval log
            qa_dialogue_turns = sum(1 for l in qa_dialogue.split("\n") if l.strip())
            log_entry = build_retrieval_log_entry(
                phase="qa",
                session_id=session.session_id,
                conv_id=-1,
                turn_id=-1,
                query=qa.question,
                retrieved_items=log_items,
                use_timelines=timelines.get("use_timeline", []),
                timeline_info=timelines.get("timeline", []),
                current_dialogue_turns=qa_dialogue_turns,
            )
            write_retrieval_log(retrieval_log_path, log_entry)

            qa_input += qa_tokens.get("input", 0)
            qa_output += qa_tokens.get("output", 0)
            qa_calls  += 1

            qa_results.append({
                "question":            qa.question,
                "generated_answer":    answer,
                "ground_truth_answer": qa.answer,
                "retrieved_memories":  retrieved_memories,
                "qa_tokens":           qa_tokens,
            })

        logger.info(f"Phase 2 done: {len(qa_results)} QA answers")

        # ==============================================================
        # Phase 3: Save snapshot, aggregate stats
        # ==============================================================
        snapshot_rel_path = None
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snapshot_dir = snapshots_dir / f"session_{session.session_id}"
            self.module.save_memory_snapshot(snapshot_dir)
            snapshot_rel_path = f"memory_snapshots/session_{session.session_id}/"
            logger.info(f"Memory snapshot saved to {snapshot_dir}")

        token_stats = {
            "call_3_summarization": {
                "input":               summ_input,
                "output":              summ_output,
                "llm_calls":           summ_calls,
                "parse_fallback_count": summ_fallback_count,
            },
            "call_4_relation": {
                "input":     rel_input,
                "output":    rel_output,
                "llm_calls": rel_calls,
            },
            "call_2_refinement": {
                "input":     refine_input,
                "output":    refine_output,
                "llm_calls": refine_calls,
            },
            "call_5_qa": {
                "input":     qa_input,
                "output":    qa_output,
                "llm_calls": qa_calls,
            },
            "total_input":     summ_input + rel_input + refine_input + qa_input,
            "total_output":    summ_output + rel_output + refine_output + qa_output,
            "total_llm_calls": summ_calls + rel_calls + refine_calls + qa_calls,
        }

        return {
            "session_id":           session.session_id,
            "config_metadata":      self.config_metadata,
            "memory_at_qa_start":   memory_at_qa_start,
            "qa_results":           qa_results,
            "token_statistics":     token_stats,
            "memory_snapshot_path": snapshot_rel_path,
        }


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def create_llm_client(model_path: str, tensor_parallel: int, gpu_memory: float,
                      max_model_len: Optional[int] = None):
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
    # ── Step 1: parse --config first so we can load the right module ──────────
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_0")
    pre_args, _ = pre_parser.parse_known_args()
    config_name = Path(pre_args.config).stem  # strip .py if provided

    import importlib.util
    config_file = _MODULE_DIR / f"{config_name}.py"
    if not config_file.exists():
        print(f"[error] Config file not found: {config_file}")
        return 1
    spec = importlib.util.spec_from_file_location(config_name, config_file)
    global cfg
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # Inject into sys.modules so all sub-modules (memory_graph, timeline, generator)
    # see the same config via `import config as cfg`.
    sys.modules["config"] = cfg

    # Import TheanineModule AFTER config is injected
    from theanine_module import TheanineModule, LLMCallLogger  # noqa: E402

    # ── Step 2: full argument parsing ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Theanine Experiment on ImplexConv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-session", type=int, required=True,
                        help="First session index (inclusive)")
    parser.add_argument("--end-session",   type=int, required=True,
                        help="Last session index (inclusive)")
    parser.add_argument("--subset",        type=str, required=True,
                        choices=["opposed", "supportive"],
                        help="Dataset subset to use")
    parser.add_argument("--model",         type=str,
                        default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory",    type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override vLLM max_model_len "
                             "(default: use llm_client.py MODEL_MAX_LENGTHS)")
    parser.add_argument("--config",        type=str, default="config_0",
                        help="Config file name (without .py). Used to load settings "
                             "and as output directory prefix.")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")

    cfg.ensure_directories(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )

    session_dir = cfg.get_session_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file    = cfg.get_results_file(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    checkpoint_file = cfg.get_checkpoint_file(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    retrieval_log_dir = cfg.get_retrieval_log_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    snapshots_dir = cfg.get_memory_snapshots_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    prompt_log_dir = cfg.get_prompt_log_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )

    logger.info("=" * 60)
    logger.info("Theanine Experiment")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Subset          : {args.subset}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Linking top_j   : {cfg.LINKING_TOP_J}")
    logger.info(f"  Retrieve top_k  : {cfg.RETRIEVE_TOP_K}")
    logger.info(f"  QA retrieve k   : {cfg.QA_RETRIEVE_K}")
    logger.info(f"  Timing conv_id  : {cfg.TIMING_CONV_ID}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  Conv IDs/day    : {cfg.CONV_IDS_PER_DAY}  (current_dialogue resets daily)")
    logger.info(f"  Minutes/turn    : {cfg.MINUTES_PER_TURN}")
    logger.info(f"  Finalize every  : {cfg.FINALIZE_EVERY_N_CONVS} conv_ids (= 1 day)")
    logger.info(f"  LLM call logging: {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_path = cfg.DATASET_OPPOSED if args.subset == "opposed" else cfg.DATASET_SUPPORTIVE
    sessions = load_implexconv_dataset(dataset_path, args.subset)

    if args.end_session >= len(sessions):
        logger.error(
            f"end_session={args.end_session} out of range "
            f"(dataset has {len(sessions)} sessions)"
        )
        return 1

    target_sessions = sessions[args.start_session: args.end_session + 1]

    # Checkpoint resume
    start_idx = 0
    if cfg.ENABLE_CHECKPOINTING:
        last = load_checkpoint(checkpoint_file)
        if last is not None:
            start_idx = last + 1
            logger.info(f"Resuming from target session index {start_idx}")

    # Init LLM client
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    config_metadata = {
        "config_name":            config_name,
        "model":                  args.model,
        "subset":                 args.subset,
        "embedding_model":        cfg.EMBEDDING_MODEL,
        "retrieve_top_k":         cfg.RETRIEVE_TOP_K,
        "linking_top_j":          cfg.LINKING_TOP_J,
        "qa_retrieve_k":          cfg.QA_RETRIEVE_K,
        "conv_ids_per_day":       cfg.CONV_IDS_PER_DAY,
        "finalize_every_n_convs": cfg.FINALIZE_EVERY_N_CONVS,
        "temperature":            cfg.TEMPERATURE,
        "max_tokens":             cfg.MAX_TOKENS,
        "session_range":          [args.start_session, args.end_session],
    }

    module = TheanineModule(llm_client, model_path=args.model)
    runner = TheanineRunner(
        module, subset=args.subset, model_path=args.model,
        config_metadata=config_metadata,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#'*60}")
    print(f"# Theanine  |  subset={args.subset}")
    print(f"# Model : {cfg.extract_model_name(args.model)}")
    print(f"# Sessions [{args.start_session}, {args.end_session}]  "
          f"({len(target_sessions) - start_idx} to process)")
    print(f"{'#'*60}\n")

    for idx, session in enumerate(target_sessions[start_idx:], start=start_idx):
        retrieval_log_path = (
            retrieval_log_dir / f"session_{session.session_id}_retrieval_log.jsonl"
        )

        # Set up per-session LLM call logger
        if cfg.ENABLE_LLM_CALL_LOGGING:
            session_prompt_log_dir = prompt_log_dir / f"session_{session.session_id}"
            module.set_llm_logger(LLMCallLogger(session_prompt_log_dir))
        else:
            module.set_llm_logger(None)

        try:
            result = runner.run_session(session, retrieval_log_path, snapshots_dir)

            # Clear memory for next session (also done inside run_session at start,
            # but explicit clear here ensures clean state on resume)
            module.clear()
            logger.info(f"Memory cleared after session {session.session_id}")

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
                f"Turns: {len(session.get_turn_pairs())}, "
                f"QA: {len(result['qa_results'])}, "
                f"Total LLM calls: {result['token_statistics']['total_llm_calls']}"
            )

        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as e:
            logger.error(f"Error on session {session.session_id}: {e}")
            import traceback
            traceback.print_exc()
            module.clear()
            raise

    print(f"\n{'#'*60}")
    print(f"# Experiment complete!  Results -> {results_file}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
