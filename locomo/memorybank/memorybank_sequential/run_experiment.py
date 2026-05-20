"""
MemoryBank Experiment Runner — LoComo (QA-Only Variant)

Runs MemoryBank memory-augmented LLM agent on the LoComo dataset.
Response generation LLM calls are removed (QA-only variant).

Per-sample flow:
  Phase 1 — Memory Construction:
    For each session (ordered), for each turn:
      1. Store turn in memory (embedding only, no LLM)
    After all turns in a session:
      2. Generate session event summary + personality (2 LLM calls)
         → event summary also added to FAISS as searchable document

  Phase 1 End:
    3. Synthesize global summaries from all session summaries (2 LLM calls)
    4. Apply Ebbinghaus forgetting curve (last session's date as "now")

  Phase 2 — QA Answering (memory frozen, no new stores):
    For each QA pair:
      5. Retrieve memories by cosine similarity (no strength update)
      6. Answer QA with category-aware prompt (1 LLM call)
      7. Write retrieval log entry

  Phase 3 — Cleanup:
    Save memory snapshot (if SAVE_MEMORY_SNAPSHOTS)
    Clear memory
    Save results + checkpoint

Usage:
    python run_experiment.py \\
        --start-sample 0 --end-sample 4 \\
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
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

# cfg is loaded dynamically in main() based on --config argument.
cfg = None  # type: ignore

from load_dataset import (
    load_locomo_dataset,
    Sample,
    LoCoMoSession,
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
            return json.load(f).get("last_completed_sample_index")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return None


def save_checkpoint(
    checkpoint_file: Path,
    sample_index: int,
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
):
    data = {
        "last_completed_sample_index": sample_index,
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
        1
        for item in retrieved_items
        if item.get("memory_type") == "dialogue_memory"
        or (
            item.get("memory_type") is None
            and not item.get("is_summary", False)
        )
    )
    summary_count = sum(
        1
        for item in retrieved_items
        if item.get("memory_type") == "session_summary"
        or item.get("is_summary", False)
    )

    return {
        "memory_type": ["dialogue_memory", "session_summary"],
        "num_retrieved": [dialogue_count, summary_count],
        "prompt_context_type": [
            "dialogue_memory",
            "session_summary",
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
) -> Dict:
    """
    Build a retrieval log entry for MemoryBank (LoComo variant).

    module_specific fields:
    - total_memories: total memory entries in the store at retrieval time
    - event_summary_length: chars in current event summary
    - user_portrait_length: chars in current personality portrait
    """
    breakdown = _build_context_breakdown(
        retrieved_items,
        event_summary_length,
        user_portrait_length,
    )
    # Filter out summary items (dia_id=None) for retrieval_scores;
    # keep all for retrieved_items (full logging).
    scores = [
        item["score"] for item in retrieved_items
        if item.get("dia_id") is not None
    ]
    return {
        "timestamp": datetime.now().isoformat(),
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
            "module": "memorybank",
            "total_memories": total_memories,
            "event_summary_length": event_summary_length,
            "user_portrait_length": user_portrait_length,
            "retrieved_item_total": breakdown["retrieved_item_total"],
            "prompt_context_type": breakdown["prompt_context_type"],
            "num_prompt_context": breakdown["num_prompt_context"],
        },
    }


# =============================================================================
# DIALOGUE FORMATTER (for session summarization)
# =============================================================================

def _format_dialogue_for_session(turns: List[Turn]) -> str:
    """Format turns into dialogue text for summarization prompts."""
    return "\n".join(f"[{turn.speaker}]: {turn.text}" for turn in turns)


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
# EXPERIMENT RUNNER
# =============================================================================

class MemoryBankRunner:
    """
    Runs the MemoryBank experiment on a single LoComo sample (QA-only variant).

    Phase 1 — Memory Construction:
        For each session, for each turn:
            Store turn (embedding only)
        After each session:
            Summarize session (event + personality, 2 LLM calls)
            Event summary added to FAISS as searchable document

    Phase 1 End:
        Synthesize global summaries (2 LLM calls)
        Apply forgetting curve (last session's date as "now")

    Phase 2 — QA Answering (memory frozen):
        For each QA: retrieve → answer → log

    Phase 3 — Cleanup:
        Save snapshot → clear → save results + checkpoint
    """

    def __init__(self, agent, model_path: str):
        self.agent = agent
        self.model_path = model_path

    def run_sample(
        self,
        sample: Sample,
        retrieval_log_path: Path,
        snapshots_dir: Path,
    ) -> Dict:
        """Run Phases 1, 1-end, 2, and 3 on a single LoComo sample."""
        total_turns = sum(len(s.turns) for s in sample.sessions)
        logger.info(f"{'='*60}")
        logger.info(
            f"Sample {sample.sample_id}  "
            f"({len(sample.sessions)} sessions, "
            f"{total_turns} turns, "
            f"{len(sample.qa)} QA)"
        )
        logger.info(f"{'='*60}")

        self.agent.clear_memory()

        # ==============================================================
        # Phase 1: Memory Construction
        # ==============================================================
        total_internal_input = total_internal_output = 0
        num_internal_calls = 0

        for session in tqdm(
            sample.sessions,
            desc=f"Sample {sample.sample_id} Phase1 (sessions)",
        ):
            # ---- Store each turn individually (embedding only) ----
            for turn in session.turns:
                # Speaker prefix: [{speaker}]: {text}
                content = f"[{turn.speaker}]: {turn.text}"
                self.agent.add_memory(
                    content=content,
                    dia_id=turn.dia_id,
                    session_id=session.session_id,
                    date_str=session.date_time,
                )

            # ---- Session-level summarization (after all turns stored) ----
            dialogue_text = _format_dialogue_for_session(session.turns)
            self.agent.on_session_end(
                session_id=session.session_id,
                date_str=session.date_time,
                dialogue_text=dialogue_text,
                speaker_a=sample.speaker_a,
                speaker_b=sample.speaker_b,
            )

            summ_tokens = self.agent.get_and_reset_summary_tokens()
            total_internal_input += summ_tokens["input"]
            total_internal_output += summ_tokens["output"]
            num_internal_calls += summ_tokens["api_calls"]

        logger.info(
            f"Phase 1 done: {total_turns} turns stored, "
            f"memory count (pre-forgetting): {self.agent.get_memory_count()}"
        )

        # ==============================================================
        # Phase 1 End: Global Synthesis + Forgetting
        # ==============================================================
        self.agent.on_phase1_end()

        synth_tokens = self.agent.get_and_reset_summary_tokens()
        total_internal_input += synth_tokens["input"]
        total_internal_output += synth_tokens["output"]
        num_internal_calls += synth_tokens["api_calls"]

        logger.info(
            f"Global summaries synthesized. "
            f"Event: {len(self.agent.get_event_summary())} chars, "
            f"Portrait: {len(self.agent.get_user_portrait())} chars"
        )

        # Apply forgetting using the last session's date as "now"
        last_date_time = sample.sessions[-1].date_time
        self.agent.apply_forgetting(last_date_time)

        logger.info(
            f"Forgetting applied (now={last_date_time}). "
            f"Memory count (post-forgetting): {self.agent.get_memory_count()}"
        )

        # ==============================================================
        # Phase 2: QA Answering (memory frozen, no new stores)
        # ==============================================================
        qa_results: List[Dict] = []
        phase2_timings: List[Dict] = []

        total_qa_input = total_qa_output = 0
        num_qa_calls = 0

        for qa in tqdm(sample.qa, desc=f"Sample {sample.sample_id} Phase2"):
            # Step 1: Retrieve using question directly (no strength update)
            t0 = time.time()
            retrieval_result = self.agent.retrieve_memory(
                qa.question,
                k=cfg.RETRIEVE_K,
                update_strength=False,
            )
            t_after_retrieval = time.time()

            # Step 2: Answer QA (category-aware)
            answer, qa_tokens, prompt_snap, qa_api_calls = self.agent.answer_qa(
                question=qa.question,
                retrieved_memory=retrieval_result.formatted,
                category=qa.category,
                adversarial_answer=qa.adversarial_answer or "",
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
                sample_id=sample.sample_id,
                session_id=None,
                dia_id=None,
                query=qa.question,
                retrieved_items=retrieval_result.items,
                total_memories=retrieval_result.total_memories,
                event_summary_length=len(self.agent.get_event_summary()),
                user_portrait_length=len(self.agent.get_user_portrait()),
            )
            write_retrieval_log(retrieval_log_path, log_entry)

            total_qa_input += qa_tokens.get("input", 0)
            total_qa_output += qa_tokens.get("output", 0)
            num_qa_calls += qa_api_calls

            # Build retrieved_memories: only entries with actual dia_ids
            retrieved_metadata = [
                {"dia_id": item["dia_id"]}
                for item in retrieval_result.items
                if item.get("dia_id") is not None
            ]

            qa_results.append({
                "question": qa.question,
                "category": qa.category,
                "generated_answer": answer,
                "ground_truth_answer": qa.final_answer,
                "evidence": qa.evidence,
                "retrieved_memories": retrieved_metadata,
                "qa_tokens": qa_tokens,
            })

        logger.info(f"Phase 2 done: {len(qa_results)} QA answers")

        # ==============================================================
        # Phase 3: Save snapshot, aggregate stats
        # ==============================================================
        snapshot_rel_path = None
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snapshot_dir = snapshots_dir / f"sample_{sample.sample_id}"
            self.agent.save_memory_snapshot(snapshot_dir)
            snapshot_rel_path = f"memory_snapshots/sample_{sample.sample_id}/"
            logger.info(f"Memory snapshot saved to {snapshot_dir}")

        token_stats = {
            "total_qa_input": total_qa_input,
            "total_qa_output": total_qa_output,
            "total_internal_input": total_internal_input,
            "total_internal_output": total_internal_output,
            "total_input": total_qa_input + total_internal_input,
            "total_output": total_qa_output + total_internal_output,
            "num_qa_api_calls": num_qa_calls,
            "num_internal_api_calls": num_internal_calls,
            "num_total_api_calls": num_qa_calls + num_internal_calls,
        }

        timing_stats = {
            "phase2_qa_avg": avg_timing(phase2_timings),
        }

        return {
            "sample_id": sample.sample_id,
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

    # Inject into sys.modules so agent.py / memory_bank.py see it
    sys.modules["config"] = cfg

    from agent import MemoryBankAgent, LLMCallLogger  # noqa: E402

    # ---- Step 2: full argument parsing ----
    parser = argparse.ArgumentParser(
        description="MemoryBank Experiment on LoComo (QA-Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-sample", type=int, required=True,
        help="First sample index (inclusive)",
    )
    parser.add_argument(
        "--end-sample", type=int, required=True,
        help="Last sample index (inclusive)",
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
    if args.start_sample > args.end_sample:
        parser.error("--start-sample must be <= --end-sample")

    cfg.ensure_directories(
        args.model, args.start_sample, args.end_sample, config_name
    )

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
    logger.info("MemoryBank Experiment — LoComo (QA-Only)")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Samples         : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Forgetting div  : {cfg.FORGETTING_DIVISOR}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    samples = load_locomo_dataset(cfg.DATASET_PATH)

    if args.end_sample >= len(samples):
        logger.error(
            f"end_sample={args.end_sample} out of range "
            f"(dataset has {len(samples)} samples)"
        )
        return 1

    target_samples = samples[args.start_sample: args.end_sample + 1]

    # Checkpoint resume
    start_idx = 0
    if cfg.ENABLE_CHECKPOINTING:
        last = load_checkpoint(checkpoint_file)
        if last is not None:
            start_idx = last + 1
            logger.info(f"Resuming from target sample index {start_idx}")

    # Init LLM
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    agent = MemoryBankAgent(llm_client, model_path=args.model)
    runner = MemoryBankRunner(agent, model_path=args.model)
    results = load_existing_results(results_file)

    print(f"\n{'#'*60}")
    print(f"# MemoryBank (QA-Only)  |  LoComo")
    print(f"# Model : {cfg.extract_model_name(args.model)}")
    print(
        f"# Samples [{args.start_sample}, {args.end_sample}]  "
        f"({len(target_samples) - start_idx} to process)"
    )
    print(f"{'#'*60}\n")

    for idx, sample in enumerate(
        target_samples[start_idx:], start=start_idx
    ):
        retrieval_log_path = (
            retrieval_log_dir
            / f"sample_{sample.sample_id}_retrieval_log.jsonl"
        )

        # Set up per-sample LLM call logger
        if cfg.ENABLE_LLM_CALL_LOGGING:
            sample_prompt_log_dir = prompt_log_dir / f"sample_{sample.sample_id}"
            agent.set_llm_logger(LLMCallLogger(sample_prompt_log_dir))
        else:
            agent.set_llm_logger(None)

        try:
            result = runner.run_sample(sample, retrieval_log_path, snapshots_dir)

            # Clear memory for next sample
            agent.clear_memory()
            logger.info(f"Memory cleared after sample {sample.sample_id}")

            results.append(result)
            save_results(results_file, results)

            if cfg.ENABLE_CHECKPOINTING:
                save_checkpoint(
                    checkpoint_file, idx,
                    args.model,
                    args.start_sample, args.end_sample,
                    config_name=config_name,
                )

            logger.info(
                f"Sample {sample.sample_id} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total API calls: "
                f"{result['token_statistics']['num_total_api_calls']}"
            )

        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as e:
            logger.error(f"Error on sample {sample.sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'#'*60}")
    print(f"# Experiment complete!  Results -> {results_file}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
