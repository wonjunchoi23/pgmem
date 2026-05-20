"""
LD-Agent Experiment Runner (ImplexConv) — QA-Only Variant

Runs the LD-Agent memory-augmented LLM agent on the ImplexConv dataset following
the QA-only protocol defined in global_readme.md.

Response generation LLM calls are removed. Phase 1 constructs and logs the response
prompt each turn but does NOT call the LLM. Memory is built identically using GT
agent responses. Only QA (Task 2) is evaluated.

Per-session flow:
  Phase 1 — Memory Construction (no response generation LLM call):
    For each (user_turn, assistant_turn) pair:
      1. Retrieve STM context + LTM memories
      2. Build response prompt + count input tokens (NO LLM call, prompt logged)
      3. Write LTM retrieval log entry (phase="prompt_construction")
      4. Update personas (user before, agent after using GT)
      5. Store GT response in STM

  Phase 2 — QA Answering (after all turns, memory frozen):
    flush_to_ltm() — commit remaining STM to LTM
    For each QA pair:
      1. Retrieve LTM memories
      2. Answer QA (supportive: yes/no/unknown; opposed: free-form)
      3. Write LTM retrieval log entry

  Phase 3 — Cleanup:
    Save memory snapshot → clear memory → save results + checkpoint

Usage:
CUDA_VISIBLE_DEVICES=0 nohup python run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 --gpu-memory 0.11 \
    --max-model-len 3000 \
    --config config_0 \
    > nohup/nohup_opp_1.7b_session_0_29.out 2>&1 &
"""

import os
import re
import sys
import json
import logging
import argparse
import importlib
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

# cfg is loaded dynamically in main() via --config argument.
# All references to cfg inside functions are resolved at call time.
cfg = None  # type: ignore

from load_dataset import load_implexconv_dataset, Session, Turn, QAPair
from ldagent_module import LDAgentModule, LLMCallLogger


# =============================================================================
# LOGGING
# =============================================================================

logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt      = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"ldagent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format=fmt,
        handlers=handlers,
    )
    return logging.getLogger(__name__)


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
    session_index:   int,
    model_path:      str,
    subset:          str,
    start_session:   int,
    end_session:     int,
    config_name:     str,
):
    checkpoint = {
        "last_completed_session_index": session_index,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model":         model_path,
            "subset":        subset,
            "start_session": start_session,
            "end_session":   end_session,
            "config_name":   config_name,
        },
    }
    _atomic_write(checkpoint_file, checkpoint)


def load_existing_results(results_file: Path) -> List[Dict]:
    if not results_file.exists():
        return []
    try:
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results from {results_file}")
        return results
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted results JSON ({e}); starting fresh.")
        return []
    except Exception as e:
        logger.warning(f"Could not load results ({e}); starting fresh.")
        return []


def save_results(results_file: Path, results: List[Dict]):
    _atomic_write(results_file, results)


def _atomic_write(path: Path, obj):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


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
    phase:          str,
    session_id:     int,
    conv_id:        int,
    turn_id:        int,
    retrieval_data: Dict,
) -> Dict:
    """Build a standardized retrieval log entry from ldagent_module retrieval_log_data."""
    relevant_memories = retrieval_data.get("relevant_memories", [])

    retrieved_items = []
    for mem in relevant_memories:
        retrieved_items.append({
            "memory_id":       str(mem.get("idx", "")),
            "content_preview": (mem.get("summary") or mem.get("dialog") or "")[:100],
            "score":           mem.get("score", 0.0),
            "source_turn": {
                "session_id":      mem.get("session_id",      0),
                "conv_id":         mem.get("conv_id",         0),
                "virtual_seconds": mem.get("virtual_seconds", 0.0),
            },
        })

    module_specific = retrieval_data.get("module_specific", {})
    return {
        "timestamp":    datetime.now().isoformat(),
        "phase":        phase,
        "session_id":   session_id,
        "conv_id":      conv_id,
        "turn_id":      turn_id,
        "query":        retrieval_data.get("query", ""),
        "memory_type":  ["ltm", "stm", "user_trait", "agent_trait"],
        "num_retrieved": [
            len(retrieved_items),
            module_specific.get("stm_context_turns", 0),
            module_specific.get("user_trait_count",  0),
            module_specific.get("agent_trait_count", 0),
        ],
        "retrieved_items": retrieved_items,
        "module_specific": module_specific,
    }


# =============================================================================
# QA ANSWER NORMALIZATION
# =============================================================================

def _normalize_yes_no_unknown(answer: str) -> str:
    """Normalize a generated answer to one of {yes, no, unknown} for supportive subset."""
    a = answer.strip().lower()
    if a in ("yes", "no", "unknown"):
        return a
    if re.match(r"^yes\b", a):
        return "yes"
    if re.match(r"^no\b", a):
        return "no"
    return "unknown"


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class LDAgentExperimentRunner:
    """
    Orchestrates the LD-Agent experiment on individual sessions (QA-only variant).

    Per session S_i:
      Phase 1 — process all turns (memory construction, no response generation LLM call)
      Phase 2 — QA answering (memory frozen after flush_to_ltm)
      Phase 3 — snapshot, clear, checkpoint
    """

    def __init__(
        self,
        llm_client,
        model_path:    str,
        subset:        str,
        start_session: int,
        end_session:   int,
        config_name:   str,
    ):
        self.llm_client    = llm_client
        self.model_path    = model_path
        self.subset        = subset
        self.start_session = start_session
        self.end_session   = end_session
        self.config_name   = config_name

        # Pre-compute directory paths (used across sessions)
        self.retrieval_log_dir = cfg.get_retrieval_log_dir(
            model_path, subset, start_session, end_session, config_name)
        self.memory_snapshots_dir = cfg.get_memory_snapshots_dir(
            model_path, subset, start_session, end_session, config_name)
        self.prompt_log_dir = cfg.get_prompt_log_dir(
            model_path, subset, start_session, end_session, config_name)

    # -------------------------------------------------------------------------
    # Phase 1 helpers
    # -------------------------------------------------------------------------

    def _process_phase1(
        self,
        session:     Session,
        agent:       LDAgentModule,
        log_writer:  RetrievalLogWriter,
    ) -> Tuple[int, Dict]:
        """
        Process all turns in a session (Phase 1, QA-only variant).

        Response prompt is constructed and logged per turn but NO LLM call is made.
        Memory is constructed identically to the base protocol using GT responses.

        Returns:
            num_turns:   number of turn pairs processed
            token_accum: accumulated token stats dict
        """
        total_resp_input  = 0
        total_other_input = total_other_output = num_other_calls = 0

        all_pairs = session.get_turn_pairs()
        pbar = tqdm(all_pairs, desc=f"  Session {session.session_id} Phase1", leave=True, ncols=100)

        for user_turn, assistant_turn in pbar:
            gt_response = assistant_turn.utterance if assistant_turn else ""

            result = agent.process_turn(
                user_utterance=user_turn.utterance,
                gt_response=gt_response,
                conv_id=user_turn.conv_id,
                turn_id=user_turn.turn_id,
                session_id=user_turn.session_id,
            )

            # Token accumulation (estimated input only — no LLM call)
            total_resp_input  += result.token_info.get("input", 0)

            total_other_input  += result.internal_token_info.get("input",  0)
            total_other_output += result.internal_token_info.get("output", 0)
            num_other_calls    += result.internal_token_info.get("calls",  0)

            # Write LTM retrieval log
            log_entry = _build_retrieval_log_entry(
                phase="prompt_construction",
                session_id=user_turn.session_id,
                conv_id=user_turn.conv_id,
                turn_id=user_turn.turn_id,
                retrieval_data=result.retrieval_log_data,
            )
            log_writer.write(log_entry)

        pbar.close()

        token_accum = {
            "total_response_input":  total_resp_input,
            "total_other_input":     total_other_input,
            "total_other_output":    total_other_output,
            "num_other_api_calls":   num_other_calls,
        }
        return len(all_pairs), token_accum

    # -------------------------------------------------------------------------
    # Phase 2 helpers
    # -------------------------------------------------------------------------

    def _process_phase2(
        self,
        session:    Session,
        agent:      LDAgentModule,
        log_writer: RetrievalLogWriter,
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Run QA answering phase (Phase 2).

        Calls flush_to_ltm() first to commit remaining STM to LTM.
        QA exchanges are never stored in memory.

        Returns:
            qa_results:        list of per-QA result dicts
            qa_timing_records: list of timing dicts (all QA questions)
            token_accum:       accumulated token stats dict (QA + flush)
        """
        # Flush remaining STM → LTM before QA; count flush as "other" tokens
        flush_tokens = agent.flush_to_ltm(session_id=session.session_id)
        flush_input  = flush_tokens.get("input",  0)
        flush_output = flush_tokens.get("output", 0)
        flush_calls  = flush_tokens.get("calls",  0)

        qa_results        = []
        qa_timing_records = []
        total_qa_input    = total_qa_output = num_qa_calls = 0

        logger.info(f"QA: {len(session.qa)} questions (subset={self.subset})")

        for qa_idx, qa in enumerate(session.qa):
            logger.info(f"  QA [{qa_idx + 1}/{len(session.qa)}]: {qa.question[:60]}...")

            qa_result = agent.get_qa_answer(
                question=qa.question,
                subset=self.subset,
            )

            total_qa_input  += qa_result.token_info.get("input",  0)
            total_qa_output += qa_result.token_info.get("output", 0)
            num_qa_calls    += qa_result.num_api_calls
            qa_timing_records.append(qa_result.timing)

            # Write LTM retrieval log (QA phase; conv_id=-1, turn_id=-1 as sentinel)
            log_entry = _build_retrieval_log_entry(
                phase="qa",
                session_id=session.session_id,
                conv_id=-1,
                turn_id=qa_idx,
                retrieval_data=qa_result.retrieval_log_data,
            )
            log_writer.write(log_entry)

            # Normalize answer for supportive subset
            generated_answer = qa_result.answer
            if self.subset == "supportive":
                generated_answer = _normalize_yes_no_unknown(generated_answer)

            qa_results.append({
                "question":            qa.question,
                "generated_answer":    generated_answer,
                "ground_truth_answer": qa.answer,
                "retrieved_memories":  qa_result.retrieved_memories,
                "qa_tokens": {
                    "input":  qa_result.token_info.get("input",  0),
                    "output": qa_result.token_info.get("output", 0),
                    "model":  self.model_path,
                },
            })

        token_accum = {
            "total_qa_input":    total_qa_input,
            "total_qa_output":   total_qa_output,
            "num_qa_api_calls":  num_qa_calls,
            "flush_input":       flush_input,
            "flush_output":      flush_output,
            "flush_calls":       flush_calls,
        }
        return qa_results, qa_timing_records, token_accum

    # -------------------------------------------------------------------------
    # Per-session entry point
    # -------------------------------------------------------------------------

    def run_session(self, session: Session) -> Dict:
        """Run Phase 1 → Phase 2 → Phase 3 for one session."""
        print(f"\n{'=' * 60}")
        print(f"Session {session.session_id}  ({len(session.get_turn_pairs())} turn pairs, "
              f"{len(session.qa)} QA)")
        print(f"{'=' * 60}")

        sample_id  = f"ldagent_{cfg.extract_model_name(self.model_path)}_{session.session_id}"
        agent      = LDAgentModule(
            llm_client=self.llm_client,
            config=cfg,
            logger=logger,
            sample_id=sample_id,
        )
        if cfg.ENABLE_LLM_CALL_LOGGING:
            llm_log_dir = self.prompt_log_dir / f"session_{session.session_id}"
            agent.set_llm_logger(LLMCallLogger(llm_log_dir))
        log_writer = RetrievalLogWriter(
            self.retrieval_log_dir / f"session_{session.session_id}_retrieval_log.jsonl"
        )

        # ---- Phase 1 ----
        num_turns, resp_accum = self._process_phase1(
            session, agent, log_writer
        )

        # ---- Phase 2 (QA) ----
        qa_results, qa_timing_records, qa_accum = self._process_phase2(
            session, agent, log_writer
        )

        # ---- Phase 3: snapshot, clear ----
        memory_snapshot_path = None
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snap_dir = self.memory_snapshots_dir / f"session_{session.session_id}"
            agent.save_snapshot(snap_dir)
            memory_snapshot_path = f"memory_snapshots/session_{session.session_id}"

        agent.clear()

        # ---- Compile token statistics ----
        ri = resp_accum["total_response_input"]   # estimated input only (no LLM call)
        qi = qa_accum["total_qa_input"]
        qo = qa_accum["total_qa_output"]
        oi = resp_accum["total_other_input"]  + qa_accum["flush_input"]
        oo = resp_accum["total_other_output"] + qa_accum["flush_output"]
        qc = qa_accum["num_qa_api_calls"]
        oc = resp_accum["num_other_api_calls"]  + qa_accum["flush_calls"]

        token_statistics = {
            "total_response_input":  ri,
            "total_qa_input":        qi,
            "total_qa_output":       qo,
            "total_input":           ri + qi + oi,
            "total_output":          qo + oo,
            "num_qa_api_calls":      qc,
            "num_total_api_calls":   qc + oc,
        }

        # ---- Timing statistics ----
        def _avg_timing(records):
            if not records:
                return {"retrieval_time": 0.0, "inference_time": 0.0, "total_time": 0.0}
            n = len(records)
            return {k: sum(r[k] for r in records) / n for k in records[0]}

        timing_statistics = {
            "phase2_qa_avg": _avg_timing(qa_timing_records),
        }

        print(f"  Phase 1: {num_turns} turns processed")
        print(f"  Phase 2: {len(qa_results)}/{len(session.qa)} QA answered")
        print(f"  Tokens — response_input(est): {ri}, QA: {qi}+{qo}, other: {oi}+{oo}")

        return {
            "session_id":           session.session_id,
            "qa_results":           qa_results,
            "timing_statistics":    timing_statistics,
            "token_statistics":     token_statistics,
            "memory_snapshot_path": memory_snapshot_path,
        }

    # -------------------------------------------------------------------------
    # Full experiment loop
    # -------------------------------------------------------------------------

    def run_experiment(
        self,
        sessions:        List[Session],
        start_session:   int,
        end_session:     int,
        results_file:    Path,
        checkpoint_file: Path,
        start_from_idx:  int = 0,
    ):
        target_sessions = sessions[start_session:end_session + 1]
        total           = len(target_sessions)

        print(f"\n{'#' * 60}")
        print(f"# LD-Agent Experiment")
        print(f"# Session range: [{start_session}, {end_session}]  ({total} sessions)")
        print(f"# Subset: {self.subset}  |  Resuming from index: {start_from_idx}")
        print(f"{'#' * 60}")

        results = load_existing_results(results_file)

        for idx, session in enumerate(target_sessions):
            if idx < start_from_idx:
                continue

            try:
                result = self.run_session(session)
                results.append(result)
                save_results(results_file, results)

                if cfg.ENABLE_CHECKPOINTING:
                    save_checkpoint(
                        checkpoint_file, idx,
                        self.model_path, self.subset,
                        start_session, end_session, self.config_name,
                    )

            except Exception as e:
                logger.error(f"Error processing session {session.session_id}: {e}")
                if cfg.ENABLE_CHECKPOINTING:
                    save_checkpoint(
                        checkpoint_file, idx - 1 if idx > 0 else -1,
                        self.model_path, self.subset,
                        start_session, end_session, self.config_name,
                    )
                raise

        print(f"\n{'#' * 60}")
        print(f"# Experiment completed — {total - start_from_idx} sessions processed")
        print(f"# Results: {results_file}")
        print(f"{'#' * 60}\n")
        return results


# =============================================================================
# LLM CLIENT FACTORY
# =============================================================================

def create_llm_client(model_path: str, tensor_parallel_size: int,
                      gpu_memory_utilization: float, max_model_len: int = None):
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
    elif cfg.LLM_ENGINE == "openai":
        c = cfg.OPENAI_CONFIG
        return _create(engine="openai", model_name=c["model_name"], api_key=c.get("api_key"))
    elif cfg.LLM_ENGINE == "together":
        c = cfg.TOGETHER_CONFIG
        return _create(engine="together", model_name=c["model_name"], api_key=c.get("api_key"))
    else:
        raise ValueError(f"Unknown LLM engine: {cfg.LLM_ENGINE}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run LD-Agent Experiment (ImplexConv)")
    parser.add_argument("--start-session",   type=int, required=True)
    parser.add_argument("--end-session",     type=int, required=True)
    parser.add_argument("--subset",          type=str, required=True,
                        choices=["opposed", "supportive"])
    parser.add_argument("--model",           type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory",      type=float, default=0.5)
    parser.add_argument("--max-model-len",   type=int, default=None,
                        help="vLLM max_model_len (optional, for large models)")
    parser.add_argument("--config",          type=str, default="config",
                        help="Config module name (without .py) in ldagent_new/")
    parser.add_argument("--max-tokens",      type=int, default=None,
                        help="Override MAX_TOKENS from config")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")

    # Dynamic config load
    global cfg
    cfg = importlib.import_module(args.config)

    # Override MAX_TOKENS if provided
    if args.max_tokens is not None:
        cfg.MAX_TOKENS = args.max_tokens

    # Select dataset file based on subset
    if args.subset == "opposed":
        dataset_file = cfg.DATASET_OPPOSED
    else:
        dataset_file = cfg.DATASET_SUPPORTIVE

    config_name = args.config

    cfg.ensure_directories(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    results_file    = cfg.get_results_file(
        args.model, args.subset, args.start_session, args.end_session, config_name)
    checkpoint_file = cfg.get_checkpoint_file(
        args.model, args.subset, args.start_session, args.end_session, config_name)
    log_dir         = cfg.get_session_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name) / "logs"

    # Create nohup directory if not exists
    nohup_dir = _MODULE_DIR / "nohup"
    nohup_dir.mkdir(parents=True, exist_ok=True)

    global logger
    logger = setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("LD-Agent Experiment (ImplexConv)")
    logger.info(f"Model:           {args.model}")
    logger.info(f"Subset:          {args.subset}")
    logger.info(f"Session range:   [{args.start_session}, {args.end_session}]")
    logger.info(f"Config:          {config_name}")
    logger.info(f"MAX_TOKENS:      {cfg.MAX_TOKENS}")
    logger.info(f"Dataset:         {dataset_file}")
    logger.info("=" * 60)

    try:
        sessions = load_implexconv_dataset(str(dataset_file))
        logger.info(f"Loaded {len(sessions)} sessions")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    if args.start_session < 0 or args.end_session >= len(sessions):
        logger.error(
            f"Invalid session range [{args.start_session}, {args.end_session}]; "
            f"dataset has {len(sessions)} sessions (0–{len(sessions) - 1})"
        )
        return 1
    if args.start_session > args.end_session:
        logger.error("start_session must be less than or equal to end_session")
        return 1

    start_from_idx = 0
    if cfg.ENABLE_CHECKPOINTING:
        last = load_checkpoint(checkpoint_file)
        if last is not None:
            start_from_idx = last + 1
            logger.info(f"Resuming from session index {start_from_idx}")

    try:
        llm_client = create_llm_client(
            args.model,
            args.tensor_parallel,
            args.gpu_memory,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        logger.error(f"Failed to initialise LLM client: {e}")
        return 1

    try:
        runner = LDAgentExperimentRunner(
            llm_client=llm_client,
            model_path=args.model,
            subset=args.subset,
            start_session=args.start_session,
            end_session=args.end_session,
            config_name=config_name,
        )
        runner.run_experiment(
            sessions=sessions,
            start_session=args.start_session,
            end_session=args.end_session,
            results_file=results_file,
            checkpoint_file=checkpoint_file,
            start_from_idx=start_from_idx,
        )
        logger.info(f"Experiment complete. Results: {results_file}")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
