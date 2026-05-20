"""
OnlyLLM Experiment Runner — PrefEval (single-chain, cumulative-checkpoint).

Flow per invocation
-------------------
Args: --end-session K, --model, --tensor-parallel, --gpu-memory, --max-model-len, --config

1. Resolve output dir; scan memory_snapshots/m_*/ for max k_existing.
2. If K <= k_existing → no-op, exit.
3. Init LLM client + tokenizer; create OnlyLLMRunner.
   If k_existing >= 0: load_snapshot(m_{k_existing}); start_k = k_existing+1.
   Else:               start_k = 0.
4. Load PrefEval samples [0..K] as a chain of Sessions.
5. For k = start_k .. K:
     a. Ingest session k turn-by-turn (just append to DialogueContext, no LLM call).
     b. Run QA at checkpoint k: batch all q_0..q_k together (chunked by QA_BATCH_SIZE).
     c. Append rows to results.jsonl + retrieval_log.jsonl.
     d. Save snapshot m_k (after QA — so snapshot existence ⇔ QA done).
     e. Update cumulative stats.json + meta.json (first run only).
"""

import os
import re
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore

import base_runner as base

QAPair = base.QAPair
Session = base.Session
Turn = base.Turn
LLMCallLogger = base.LLMCallLogger
QA_PROMPT = base.QA_PROMPT
QA_SCHEMA = base.QA_SCHEMA
OnlyLLMRunner = base.OnlyLLMRunner
build_retrieval_log_entry = base.build_retrieval_log_entry
write_retrieval_log = base.write_retrieval_log
append_jsonl = base.append_jsonl
atomic_write_json = base.atomic_write_json
usage_to_token_info = base.usage_to_token_info
token_info_to_usage = base.token_info_to_usage

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"onlyllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


# =============================================================================
# RESUME DETECTION
# =============================================================================

_M_K_RE = re.compile(r"^m_(\d+)$")


def find_max_existing_k(snapshots_dir: Path) -> int:
    """Return the largest k for which snapshots_dir/m_k/context.json exists. -1 if none."""
    if not snapshots_dir.exists():
        return -1
    max_k = -1
    for p in snapshots_dir.iterdir():
        if not p.is_dir():
            continue
        m = _M_K_RE.match(p.name)
        if m and (p / "context.json").exists():
            k = int(m.group(1))
            max_k = max(max_k, k)
    return max_k


# =============================================================================
# CUMULATIVE STATS
# =============================================================================

def _empty_stats() -> Dict:
    return {
        "call_4_qa": {
            "input": 0,
            "output": 0,
            "llm_calls": 0,
            "parse_fallback_count": 0,
        },
        "checkpoints_completed": [],
    }


def load_stats(stats_file: Path) -> Dict:
    if not stats_file.exists():
        return _empty_stats()
    try:
        with open(stats_file) as f:
            data = json.load(f)
        empty = _empty_stats()
        for k, v in empty.items():
            if k not in data:
                data[k] = v
        return data
    except Exception as e:
        logger.warning(f"Could not load stats.json ({e}); starting fresh")
        return _empty_stats()


def merge_stats(stats: Dict, qa_totals: Dict, parse_fallback_delta: int, k: int) -> None:
    stats["call_4_qa"]["input"] += qa_totals["input"]
    stats["call_4_qa"]["output"] += qa_totals["output"]
    stats["call_4_qa"]["llm_calls"] += qa_totals["llm_calls"]
    stats["call_4_qa"]["parse_fallback_count"] += parse_fallback_delta

    if k not in stats["checkpoints_completed"]:
        stats["checkpoints_completed"].append(k)
        stats["checkpoints_completed"].sort()


# =============================================================================
# CHAIN RUNNER
# =============================================================================

class ChainRunner:
    def __init__(self, runner: OnlyLLMRunner, llm_client, model_path: str):
        self.runner = runner
        self.llm_client = llm_client
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Phase 1 — append session turns into DialogueContext
    # ------------------------------------------------------------------

    def ingest_session(self, session: Session) -> None:
        for turn in tqdm(session.turns, desc=f"Ingest k={session.conv_id}", leave=False):
            self.runner.ctx.add_turn(turn)

    # ------------------------------------------------------------------
    # Phase 2 — batched QA at checkpoint k
    # ------------------------------------------------------------------

    def run_checkpoint_qa(
        self,
        sessions_so_far: List[Session],
        k: int,
        retrieval_log_path: Path,
    ) -> Tuple[List[Dict], Dict, int]:
        """At checkpoint k, evaluate q_0..q_k against the current context.

        Returns (rows_for_results_jsonl, qa_token_totals, parse_fallback_delta).
        """
        jobs = []
        for j, sess in enumerate(sessions_so_far):
            qa = sess.qa[0]
            job_info = self.runner.build_qa_job(qa)
            jobs.append({
                "j": j,
                "qa": qa,
                "prompt": job_info["prompt"],
                "turns_in_prompt": job_info["turns_in_prompt"],
                "sessions_in_prompt": job_info["sessions_in_prompt"],
                "token_budget": job_info["token_budget"],
            })

        rows: List[Dict] = []
        qa_totals = {"input": 0, "output": 0, "llm_calls": 0}
        parse_fallback_delta = 0

        if not jobs:
            return rows, qa_totals, parse_fallback_delta

        fallback_count_before = self.runner._qa_parse_fallback_count

        for chunk_start in range(0, len(jobs), cfg.QA_BATCH_SIZE):
            chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            chunk_prompts = [c["prompt"] for c in chunk]

            results, usages, logged_flags = self._batch_generate_with_retry(
                prompts=chunk_prompts,
                system_prompt=self.runner.QA_SYSTEM_PROMPT,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=QA_SCHEMA,
                sequential_retry_qas=[c["qa"] for c in chunk],
            )

            for job, result, usage, already_logged in zip(chunk, results, usages, logged_flags):
                qa = job["qa"]

                if not already_logged and self.runner.llm_logger is not None:
                    self.runner.llm_logger.log("call_4_qa", "", job["prompt"], result)

                in_toks = usage.get("prompt_tokens", 0)
                out_toks = usage.get("completion_tokens", 0)
                qa_totals["input"] += in_toks
                qa_totals["output"] += out_toks
                qa_totals["llm_calls"] += 1

                write_retrieval_log(
                    retrieval_log_path,
                    build_retrieval_log_entry(
                        k=k,
                        question_session=job["j"],
                        query=qa.question,
                        turns_in_prompt=job["turns_in_prompt"],
                        sessions_in_prompt=job["sessions_in_prompt"],
                        token_budget=job["token_budget"],
                    ),
                )

                answer = result.get("answer", "") if isinstance(result, dict) else ""
                rows.append({
                    "k": k,
                    "question_session": job["j"],
                    "question": qa.question,
                    "model_answer": answer,
                    "topic": qa.topic,
                    "persona": qa.persona,
                    "preference": qa.preference,
                    "explanation": qa.explanation,
                    "retrieved_memories": [],
                    "qa_tokens": {
                        "input": in_toks,
                        "output": out_toks,
                        "model": self.model_path,
                    },
                })

        parse_fallback_delta = self.runner._qa_parse_fallback_count - fallback_count_before
        return rows, qa_totals, parse_fallback_delta

    # ------------------------------------------------------------------
    # Batched generate with per-item JSON retry fallback
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        guided_json,
        sequential_retry_qas: List[QAPair],
    ) -> Tuple[List, List[Dict], List[bool]]:
        if not prompts:
            return [], [], []

        try:
            texts, usages = self.llm_client.generate_batch_raw(
                prompts=prompts,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                guided_json=guided_json,
                return_usage=True,
            )
            if len(texts) != len(prompts) or len(usages) != len(prompts):
                raise RuntimeError("Batch generation returned a mismatched number of outputs")
        except Exception as exc:
            logger.warning(f"Batch generation failed; falling back to sequential retry for whole chunk: {exc}")
            results, usages, logged = [], [], []
            for qa in sequential_retry_qas:
                r, u, l = self.runner.answer_qa_sequential(qa)
                results.append(r)
                usages.append(u)
                logged.append(l)
            return results, usages, logged

        already_logged = [False] * len(prompts)
        if guided_json is None:
            return texts, usages, already_logged

        from llm_client import _parse_json_response

        parsed = []
        retry_indices = []
        for idx, text in enumerate(texts):
            try:
                parsed.append(_parse_json_response(text) if isinstance(text, str) else text)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed.append(base._parse_json_robust(text) if isinstance(text, str) else text)
                except (json.JSONDecodeError, ValueError):
                    parsed.append(None)
                    retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(f"Batch item {idx} failed JSON parse — retrying sequentially")
            r, u, l = self.runner.answer_qa_sequential(sequential_retry_qas[idx])
            parsed[idx] = r
            usages[idx] = u
            already_logged[idx] = l

        return parsed, usages, already_logged


# =============================================================================
# TOKENIZER LOADER
# =============================================================================

def load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
        )
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise
        logger.info("AutoTokenizer failed (TokenizersBackend), falling back to mistral_common...")
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        class _MistralTokenizerWrapper:
            def __init__(self, mt):
                self._tok = mt.instruct_tokenizer.tokenizer

            def encode(self, text, add_special_tokens=False):
                return self._tok.encode(text, bos=False, eos=False)

        return _MistralTokenizerWrapper(
            MistralTokenizer.from_hf_hub(model_path, token=os.environ.get("HF_TOKEN"))
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ---- Step 1: load config ----
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_ub")
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
    base.cfg = cfg

    # ---- Step 2: parse args ----
    parser = argparse.ArgumentParser(
        description="OnlyLLM PrefEval Chain Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--end-session", type=int, required=True,
                        help="Last sample index in chain (chain = samples[0..K])")
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Model max context length. Required when HISTORY_ALL_GIVEN=True "
            "(used for token budget). Optional otherwise."
        ),
    )
    parser.add_argument("--qa-batch-size", type=int, default=cfg.QA_BATCH_SIZE,
                        help="Number of QA prompts per generate_batch_raw call.")
    parser.add_argument("--config", type=str, default="config_ub")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.end_session < 0:
        parser.error("--end-session must be >= 0")
    if args.qa_batch_size < 1:
        parser.error("--qa-batch-size must be >= 1")
    if cfg.HISTORY_ALL_GIVEN and args.max_model_len is None:
        parser.error("--max-model-len is required when HISTORY_ALL_GIVEN=True")

    # Apply qa_batch_size override at runtime
    cfg.QA_BATCH_SIZE = args.qa_batch_size

    cfg.ensure_directories(args.model, config_name)

    snapshots_dir  = cfg.get_memory_snapshots_dir(args.model, config_name)
    results_file   = cfg.get_results_file(args.model, config_name)
    retrieval_log  = cfg.get_retrieval_log_file(args.model, config_name)
    stats_file     = cfg.get_stats_file(args.model, config_name)
    meta_file      = cfg.get_meta_file(args.model, config_name)
    prompt_log_dir = cfg.get_prompt_log_dir(args.model, config_name)
    run_log_dir    = cfg.get_run_log_dir(args.model, config_name)

    global logger
    logger = setup_logging(run_log_dir)

    # ---- Step 3: resume detection ----
    k_existing = find_max_existing_k(snapshots_dir)
    K = args.end_session

    mode_label = (
        "all_given (token-budget trim)"
        if cfg.HISTORY_ALL_GIVEN
        else f"window (last {cfg.MAX_CONTEXT_TURNS} turns)"
    )

    logger.info("=" * 60)
    logger.info("OnlyLLM PrefEval Chain")
    logger.info(f"  Config       : {config_name}")
    logger.info(f"  Model        : {args.model}")
    logger.info(f"  end_session  : {K}")
    logger.info(f"  k_existing   : {k_existing}")
    logger.info(f"  History mode : {mode_label}")
    if cfg.HISTORY_ALL_GIVEN:
        logger.info(f"  Max model len: {args.max_model_len}")
        logger.info(f"  Output rsv   : {cfg.OUTPUT_TOKEN_RESERVE} tokens")
        logger.info(f"  Safety margn : {cfg.CONTEXT_SAFETY_MARGIN} tokens")
    logger.info(f"  QA batch size: {args.qa_batch_size}")
    logger.info(f"  Output dir   : {cfg.get_output_dir(args.model, config_name)}")
    logger.info("=" * 60)

    if K <= k_existing:
        logger.info(f"Already complete up to k={k_existing} (>= K={K}). Nothing to do.")
        return 0

    start_k = k_existing + 1

    # ---- Step 4: load chain ----
    from load_dataset import load_prefeval_chain
    sessions = load_prefeval_chain(cfg.DATASET_PATH, end_session=K)

    # ---- Step 5: init LLM client + tokenizer + runner ----
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model)
    logger.info("Tokenizer ready.")

    logger.info("Initialising LLM client...")
    llm_client = base.create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    runner = OnlyLLMRunner(
        llm_client=llm_client,
        tokenizer=tokenizer,
        model_path=args.model,
        max_model_len=args.max_model_len,
    )
    if cfg.ENABLE_LLM_CALL_LOGGING:
        runner.set_llm_logger(LLMCallLogger(prompt_log_dir))

    # Resume: load last snapshot
    if k_existing >= 0:
        snap_dir = snapshots_dir / f"m_{k_existing}"
        logger.info(f"Loading snapshot {snap_dir}")
        runner.load_snapshot(snap_dir)
        logger.info(f"Resumed with {len(runner.ctx)} turns in context")

    # ---- Step 6: meta.json on first run ----
    if not meta_file.exists():
        config_metadata = {
            "config_name":           config_name,
            "model":                 args.model,
            "history_all_given":     cfg.HISTORY_ALL_GIVEN,
            "max_context_turns":     cfg.MAX_CONTEXT_TURNS,
            "output_token_reserve":  cfg.OUTPUT_TOKEN_RESERVE,
            "context_safety_margin": cfg.CONTEXT_SAFETY_MARGIN,
            "max_model_len":         args.max_model_len,
            "temperature":           cfg.TEMPERATURE,
            "max_tokens":            cfg.MAX_TOKENS,
            "conv_ids_per_day":      cfg.CONV_IDS_PER_DAY,
            "minutes_per_turn":      cfg.MINUTES_PER_TURN,
            "qa_batch_size":         args.qa_batch_size,
            "first_run_at":          datetime.now().isoformat(),
        }
        atomic_write_json(meta_file, config_metadata)

    # ---- Step 7: stats.json (cumulative) ----
    stats = load_stats(stats_file)

    # ---- Step 8: chain loop ----
    chain = ChainRunner(runner, llm_client, args.model)

    print(f"\n{'#'*60}")
    print(f"# OnlyLLM PrefEval Chain  |  model={cfg.extract_model_name(args.model)}")
    print(f"# Mode  : {mode_label}")
    print(f"# Sessions {start_k}..{K}  ({K - start_k + 1} to ingest)")
    print(f"{'#'*60}\n")

    for k in tqdm(range(start_k, K + 1), desc="Chain checkpoints"):
        session = sessions[k]
        logger.info(f"--- checkpoint k={k}: ingesting session (conv_id={session.conv_id}, "
                    f"{session.total_turns} turns) ---")

        # (a) Ingest session k
        chain.ingest_session(session)

        # (b) QA at checkpoint k (q_0..q_k)
        logger.info(f"k={k}: running QA over {k + 1} questions")
        rows, qa_totals, parse_fallback_delta = chain.run_checkpoint_qa(
            sessions_so_far=sessions[:k + 1],
            k=k,
            retrieval_log_path=retrieval_log,
        )

        # (c) Append rows
        for row in rows:
            append_jsonl(results_file, row)

        # (d) Save snapshot AFTER QA succeeds (snapshot existence ⇔ QA done)
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snap_dir = snapshots_dir / f"m_{k}"
            runner.save_snapshot(snap_dir, k=k)
            logger.info(f"Saved snapshot {snap_dir}")

        # (e) Update cumulative stats
        merge_stats(stats, qa_totals, parse_fallback_delta, k)
        atomic_write_json(stats_file, stats)

        logger.info(f"k={k} done. ctx_turns={len(runner.ctx)}, "
                    f"qa_in={qa_totals['input']}, qa_out={qa_totals['output']}, "
                    f"qa_calls={qa_totals['llm_calls']}")

    print(f"\n{'#'*60}")
    print(f"# Chain complete!")
    print(f"# Results -> {results_file}")
    print(f"# Snapshots -> {snapshots_dir}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
