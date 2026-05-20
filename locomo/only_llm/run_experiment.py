"""
OnlyLLM Batch Experiment Runner — LoCoMo (QA-Only Variant)

This batch variant keeps the original OnlyLLM context/prompt logic as much as
possible while batching independent QA LLM calls across samples.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))
MODULE_LOG_STEM = _MODULE_DIR.name.replace("_", "")

cfg = None  # type: ignore

import base_runner as base

QAPair = base.QAPair
Sample = base.Sample
LLMCallLogger = base.LLMCallLogger
QA_SCHEMA = base.QA_SCHEMA
OnlyLLMRunner = base.OnlyLLMRunner
build_retrieval_log_entry = base.build_retrieval_log_entry
load_existing_results = base.load_existing_results
save_results = base.save_results
write_retrieval_log = base.write_retrieval_log
load_locomo_dataset = base.load_locomo_dataset
atomic_write_json = base.atomic_write_json
usage_to_token_info = base.usage_to_token_info
token_info_to_usage = base.token_info_to_usage
build_memory_at_qa_start = base.build_memory_at_qa_start

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"{MODULE_LOG_STEM}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path) -> Tuple[Set[str], Optional[int]]:
    if not checkpoint_file.exists():
        return set(), None
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        if "completed_sample_ids" in data:
            return {str(sample_id) for sample_id in data["completed_sample_ids"]}, None
        return set(), data.get("last_completed_sample_index")
    except Exception as exc:
        logger.warning(f"Could not load checkpoint: {exc}")
        return set(), None


def save_checkpoint(
    checkpoint_file: Path,
    completed_sample_ids: Set[str],
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
):
    data = {
        "completed_sample_ids": sorted(completed_sample_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name": config_name,
            "model": model_path,
            "start_sample": start_sample,
            "end_sample": end_sample,
        },
    }
    atomic_write_json(checkpoint_file, data)


# =============================================================================
# BATCHED RUNNER
# =============================================================================

class BatchedOnlyLLMRunner:
    def __init__(
        self,
        llm_client,
        tokenizer,
        model_path: str,
        max_model_len: Optional[int],
        qa_batch_size: int,
        config_metadata: Optional[Dict] = None,
    ):
        self.client = llm_client
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.qa_batch_size = qa_batch_size
        self.config_metadata = config_metadata or {}

    def run_batch(
        self,
        samples: List[Sample],
        retrieval_log_paths: List[Path],
        llm_call_log_dirs: List[Path],
    ) -> List[Dict]:
        runners = [
            OnlyLLMRunner(
                llm_client=self.client,
                tokenizer=self.tokenizer,
                model_path=self.model_path,
                max_model_len=self.max_model_len,
            )
            for _ in samples
        ]

        for runner, llm_call_log_dir in zip(runners, llm_call_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                runner.set_llm_logger(LLMCallLogger(llm_call_log_dir))
            else:
                runner.set_llm_logger(None)

        self._run_phase1_contexts(samples, runners)
        qa_results_list, phase2_stats = self._run_phase2_batched(
            samples,
            runners,
            retrieval_log_paths,
        )

        results = []
        for sample, runner, qa_results, stats in zip(samples, runners, qa_results_list, phase2_stats):
            runner.ctx.clear()
            logger.info(f"Context cleared: sample {sample.sample_id}")

            token_stats = {
                "call_5_qa": {
                    "input": stats["qa_input"],
                    "output": stats["qa_output"],
                    "llm_calls": stats["num_qa_calls"],
                    "parse_fallback_count": stats["qa_parse_fallback_count"],
                },
                "total_input": stats["qa_input"],
                "total_output": stats["qa_output"],
                "total_llm_calls": stats["num_qa_calls"],
            }
            results.append({
                "sample_id": sample.sample_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": build_memory_at_qa_start(),
                "qa_results": qa_results,
                "token_statistics": token_stats,
                "memory_snapshot_path": None,
            })
        return results

    def _run_phase1_contexts(self, samples: List[Sample], runners: List[OnlyLLMRunner]) -> None:
        for sample, runner in tqdm(list(zip(samples, runners)), desc="Phase1 contexts"):
            total_turns = sum(len(session.turns) for session in sample.sessions)
            logger.info(
                f"Sample {sample.sample_id} "
                f"({len(sample.sessions)} sessions, {total_turns} turns, {len(sample.qa)} QA) "
                f"[mode={'all_given' if cfg.HISTORY_ALL_GIVEN else 'window'}]"
            )
            runner.ctx.clear()
            for session in sample.sessions:
                for turn in session.turns:
                    runner.ctx.add_turn(session.session_id, session.date_time, turn)
            logger.info(f"Phase 1 done: sample {sample.sample_id}, {len(runner.ctx)} turns stored in context")

    def _run_phase2_batched(
        self,
        samples: List[Sample],
        runners: List[OnlyLLMRunner],
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        qa_jobs = []
        for sample_idx, (sample, runner) in enumerate(zip(samples, runners)):
            for qa_idx, qa in enumerate(sample.qa):
                choice_seed = f"{sample.sample_id}::{qa_idx}::{qa.question}"
                job_info = runner.build_qa_job(qa, choice_order_seed=choice_seed)
                qa_jobs.append({
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "sample_id": sample.sample_id,
                    "qa": qa,
                    "choice_seed": choice_seed,
                    **job_info,
                })

        qa_results_per_sample = [[] for _ in samples]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0, "qa_parse_fallback_count": 0}
            for _ in samples
        ]

        if not qa_jobs:
            return qa_results_per_sample, phase2_stats

        jobs_by_temperature: Dict[float, List[Dict]] = {}
        for job in qa_jobs:
            jobs_by_temperature.setdefault(job["temperature"], []).append(job)

        all_outputs = []
        for temperature, jobs in jobs_by_temperature.items():
            for chunk_start in tqdm(
                range(0, len(jobs), self.qa_batch_size),
                desc=f"Phase2 QA temp={temperature}",
            ):
                chunk = jobs[chunk_start:chunk_start + self.qa_batch_size]
                chunk_prompts = [job["prompt"] for job in chunk]

                def retry_single(local_idx: int, _prompt: str):
                    job = chunk[local_idx]
                    runner = runners[job["sample_idx"]]
                    qa_gen = runner.answer_qa(
                        job["qa"],
                        choice_order_seed=job["choice_seed"],
                    )
                    return (
                        {"answer": qa_gen["generated_answer"]},
                        token_info_to_usage(qa_gen["qa_tokens"]),
                        True,
                    )

                chunk_results, chunk_usages, chunk_logged = self._batch_generate_with_retry(
                    prompts=chunk_prompts,
                    system_prompt=None,
                    max_tokens=cfg.MAX_TOKENS,
                    temperature=temperature,
                    guided_json=QA_SCHEMA,
                    sequential_retry_fn=retry_single,
                )
                for job, result, usage, already_logged in zip(
                    chunk,
                    chunk_results,
                    chunk_usages,
                    chunk_logged,
                ):
                    all_outputs.append((job, result, usage, already_logged))

        all_outputs.sort(key=lambda item: (item[0]["sample_idx"], item[0]["qa_idx"]))

        for job, result, usage, already_logged in all_outputs:
            sample_idx = job["sample_idx"]
            runner = runners[sample_idx]
            qa = job["qa"]

            if not already_logged and runner.llm_logger is not None:
                runner.llm_logger.log("call_5_qa", "", job["prompt"], result)

            token_info = usage_to_token_info(usage, self.model_path)
            phase2_stats[sample_idx]["qa_input"] += token_info["input"]
            phase2_stats[sample_idx]["qa_output"] += token_info["output"]
            phase2_stats[sample_idx]["num_qa_calls"] += 1
            phase2_stats[sample_idx]["qa_parse_fallback_count"] += int(already_logged)

            write_retrieval_log(
                retrieval_log_paths[sample_idx],
                build_retrieval_log_entry(
                    sample_id=job["sample_id"],
                    query=qa.question,
                    turns_in_prompt=job["turns_in_prompt"],
                    sessions_in_prompt=job["sessions_in_prompt"],
                    token_budget=job["token_budget"],
                ),
            )

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            qa_results_per_sample[sample_idx].append({
                "question": qa.question,
                "category": qa.category,
                "generated_answer": answer,
                "ground_truth_answer": qa.final_answer,
                "evidence": qa.evidence,
                "retrieved_memories": [],
                "qa_tokens": token_info,
            })

        return qa_results_per_sample, phase2_stats

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        guided_json=None,
        sequential_retry_fn=None,
    ) -> Tuple[List, List[Dict], List[bool]]:
        if not prompts:
            return [], [], []

        try:
            texts, usages = self.client.generate_batch_raw(
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
            if sequential_retry_fn is None:
                raise
            logger.warning(f"Batch generation failed; falling back to sequential retry for whole chunk: {exc}")
            results = []
            usages = []
            already_logged = []
            for idx, prompt in enumerate(prompts):
                result, usage, logged = sequential_retry_fn(idx, prompt)
                results.append(result)
                usages.append(usage)
                already_logged.append(logged)
            return results, usages, already_logged

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
            logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
            if sequential_retry_fn is not None:
                result, usage, logged = sequential_retry_fn(idx, prompts[idx])
                parsed[idx] = result
                usages[idx] = usage
                already_logged[idx] = logged
                continue
            parsed[idx] = {}

        return parsed, usages, already_logged


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def create_llm_client(
    model_path: str,
    tensor_parallel: int,
    gpu_memory: float,
    max_model_len: Optional[int],
):
    return base.create_llm_client(
        model_path=model_path,
        tensor_parallel=tensor_parallel,
        gpu_memory=gpu_memory,
        max_model_len=max_model_len,
    )


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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_lb")
    pre_args, _ = pre_parser.parse_known_args()
    config_name = Path(pre_args.config).stem

    config_file = _MODULE_DIR / f"{config_name}.py"
    if not config_file.exists():
        print(f"[error] Config file not found: {config_file}")
        return 1

    import importlib.util

    spec = importlib.util.spec_from_file_location(config_name, config_file)
    global cfg
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    sys.modules["config"] = cfg
    base.cfg = cfg

    parser = argparse.ArgumentParser(
        description="OnlyLLM Batch Experiment on LoCoMo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-sample", type=int, required=True, help="First sample index (inclusive)")
    parser.add_argument("--end-sample", type=int, required=True, help="Last sample index (inclusive)")
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Model max context length. Required when HISTORY_ALL_GIVEN=True "
            "(used for token budget). Optional otherwise."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.BATCH_SIZE,
        help="Number of samples to process together.",
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        default=cfg.QA_BATCH_SIZE,
        help="Number of QA prompts per generate_batch_raw call.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_lb",
        help="Config file name (without .py). Used to load settings and as output directory prefix.",
    )
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_sample > args.end_sample:
        parser.error("--start-sample must be <= --end-sample")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.qa_batch_size < 1:
        parser.error("--qa-batch-size must be >= 1")
    if cfg.HISTORY_ALL_GIVEN and args.max_model_len is None:
        parser.error("--max-model-len is required when HISTORY_ALL_GIVEN=True")

    cfg.ensure_directories(args.model, args.start_sample, args.end_sample, config_name)

    sample_dir = cfg.get_sample_dir(args.model, args.start_sample, args.end_sample, config_name)
    global logger
    logger = setup_logging(sample_dir / "logs")

    results_file = cfg.get_results_file(args.model, args.start_sample, args.end_sample, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.start_sample, args.end_sample, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.start_sample, args.end_sample, config_name)
    prompt_log_dir = cfg.get_prompt_log_dir(args.model, args.start_sample, args.end_sample, config_name)

    mode_label = (
        "all_given (token-budget trim)"
        if cfg.HISTORY_ALL_GIVEN
        else f"window (last {cfg.MAX_CONTEXT_TURNS} turns)"
    )

    logger.info("=" * 60)
    logger.info("OnlyLLM Batch Experiment — LoCoMo")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Samples         : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {args.qa_batch_size}")
    logger.info(f"  History mode    : {mode_label}")
    if cfg.HISTORY_ALL_GIVEN:
        logger.info(f"  Max model len   : {args.max_model_len}")
        logger.info(f"  Output reserve  : {cfg.OUTPUT_TOKEN_RESERVE} tokens")
        logger.info(f"  Safety margin   : {cfg.CONTEXT_SAFETY_MARGIN} tokens")
    else:
        logger.info(f"  Max model len   : {args.max_model_len if args.max_model_len else 'auto'}")
    logger.info(f"  LLM call logging: {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    logger.info("Loading dataset...")
    samples = load_locomo_dataset(cfg.DATASET_PATH)
    if args.end_sample >= len(samples):
        logger.error(f"end_sample={args.end_sample} out of range (dataset has {len(samples)} samples)")
        return 1

    target_samples = samples[args.start_sample: args.end_sample + 1]

    completed_sample_ids: Set[str] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_sample_ids, last_completed_index = load_checkpoint(checkpoint_file)
        if last_completed_index is not None:
            completed_sample_ids.update(
                str(target_samples[i].sample_id)
                for i in range(min(last_completed_index + 1, len(target_samples)))
            )
        if completed_sample_ids:
            logger.info(f"Resuming: {len(completed_sample_ids)} samples already completed")

    pending_samples = [
        sample for sample in target_samples
        if str(sample.sample_id) not in completed_sample_ids
    ]
    if not pending_samples:
        logger.info("All samples already completed.")
        return 0

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model)
    logger.info("Tokenizer ready.")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(args.model, args.tensor_parallel, args.gpu_memory, args.max_model_len)
    logger.info("LLM client ready.")

    runner = BatchedOnlyLLMRunner(
        llm_client=llm_client,
        tokenizer=tokenizer,
        model_path=args.model,
        max_model_len=args.max_model_len,
        qa_batch_size=args.qa_batch_size,
        config_metadata={
            "config_name": config_name,
            "model": args.model,
            "llm_engine": cfg.LLM_ENGINE,
            "temperature": cfg.TEMPERATURE,
            "temperature_c5": cfg.TEMPERATURE_C5,
            "max_tokens": cfg.MAX_TOKENS,
            "sample_range": [args.start_sample, args.end_sample],
            "batch_size": args.batch_size,
            "qa_batch_size": args.qa_batch_size,
            "history_all_given": cfg.HISTORY_ALL_GIVEN,
            "max_context_turns": cfg.MAX_CONTEXT_TURNS,
            "output_token_reserve": cfg.OUTPUT_TOKEN_RESERVE,
            "context_safety_margin": cfg.CONTEXT_SAFETY_MARGIN,
            "max_model_len": args.max_model_len,
        },
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print("# OnlyLLM Batch  |  LoCoMo")
    print(f"# Model   : {cfg.extract_model_name(args.model)}")
    print(f"# Mode    : {mode_label}")
    print(f"# Samples [{args.start_sample}, {args.end_sample}]  ({len(pending_samples)} to process)")
    print(f"# Batch size    : {args.batch_size}")
    print(f"# QA batch size : {args.qa_batch_size}")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_samples) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_samples[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        sample_ids = [sample.sample_id for sample in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: samples {sample_ids}")

        batch_retrieval_log_paths = [
            retrieval_log_dir / f"sample_{sample.sample_id}_retrieval_log.jsonl"
            for sample in batch
        ]
        batch_prompt_log_dirs = [
            prompt_log_dir / f"sample_{sample.sample_id}"
            for sample in batch
        ]

        try:
            batch_results = runner.run_batch(
                samples=batch,
                retrieval_log_paths=batch_retrieval_log_paths,
                llm_call_log_dirs=batch_prompt_log_dirs,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as exc:
            logger.error(f"Batch {batch_idx + 1} failed with hard error: {exc}")
            import traceback

            traceback.print_exc()
            return 1

        for result in batch_results:
            results.append(result)
            save_results(results_file, results)

            if cfg.ENABLE_CHECKPOINTING:
                completed_sample_ids.add(str(result["sample_id"]))
                save_checkpoint(
                    checkpoint_file=checkpoint_file,
                    completed_sample_ids=completed_sample_ids,
                    model_path=args.model,
                    start_sample=args.start_sample,
                    end_sample=args.end_sample,
                    config_name=config_name,
                )

            logger.info(
                f"Sample {result['sample_id']} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total LLM calls: {result['token_statistics']['total_llm_calls']}"
            )

    print(f"\n{'#' * 60}")
    print(f"# Batch experiment complete!  Results -> {results_file}")
    print(f"{'#' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
