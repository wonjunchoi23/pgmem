"""
A-MEM Batch Experiment Runner — LoComo

Usage example:
    python run_experiment.py \\
        --start-sample 0 --end-sample 9 \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --tensor-parallel 1 --gpu-memory 0.5 \\
        --config config_0
"""

import argparse
import json
import logging
import os
import sys
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
MODULE_SLUG = _MODULE_DIR.name

cfg = None  # type: ignore

from load_dataset import LoCoMoSession, QAPair, Sample, Turn, load_locomo_dataset

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"{MODULE_SLUG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            return {str(x) for x in data["completed_sample_ids"]}, None
        return set(), data.get("last_completed_sample_index")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
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
# LLM CALL LOGGING
# =============================================================================

class LLMCallLogger:
    CALL_DIRS = [
        "call_1_note_construction",
        "call_2_evolution",
        "call_3_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "call_type": call_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output if not isinstance(output, dict) else {
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# RETRIEVAL LOGGING
# =============================================================================

def write_retrieval_log(log_path: Path, entry: Dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    phase: str,
    sample_id: str,
    session_id: Optional[int],
    dia_id: Optional[str],
    query: str,
    retrieved_items: List[Dict],
    num_linked: int,
) -> Dict:
    return {
        "phase": phase,
        "sample_id": sample_id,
        "session_id": session_id,
        "dia_id": dia_id,
        "query": query,
        "memory_type": ["direct", "linked"],
        "num_retrieved": [len(retrieved_items), num_linked],
        "retrieved_items": retrieved_items,
        "retrieval_scores": [item["score"] for item in retrieved_items],
        "module_specific": {
            "module": MODULE_SLUG,
            "num_direct_hits": len(retrieved_items),
            "num_linked_neighbors": num_linked,
        },
    }


# =============================================================================
# HELPERS
# =============================================================================

def flatten_sample_turns(sample: Sample) -> List[Dict]:
    flat_turns: List[Dict] = []
    for session in sample.sessions:
        for turn in session.turns:
            flat_turns.append({
                "session_id": session.session_id,
                "date_time": session.date_time,
                "dia_id": turn.dia_id,
                "speaker": turn.speaker,
                "text": turn.text,
                "memory_content": turn.to_memory_content(),
                "timestamp": f"{turn.dia_id}|{session.date_time}",
            })
    return flat_turns


# =============================================================================
# BATCHED RUNNER
# =============================================================================

class BatchedAMEMRunner:
    QA_SYSTEM_PROMPT = (
        "You are a helpful assistant answering a question about a user "
        "based on their conversation history stored in memory. "
        "Respond in JSON format with an 'answer' field."
    )

    def __init__(self, llm_client, model_path: str, shared_embedding_model):
        self.llm_client = llm_client
        self.model_path = model_path
        self.shared_embedding_model = shared_embedding_model

    def run_batch(
        self,
        samples: List[Sample],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
        config_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        from agent import BaseAgent

        agents = [
            BaseAgent(self.llm_client, self.model_path, embedding_model=self.shared_embedding_model)
            for _ in samples
        ]

        for agent, prompt_log_dir in zip(agents, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                agent.set_llm_logger(LLMCallLogger(prompt_log_dir))

        phase1_token_stats = self._run_phase1_batched(samples, agents)

        memory_stats_list = [agent.get_memory_stats() for agent in agents]
        internal_stats_list = [agent.get_and_reset_internal_stats() for agent in agents]

        qa_results_list, phase2_stats = self._run_phase2_batched(samples, agents, retrieval_log_paths)

        results = []
        for i, (sample, agent) in enumerate(zip(samples, agents)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"sample_{sample.sample_id}"
                agent.save_memory_snapshot(snapshot_dir)
                logger.info(f"Memory snapshot saved: sample {sample.sample_id}")

            agent.clear_memory()
            logger.info(f"Memory cleared: sample {sample.sample_id}")

            p1 = phase1_token_stats[i]
            p2 = phase2_stats[i]
            call1 = p1["call_1_note_construction"]
            call2 = p1["call_2_evolution"]
            call3 = {"input": p2["qa_input"], "output": p2["qa_output"], "llm_calls": p2["num_qa_calls"]}
            token_stats = {
                "call_1_note_construction": {
                    **call1,
                    "parse_fallback_count": internal_stats_list[i]["note_parse_fallback_count"],
                },
                "call_2_evolution":         call2,
                "call_3_qa":                call3,
                "total_input":     call1["input"]     + call2["input"]     + call3["input"],
                "total_output":    call1["output"]    + call2["output"]    + call3["output"],
                "total_llm_calls": call1["llm_calls"] + call2["llm_calls"] + call3["llm_calls"],
            }
            evo_stats = {
                "evo_triggered_count": internal_stats_list[i]["evo_triggered_count"],
                "actions_taken":       internal_stats_list[i]["actions_taken"],
            }
            result = {
                "sample_id": sample.sample_id,
                "memory_at_qa_start": memory_stats_list[i],
                "qa_results": qa_results_list[i],
                "token_statistics": token_stats,
                "evolution_statistics": evo_stats,
                "memory_snapshot_path": (
                    f"memory_snapshots/sample_{sample.sample_id}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            }
            if config_metadata is not None:
                result["config_metadata"] = config_metadata
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_batched(self, samples: List[Sample], agents) -> List[Dict]:
        turns_per_sample = [flatten_sample_turns(sample) for sample in samples]
        max_turns = max((len(turns) for turns in turns_per_sample), default=0)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active_turns = []
            for i, sample in enumerate(samples):
                if turn_idx < len(turns_per_sample[i]):
                    active_turns.append((i, sample, agents[i], turns_per_sample[i][turn_idx]))
            self._process_turn_batch(active_turns)

        return [agent.get_and_reset_memory_tokens() for agent in agents]

    def _process_turn_batch(self, active_turns: List[Tuple[int, Sample, object, Dict]]):
        from memory_layer import LLMWrapper, _EVOLUTION_GUIDED_JSON

        if not active_turns:
            return

        analyze_jobs = []
        for sample_idx, sample, agent, turn in active_turns:
            prompt = agent.memory_system.build_analyze_prompt(turn["memory_content"])
            analyze_jobs.append((sample_idx, sample, agent, turn, prompt))

        analyze_prompts = [job[4] for job in analyze_jobs]
        analyze_texts, analyze_usages = self._batch_generate_with_retry(
            prompts=analyze_prompts,
            system_prompt=LLMWrapper.PLAIN_TEXT_SYSTEM,
            max_tokens=1000,
            temperature=cfg.TEMPERATURE,
            guided_json=None,
        )

        for (sample_idx, sample, agent, turn, prompt), text, usage in zip(analyze_jobs, analyze_texts, analyze_usages):
            if agent.memory_system.llm_logger is not None:
                agent.memory_system.llm_logger.log("call_1_note_construction", "", prompt, text)
            agent.accumulate_memory_tokens(
                usage["prompt_tokens"], usage["completion_tokens"], 1,
                call_type="call_1_note_construction",
            )

        evolve_jobs = []
        for (sample_idx, sample, agent, turn, prompt), text in zip(analyze_jobs, analyze_texts):
            note, evolve_prompt, evolve_ctx = agent.memory_system.apply_analyze_result(
                turn["memory_content"],
                text,
                time=turn["timestamp"],
            )
            if evolve_prompt is None:
                agent.memory_system.store_note(note)
            else:
                evolve_jobs.append((sample_idx, sample, agent, turn, evolve_prompt, note, evolve_ctx))

        if not evolve_jobs:
            return

        evolve_prompts = [job[4] for job in evolve_jobs]
        evolve_results, evolve_usages = self._batch_generate_with_retry(
            prompts=evolve_prompts,
            system_prompt=LLMWrapper.JSON_SYSTEM,
            max_tokens=1000,
            temperature=cfg.TEMPERATURE,
            guided_json=_EVOLUTION_GUIDED_JSON,
        )

        for (sample_idx, sample, agent, turn, prompt, note, evolve_ctx), result, usage in zip(
            evolve_jobs, evolve_results, evolve_usages
        ):
            if agent.memory_system.llm_logger is not None:
                agent.memory_system.llm_logger.log("call_2_evolution", "", prompt, result)
            agent.accumulate_memory_tokens(
                usage["prompt_tokens"], usage["completion_tokens"], 1,
                call_type="call_2_evolution",
            )
            if isinstance(result, dict):
                agent.memory_system.apply_evolve_result(note, result, evolve_ctx)
            else:
                agent.memory_system.store_note(note)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        samples: List[Sample],
        agents,
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        from agent import QA_SCHEMA

        qa_jobs = []
        for sample_idx, (sample, agent) in enumerate(zip(samples, agents)):
            for qa_idx, qa in enumerate(sample.qa):
                retrieved_memory, retrieved_metadata = agent.retrieve_memory_with_metadata(qa.question)

                log_items, num_linked = agent.retrieve_for_log(qa.question)
                write_retrieval_log(
                    retrieval_log_paths[sample_idx],
                    build_retrieval_log_entry(
                        phase="qa",
                        sample_id=sample.sample_id,
                        session_id=None,
                        dia_id=None,
                        query=qa.question,
                        retrieved_items=log_items,
                        num_linked=num_linked,
                    ),
                )

                choice_seed = f"{sample.sample_id}::{qa_idx}::{qa.question}"
                prompt, temperature = agent.build_qa_prompt(
                    question=qa.question,
                    retrieved_memory=retrieved_memory,
                    category=qa.category,
                    adversarial_answer=qa.adversarial_answer or "",
                    choice_order_seed=choice_seed,
                )
                qa_jobs.append({
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "sample_id": sample.sample_id,
                    "qa": qa,
                    "retrieved_memory": retrieved_memory,
                    "retrieved_metadata": retrieved_metadata,
                    "prompt": prompt,
                    "temperature": temperature,
                    "choice_seed": choice_seed,
                })

        qa_results_per_sample = [[] for _ in samples]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0}
            for _ in samples
        ]

        if not qa_jobs:
            return qa_results_per_sample, list(phase2_stats)

        jobs_by_temperature: Dict[float, List[Dict]] = {}
        for job in qa_jobs:
            jobs_by_temperature.setdefault(job["temperature"], []).append(job)

        all_outputs = []
        for temperature, jobs in jobs_by_temperature.items():
            for chunk_start in tqdm(range(0, len(jobs), cfg.QA_BATCH_SIZE), desc=f"Phase2 QA temp={temperature}"):
                chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
                chunk_prompts = [job["prompt"] for job in chunk]

                chunk_results, chunk_usages = self._batch_generate_with_retry(
                    prompts=chunk_prompts,
                    system_prompt=self.QA_SYSTEM_PROMPT,
                    max_tokens=cfg.MAX_TOKENS,
                    temperature=temperature,
                    guided_json=QA_SCHEMA,
                )

                for job, result, usage in zip(chunk, chunk_results, chunk_usages):
                    all_outputs.append((job, result, usage))

        all_outputs.sort(key=lambda item: (item[0]["sample_idx"], item[0]["qa_idx"]))

        for job, result, usage in all_outputs:
            sample_idx = job["sample_idx"]
            if agents[sample_idx]._llm_logger is not None:
                agents[sample_idx]._llm_logger.log("call_3_qa", "", job["prompt"], result)

            phase2_stats[sample_idx]["qa_input"] += usage["prompt_tokens"]
            phase2_stats[sample_idx]["qa_output"] += usage["completion_tokens"]
            phase2_stats[sample_idx]["num_qa_calls"] += 1

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            qa_results_per_sample[sample_idx].append({
                "question": job["qa"].question,
                "category": job["qa"].category,
                "generated_answer": answer,
                "ground_truth_answer": job["qa"].final_answer,
                "evidence": job["qa"].evidence,
                "retrieved_memories": job["retrieved_metadata"],
                "qa_tokens": {
                    "input": usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "model": self.model_path,
                },
            })

        return qa_results_per_sample, list(phase2_stats)

    # ------------------------------------------------------------------
    # Batch LLM helper
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        guided_json=None,
    ) -> Tuple[List, List[Dict]]:
        """Batch generate. Returns (results, usages).

        For plain text (guided_json=None): results is a list of strings.
        For JSON (guided_json set): results is a list of dicts.
        Items that fail JSON parsing are retried sequentially via generate().

        vLLM hard errors (OOM, internal crash) propagate immediately.
        """
        if not prompts:
            return [], []

        texts, usages = self.llm_client.generate_batch_raw(
            prompts=prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            guided_json=guided_json,
            return_usage=True,
        )

        if guided_json is None:
            return texts, usages

        from llm_client import _parse_json_response

        parsed = []
        retry_indices = []
        for idx, text in enumerate(texts):
            try:
                parsed.append(_parse_json_response(text) if isinstance(text, str) else text)
            except (json.JSONDecodeError, ValueError):
                parsed.append(None)
                retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
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
                        "completion_tokens": usage_info.get("completion_tokens", 0),
                    }
                    parsed[idx] = retry_result
                else:
                    parsed[idx] = {}
            except Exception as e:
                logger.error(f"Sequential retry for batch item {idx} failed: {e}")
                parsed[idx] = {}

        return parsed, usages


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

    from agent import BaseAgent  # noqa: F401

    parser = argparse.ArgumentParser(
        description="A-MEM Batch Experiment on LoComo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-sample", type=int, required=True)
    parser.add_argument("--end-sample", type=int, required=True)
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_sample > args.end_sample:
        parser.error("--start-sample must be <= --end-sample")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    config_metadata = {
        "config_name":        config_name,
        "model":              args.model,
        "embedding_model":    cfg.EMBEDDING_MODEL,
        "temperature":        cfg.TEMPERATURE,
        "temperature_c5":     cfg.TEMPERATURE_C5,
        "max_tokens":         cfg.MAX_TOKENS,
        "sample_range":       [args.start_sample, args.end_sample],
        "retrieve_k":         cfg.RETRIEVE_K,
        "evolution_threshold": cfg.EVOLUTION_THRESHOLD,
    }

    cfg.ensure_directories(args.model, args.start_sample, args.end_sample, config_name)

    sample_dir = cfg.get_sample_dir(args.model, args.start_sample, args.end_sample, config_name)
    global logger
    logger = setup_logging(sample_dir / "logs")

    results_file = cfg.get_results_file(args.model, args.start_sample, args.end_sample, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.start_sample, args.end_sample, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.start_sample, args.end_sample, config_name)
    snapshots_dir = cfg.get_memory_snapshots_dir(args.model, args.start_sample, args.end_sample, config_name)
    prompt_log_dir = cfg.get_prompt_log_dir(args.model, args.start_sample, args.end_sample, config_name)

    logger.info("=" * 60)
    logger.info("A-MEM Batch Experiment — LoComo")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Samples         : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
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

    pending_samples = [sample for sample in target_samples if str(sample.sample_id) not in completed_sample_ids]
    if not pending_samples:
        logger.info("All samples already completed.")
        return 0

    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer

    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(args.model, args.tensor_parallel, args.gpu_memory, max_model_len=args.max_model_len)
    logger.info("LLM client ready.")

    runner = BatchedAMEMRunner(
        llm_client=llm_client,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# A-MEM Batch  |  LoComo")
    print(f"# Model   : {cfg.extract_model_name(args.model)}")
    print(f"# Samples [{args.start_sample}, {args.end_sample}]  ({len(pending_samples)} to process)")
    print(f"# Batch size : {args.batch_size}")
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
                snapshots_dir=snapshots_dir,
                prompt_log_dirs=batch_prompt_log_dirs,
                config_metadata=config_metadata,
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed with hard error: {e}")
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
