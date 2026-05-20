"""
GraphMem v5 Experiment Runner — LoComo batch entry point.

Usage:
    python run_experiment.py \\
        --start-sample 0 --end-sample 9 \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --tensor-parallel 1 --gpu-memory 0.5 \\
        --config config_0
"""

import argparse
import importlib.util
import json
import logging
import os
import re
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

from load_dataset import Sample, load_locomo_dataset

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

def load_checkpoint(checkpoint_file: Path) -> Set[str]:
    if not checkpoint_file.exists():
        return set()
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        return {str(x) for x in data.get("completed_sample_ids", [])}
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def save_checkpoint(
    checkpoint_file: Path,
    completed_sample_ids: Set[str],
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_0",
) -> None:
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
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load existing results ({e}). Starting fresh.")
        return []


def save_results(results_file: Path, results: List[Dict]) -> None:
    _atomic_write(results_file, results)


def _atomic_write(path: Path, data) -> None:
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
# HELPERS
# =============================================================================

_DATE_PATTERN = re.compile(r"(\d{1,2})\s+(\w+),?\s+(\d{4})")
_DATE_FORMATS = ["%d %B %Y", "%d %b %Y"]
_TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})\s*([ap]m)", re.IGNORECASE)


def parse_session_datetime(date_str: str) -> float:
    """Parse LoComo session datetime string (e.g. '1:56 pm on 8 May, 2023')
    to epoch seconds. Returns 0.0 only when no date component is recoverable."""
    if not date_str:
        return 0.0
    s = date_str.strip()

    m = _DATE_PATTERN.search(s)
    if not m:
        logger.debug(f"Could not parse session datetime: {date_str!r}")
        return 0.0
    day, month, year = m.group(1), m.group(2), m.group(3)
    base = None
    for fmt in _DATE_FORMATS:
        try:
            base = datetime.strptime(f"{day} {month} {year}", fmt)
            break
        except ValueError:
            continue
    if base is None:
        return 0.0

    tm = _TIME_PATTERN.search(s)
    if tm:
        hour = int(tm.group(1)) % 12
        minute = int(tm.group(2))
        if tm.group(3).lower() == "pm":
            hour += 12
        base = base.replace(hour=hour, minute=minute)
    return base.timestamp()


def flatten_sample_turns(sample: Sample, minutes_per_turn: int) -> List[Dict]:
    flat_turns: List[Dict] = []
    for session in sample.sessions:
        base_ts = parse_session_datetime(session.date_time)
        for turn_idx, turn in enumerate(session.turns):
            ts = base_ts + turn_idx * minutes_per_turn * 60.0
            flat_turns.append({
                "session_id": session.session_id,
                "turn_idx": turn_idx,
                "timestamp_seconds": ts,
                "speaker": turn.speaker,
                "text": turn.text,
                "dia_id": turn.dia_id,
            })
    return flat_turns


def write_retrieval_log(log_path: Path, entry: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    sample_id: str,
    query: str,
    retrieval_result,
) -> Dict:
    rr = retrieval_result
    return {
        "phase": "qa",
        "sample_id": sample_id,
        "session_id": None,
        "dia_id": None,
        "query": query,
        "memory_type": "graph_retrieval",
        "num_retrieved": len(rr.all_final_nodes),
        "retrieved_items": [
            {
                "node_id": n.node_id,
                "node_type": n.node_type,
                "dia_id": n.dia_id,
                "speaker": n.speaker,
                "content_preview": n.content[:120],
                "score": rr.node_scores.get(n.node_id, 0.0),
            }
            for n in rr.all_final_nodes
        ],
        "retrieval_scores": [rr.node_scores.get(n.node_id, 0.0) for n in rr.all_final_nodes],
        "module_specific": {
            "module": MODULE_SLUG,
            "num_by_slot": {
                "aps": len(rr.active_persona),
                "traits_stable": len(rr.traits_stable),
                "traits_challenged": len(rr.traits_challenged),
                "states_conflict": len(rr.states_conflict),
                "memories_conflict": len(rr.memories_conflict),
                "states_relevant": len(rr.states_relevant),
                "memories_relevant": len(rr.memories_relevant),
            },
            "seed_counts": {
                "context": len(rr.seed_contexts),
                "memory": len(rr.seed_memories),
                "state": len(rr.seed_states),
                "trait": len(rr.seed_traits),
            },
            "pool_size": len(rr.pool_nodes),
        },
    }


# =============================================================================
# BATCH RUNNER
# =============================================================================

class GraphMemBatchRunner:
    """Processes LoComo samples in batches, batching LLM calls across modules."""

    def __init__(
        self,
        llm_client,
        model_path: str,
        shared_embed_model,
        shared_nlp,
        config_metadata: Dict,
    ):
        self.llm_client = llm_client
        self.model_path = model_path
        self.shared_embed_model = shared_embed_model
        self.shared_nlp = shared_nlp
        self.config_metadata = config_metadata

    def run_batch(
        self,
        samples: List[Sample],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
        enable_call_logging_per_sample: List[bool],
    ) -> List[Dict]:
        from graphmem_module import GraphMemModule, LLMCallLogger

        modules = [
            GraphMemModule(
                self.llm_client,
                model_path=self.model_path,
                config=cfg,
                embed_model=self.shared_embed_model,
                nlp=self.shared_nlp,
            )
            for _ in samples
        ]

        for module, prompt_log_dir, enable_log in zip(
            modules, prompt_log_dirs, enable_call_logging_per_sample
        ):
            if cfg.ENABLE_LLM_CALL_LOGGING and enable_log:
                module.set_llm_logger(LLMCallLogger(prompt_log_dir))

        phase1_token_stats, phase1_internal_stats = self._run_phase1_batched(samples, modules)
        memory_stats = [module.get_memory_stats() for module in modules]
        qa_results_list, phase2_stats = self._run_phase2_batched(samples, modules, retrieval_log_paths)

        results = []
        for i, (sample, module) in enumerate(zip(samples, modules)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"sample_{sample.sample_id}"
                module.save_snapshot(str(snapshot_dir))

            module.clear()

            token_stats = {}
            total_input = 0
            total_output = 0
            total_llm_calls = 0
            for call_type, counts in phase1_token_stats[i].items():
                token_stats[call_type] = counts
                total_input += counts["input"]
                total_output += counts["output"]
                total_llm_calls += counts["llm_calls"]
            qa_stat = phase2_stats[i]
            token_stats["call_6_qa"] = {
                "input": qa_stat["qa_input"],
                "output": qa_stat["qa_output"],
                "llm_calls": qa_stat["num_qa_calls"],
            }
            total_input += qa_stat["qa_input"]
            total_output += qa_stat["qa_output"]
            total_llm_calls += qa_stat["num_qa_calls"]
            token_stats["total_input"] = total_input
            token_stats["total_output"] = total_output
            token_stats["total_llm_calls"] = total_llm_calls

            results.append({
                "sample_id": sample.sample_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": memory_stats[i],
                "qa_results": qa_results_list[i],
                "token_statistics": token_stats,
                "phase1_statistics": phase1_internal_stats[i],
                "memory_snapshot_path": (
                    f"memory_snapshots/sample_{sample.sample_id}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            })

        return results

    # ------------------------------------------------------------------
    # Phase 1: replay turns into memory
    # ------------------------------------------------------------------

    def _run_phase1_batched(
        self,
        samples: List[Sample],
        modules,
    ) -> Tuple[List[Dict], List[Dict]]:
        turns_per_sample = [
            flatten_sample_turns(sample, cfg.MINUTES_PER_TURN_IN_SESSION)
            for sample in samples
        ]
        max_turns = max((len(t) for t in turns_per_sample), default=0)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active = [
                (i, modules[i], turns_per_sample[i][turn_idx])
                for i in range(len(samples))
                if turn_idx < len(turns_per_sample[i])
            ]
            if not active:
                continue

            # pre-turn calls (session boundary → memory extraction)
            pending = []
            for i, module, turn in active:
                call = module.prepare_pre_turn_call(
                    session_id=turn["session_id"],
                    turn_idx=turn["turn_idx"],
                    timestamp_seconds=turn["timestamp_seconds"],
                )
                if call is not None:
                    pending.append((i, module, call))
            self._drain_pending_calls(pending)

            # process turn core
            processed = []
            for i, module, turn in active:
                module.process_turn_core(
                    speaker=turn["speaker"],
                    text=turn["text"],
                    session_id=turn["session_id"],
                    turn_idx=turn["turn_idx"],
                    timestamp_seconds=turn["timestamp_seconds"],
                    dia_id=turn["dia_id"],
                )
                processed.append((i, module, turn))

            # post-turn calls (state extraction)
            pending = []
            for i, module, turn in processed:
                call = module.prepare_post_turn_call(session_id=turn["session_id"])
                if call is not None:
                    pending.append((i, module, call))
            self._drain_pending_calls(pending)

            for _, module, _ in processed:
                module.advance_turn()

        # finalize: flush remaining chunk
        pending = []
        for i, (sample, module) in enumerate(zip(samples, modules)):
            last_session_id = sample.sessions[-1].session_id if sample.sessions else 1
            call = module.prepare_finalize_call(session_id=last_session_id)
            if call is not None:
                pending.append((i, module, call))
        self._drain_pending_calls(pending)

        phase1_token_stats = [module.get_and_reset_internal_tokens() for module in modules]
        phase1_internal_stats = [module.get_and_reset_internal_stats() for module in modules]
        return phase1_token_stats, phase1_internal_stats

    def _drain_pending_calls(self, pending_jobs: List[Tuple[int, object, object]]) -> None:
        current_jobs = pending_jobs
        while current_jobs:
            next_jobs = []
            groups: Dict[str, List] = {}
            for item in current_jobs:
                _, _, call = item
                groups.setdefault(call.call_type, []).append(item)

            for call_type, jobs in groups.items():
                sample_call = jobs[0][2]
                prompts = [call.user_prompt for _, _, call in jobs]
                results, usages = self._batch_generate_with_retry(
                    prompts=prompts,
                    system_prompt=sample_call.system_prompt,
                    max_tokens=sample_call.max_tokens,
                    temperature=cfg.TEMPERATURE,
                    guided_json=sample_call.guided_json,
                )

                # Empty-judgment retry for relation-extraction calls.
                # Mirrors updater.py's sequential JUDGMENT_RETRY policy.
                results = list(results)
                usages = list(usages)
                for retry_attempt in range(1, cfg.JUDGMENT_RETRY + 1):
                    retry_indices = []
                    for i, (job, result) in enumerate(zip(jobs, results)):
                        _, _, call = job
                        if call.expected_judgment_count <= 0:
                            continue
                        judgments = result.get("judgments", []) if isinstance(result, dict) else []
                        if isinstance(judgments, list) and len(judgments) == 0:
                            retry_indices.append(i)
                    if not retry_indices:
                        break
                    retry_prompts = [
                        jobs[i][2].user_prompt
                        + f"\nPrevious attempt returned empty judgments; you MUST output exactly {jobs[i][2].expected_judgment_count} judgments."
                        for i in retry_indices
                    ]
                    retry_results, retry_usages = self._batch_generate_with_retry(
                        prompts=retry_prompts,
                        system_prompt=sample_call.system_prompt,
                        max_tokens=sample_call.max_tokens,
                        temperature=cfg.TEMPERATURE,
                        guided_json=sample_call.guided_json,
                    )
                    for k, i in enumerate(retry_indices):
                        results[i] = retry_results[k]
                        prev = usages[i]
                        ru = retry_usages[k]
                        usages[i] = {
                            "prompt_tokens": prev.get("prompt_tokens", 0) + ru.get("prompt_tokens", 0),
                            "completion_tokens": prev.get("completion_tokens", 0) + ru.get("completion_tokens", 0),
                        }

                for (idx, module, call), result, usage in zip(jobs, results, usages):
                    module.log_call(call.log_dir, call.system_prompt, call.user_prompt, result)
                    module.accumulate_internal_usage(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        call_type=call.log_dir,
                    )
                    # Exhaustion fallback: empty judgments → IRRELEVANT for every expected pair.
                    if call.expected_judgment_count > 0:
                        judgments = result.get("judgments", []) if isinstance(result, dict) else []
                        if isinstance(judgments, list) and len(judgments) == 0:
                            module.apply_irrelevant_fallback(call)
                    next_call = module.apply_pending_call(call, result if isinstance(result, dict) else {})
                    if next_call is not None:
                        next_jobs.append((idx, module, next_call))

            current_jobs = next_jobs

    # ------------------------------------------------------------------
    # Phase 2: category-aware QA
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        samples: List[Sample],
        modules,
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        qa_jobs = []

        for sample_idx, (sample, module) in enumerate(zip(samples, modules)):
            for qa_idx, qa in enumerate(sample.qa):
                choice_seed = f"{sample.sample_id}::{qa_idx}::{qa.question}"
                prepared = module.prepare_qa(
                    question=qa.question,
                    category=qa.category,
                    adversarial_answer=qa.adversarial_answer or "",
                    choice_order_seed=choice_seed,
                )
                write_retrieval_log(
                    retrieval_log_paths[sample_idx],
                    build_retrieval_log_entry(
                        sample_id=sample.sample_id,
                        query=qa.question,
                        retrieval_result=prepared.retrieval_result,
                    ),
                )
                qa_jobs.append({
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "sample_id": sample.sample_id,
                    "qa": qa,
                    "prepared": prepared,
                    "choice_seed": choice_seed,
                    "temperature": prepared.temperature,
                })

        qa_results_per_sample = [[] for _ in samples]
        phase2_stats = [{"qa_input": 0, "qa_output": 0, "num_qa_calls": 0} for _ in samples]

        if not qa_jobs:
            return qa_results_per_sample, phase2_stats

        jobs_by_temp: Dict[float, List[Dict]] = {}
        for job in qa_jobs:
            jobs_by_temp.setdefault(job["temperature"], []).append(job)

        all_outputs: List[Tuple[Dict, Dict, Dict]] = []
        for temperature, jobs in jobs_by_temp.items():
            for chunk_start in tqdm(
                range(0, len(jobs), cfg.QA_BATCH_SIZE),
                desc=f"Phase2 QA temp={temperature}",
            ):
                chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
                prompts = [job["prepared"].prompt for job in chunk]
                system_prompt = chunk[0]["prepared"].system_prompt
                schema = chunk[0]["prepared"].schema

                chunk_results, chunk_usages = self._batch_generate_with_retry(
                    prompts=prompts,
                    system_prompt=system_prompt,
                    max_tokens=cfg.MAX_TOKENS,
                    temperature=temperature,
                    guided_json=schema,
                )
                for job, result, usage in zip(chunk, chunk_results, chunk_usages):
                    all_outputs.append((job, result, usage))

        all_outputs.sort(key=lambda x: (x[0]["sample_idx"], x[0]["qa_idx"]))

        for job, result, usage in all_outputs:
            sample_idx = job["sample_idx"]
            prepared = job["prepared"]
            qa = job["qa"]

            modules[sample_idx].log_call("call_6_qa", prepared.system_prompt, prepared.prompt, result)

            phase2_stats[sample_idx]["qa_input"] += usage.get("prompt_tokens", 0)
            phase2_stats[sample_idx]["qa_output"] += usage.get("completion_tokens", 0)
            phase2_stats[sample_idx]["num_qa_calls"] += 1

            if isinstance(result, dict):
                if qa.category == 5:
                    from generator import _c5_choice_order
                    option_a, option_b = _c5_choice_order(qa.adversarial_answer or "", job["choice_seed"])
                    choice = result.get("choice", "").strip().upper()
                    answer = option_a if choice == "A" else (option_b if choice == "B" else result.get("answer", ""))
                else:
                    answer = result.get("answer", "")
            else:
                answer = ""

            qa_results_per_sample[sample_idx].append({
                "question": qa.question,
                "category": qa.category,
                "generated_answer": answer,
                "ground_truth_answer": qa.final_answer,
                "evidence": qa.evidence,
                "retrieved_memories": prepared.retrieved_memories,
                "qa_tokens": {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "model": self.model_path,
                },
            })

        return qa_results_per_sample, phase2_stats

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
        if not prompts:
            return [], []

        try:
            texts, usages = self.llm_client.generate_batch_raw(
                prompts=prompts,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                guided_json=guided_json,
                return_usage=True,
            )
        except Exception:
            texts = []
            usages = []
            for prompt in prompts:
                try:
                    result = self.llm_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        guided_json=guided_json,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_retry=cfg.JSON_RETRY,
                        return_usage=True,
                    )
                    if guided_json is None:
                        texts.append(result.get("content", "") if isinstance(result, dict) else str(result))
                    else:
                        texts.append(result)
                    usage = result.get("_usage", {}) if isinstance(result, dict) else {}
                    usages.append({
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    })
                except Exception as exc:
                    logger.error(f"Sequential fallback generation failed: {exc}")
                    texts.append({} if guided_json is not None else "")
                    usages.append({"prompt_tokens": 0, "completion_tokens": 0})

        if guided_json is None:
            return texts, usages

        parsed = []
        retry_indices = []
        for idx, text in enumerate(texts):
            if isinstance(text, dict):
                parsed.append({k: v for k, v in text.items() if k != "_usage"})
                continue
            try:
                parsed.append(_try_parse_json(text))
            except Exception:
                parsed.append(None)
                retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
            try:
                retry = self.llm_client.generate(
                    prompt=prompts[idx],
                    system_prompt=system_prompt,
                    guided_json=guided_json,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                usage = retry.get("_usage", {}) if isinstance(retry, dict) else {}
                usages[idx] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                parsed[idx] = {k: v for k, v in retry.items() if k != "_usage"} if isinstance(retry, dict) else {}
            except Exception as exc:
                logger.error(f"JSON retry failed: {exc}")
                parsed[idx] = {}

        return parsed, usages


# =============================================================================
# HELPERS
# =============================================================================

def _try_parse_json(text: str) -> dict:
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def create_llm_client(
    model_path: str,
    tensor_parallel: int,
    gpu_memory: float,
    max_model_len: Optional[int] = None,
):
    from llm_client import create_llm_client as _create

    engine = cfg.LLM_ENGINE
    if engine == "vllm":
        kwargs = dict(
            engine="vllm",
            model_path=model_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_memory,
            download_dir=None,
            enable_prefix_caching=True,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        return _create(**kwargs)
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

    config_file = _MODULE_DIR / f"{config_name}.py"
    if not config_file.exists():
        print(f"[error] Config file not found: {config_file}")
        return 1

    spec = importlib.util.spec_from_file_location(config_name, config_file)
    global cfg
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    sys.modules["config"] = cfg

    parser = argparse.ArgumentParser(description="GraphMem v5 Experiment on LoComo")
    parser.add_argument("--start-sample", type=int, required=True)
    parser.add_argument("--end-sample", type=int, required=True)
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

    if args.start_sample > args.end_sample:
        parser.error("--start-sample must be <= --end-sample")

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
    logger.info("GraphMem v5 Experiment — LoComo")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Samples         : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Temperature C5  : {cfg.TEMPERATURE_C5}")
    logger.info(f"  JUDGMENT_RETRY  : {cfg.JUDGMENT_RETRY}")
    logger.info(f"  Log first N     : {cfg.LLM_CALL_LOG_FIRST_N_SAMPLES}")
    logger.info("=" * 60)

    logger.info("Loading dataset...")
    samples = load_locomo_dataset(str(cfg.DATASET_PATH))
    if args.end_sample >= len(samples):
        logger.error(f"end_sample={args.end_sample} out of range (dataset has {len(samples)} samples)")
        return 1

    target_samples = samples[args.start_sample: args.end_sample + 1]

    completed_sample_ids: Set[str] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_sample_ids = load_checkpoint(checkpoint_file)
        if completed_sample_ids:
            logger.info(f"Resuming: {len(completed_sample_ids)} samples already completed")

    pending_with_idx = [
        (i, s) for i, s in enumerate(target_samples)
        if str(s.sample_id) not in completed_sample_ids
    ]
    if not pending_with_idx:
        logger.info("All samples already completed.")
        return 0

    logger.info("Loading shared embedding/spaCy models...")
    from sentence_transformers import SentenceTransformer
    import spacy

    shared_embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    try:
        shared_nlp = spacy.load(cfg.SPACY_MODEL)
    except OSError:
        spacy.cli.download(cfg.SPACY_MODEL)
        shared_nlp = spacy.load(cfg.SPACY_MODEL)

    logger.info("Initializing LLM client...")
    llm_client = create_llm_client(args.model, args.tensor_parallel, args.gpu_memory, max_model_len=args.max_model_len)
    logger.info("LLM client ready.")

    config_metadata = {
        "config_name": config_name,
        "model": args.model,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "temperature": cfg.TEMPERATURE,
        "temperature_c5": cfg.TEMPERATURE_C5,
        "max_tokens": cfg.MAX_TOKENS,
        "sample_range": [args.start_sample, args.end_sample],
        "state_extraction_h": cfg.STATE_EXTRACTION_H,
        "state_max_count": cfg.STATE_MAX_COUNT,
        "state_ref_context_turns": cfg.STATE_REF_CONTEXT_TURNS,
        "trait_extraction_chunks": cfg.TRAIT_EXTRACTION_CHUNKS,
        "trait_max_count": cfg.TRAIT_MAX_COUNT,
        "chunk_size_conv": cfg.CHUNK_SIZE_CONV,
        "k_sf": cfg.K_SF,
        "k_memory_final": cfg.K_MEMORY_FINAL,
        "k_aps": cfg.K_APS,
        "k_t_final": cfg.K_T_FINAL,
        "w_sr": cfg.W_SR,
        "judgment_retry": cfg.JUDGMENT_RETRY,
        "state_new_rel_prev_window": cfg.STATE_NEW_REL_PREV_WINDOW,
        "enable_extra_relation_extraction": cfg.ENABLE_EXTRA_RELATION_EXTRACTION,
        "enable_shift_chain_pruning": cfg.ENABLE_SHIFT_CHAIN_PRUNING,
        "aps_exclude_shift_source": cfg.APS_EXCLUDE_SHIFT_SOURCE,
        "minutes_per_turn_in_session": cfg.MINUTES_PER_TURN_IN_SESSION,
        "batch_size": args.batch_size,
    }

    runner = GraphMemBatchRunner(
        llm_client=llm_client,
        model_path=args.model,
        shared_embed_model=shared_embed_model,
        shared_nlp=shared_nlp,
        config_metadata=config_metadata,
    )

    results = load_existing_results(results_file)
    pending_samples = [s for _, s in pending_with_idx]
    pending_global_indices = [gi for gi, _ in pending_with_idx]

    print(f"\n{'#' * 60}")
    print(f"# GraphMem v5  |  LoComo")
    print(f"# Model   : {cfg.extract_model_name(args.model)}")
    print(f"# Samples [{args.start_sample}, {args.end_sample}]  ({len(pending_samples)} to process)")
    print(f"# Batch size : {args.batch_size}")
    print(f"{'#' * 60}\n")

    num_batches = (len(pending_samples) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        slice_start = batch_idx * args.batch_size
        slice_end = slice_start + args.batch_size
        batch = pending_samples[slice_start:slice_end]
        batch_global_indices = pending_global_indices[slice_start:slice_end]
        sample_ids = [s.sample_id for s in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: samples {sample_ids}")

        batch_retrieval_log_paths = [
            retrieval_log_dir / f"sample_{s.sample_id}_retrieval_log.jsonl"
            for s in batch
        ]
        batch_prompt_log_dirs = [
            prompt_log_dir / f"sample_{s.sample_id}"
            for s in batch
        ]
        batch_enable_logging = [
            gi < cfg.LLM_CALL_LOG_FIRST_N_SAMPLES
            for gi in batch_global_indices
        ]

        try:
            batch_results = runner.run_batch(
                samples=batch,
                retrieval_log_paths=batch_retrieval_log_paths,
                snapshots_dir=snapshots_dir,
                prompt_log_dirs=batch_prompt_log_dirs,
                enable_call_logging_per_sample=batch_enable_logging,
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
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
    print(f"# Experiment complete!  Results -> {results_file}")
    print(f"{'#' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
