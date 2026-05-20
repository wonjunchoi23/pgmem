"""
Dense Retrieval Experiment Runner — LoCoMo (Batched, QA-Only Variant)
"""

import argparse
import hashlib
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
MODULE_SLUG = _MODULE_DIR.name
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore

from load_dataset import QAPair, Sample, load_locomo_dataset


logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"{MODULE_SLUG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


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
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results from {results_file}")
        return results
    except Exception as exc:
        logger.warning(f"Failed to load existing results ({exc}). Starting fresh.")
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


def write_retrieval_log(log_path: Path, entry: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    phase: str,
    sample_id: str,
    session_id: Optional[int],
    dia_id: Optional[str],
    query: str,
    retrieved_pairs: List[Tuple],
    store_size_at_retrieval: int,
) -> Dict:
    retrieved_items = [
        {
            "content_preview": memory.content[:120],
            "score": score,
            "source_turn": {
                "sample_id": memory.sample_id,
                "session_id": memory.session_id,
                "dia_id": memory.dia_id,
            },
        }
        for memory, score in retrieved_pairs
    ]
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "sample_id": sample_id,
        "session_id": session_id,
        "dia_id": dia_id,
        "query": query[:500],
        "memory_type": "dense_vector",
        "num_retrieved": len(retrieved_pairs),
        "retrieved_items": retrieved_items,
        "retrieval_scores": [score for _, score in retrieved_pairs],
        "module_specific": {
            "module": MODULE_SLUG,
            "store_size_at_retrieval": store_size_at_retrieval,
        },
    }


class LLMCallLogger:
    CALL_DIRS = ["call_1_qa"]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for call_dir in self.CALL_DIRS:
            (self._base / call_dir).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output if not isinstance(output, dict) else {
                key: value for key, value in output.items() if key != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _escape_control_chars_in_strings(text: str) -> str:
    result = []
    in_string = False
    prev_backslash = False
    for ch in text:
        if prev_backslash:
            result.append(ch)
            prev_backslash = False
        elif ch == "\\" and in_string:
            result.append(ch)
            prev_backslash = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _parse_json_robust(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    if not text:
        raise json.JSONDecodeError("Empty response from model", "", 0)

    original_error = None

    def _try_all(raw_text: str):
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_escape_control_chars_in_strings(raw_text))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(raw_text))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(_escape_control_chars_in_strings(raw_text)))
        except json.JSONDecodeError:
            pass
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        original_error = exc

    result = _try_all(text)
    if result is not None:
        return result

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        result = _try_all(match.group())
        if result is not None:
            return result

    raise original_error


QA_COMMON_ANSWERING_INSTRUCTION = (
    "Answer only the question, and do not provide any explanation or extra details "
    "beyond the short answer."
)

_QA_PROMPT_SHORT_PHRASE_TEMPLATE = (
    "Based on the context: {context}, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context whenever possible.\n"
    f"{QA_COMMON_ANSWERING_INSTRUCTION}\n\n"
    "Question: {question} Short answer:"
)

QA_PROMPT_MULTI_HOP = _QA_PROMPT_SHORT_PHRASE_TEMPLATE

QA_PROMPT_TEMPORAL = (
    "Based on the context: {context}, answer the following question. "
    "Use DATE of CONVERSATION to answer with an approximate date.\n"
    "Please generate the shortest possible answer, using words from the conversation "
    "where possible, and avoid using any subjects.\n"
    f"{QA_COMMON_ANSWERING_INSTRUCTION}\n\n"
    "Question: {question} Short answer:"
)

QA_PROMPT_OPEN_DOMAIN = _QA_PROMPT_SHORT_PHRASE_TEMPLATE

QA_PROMPT_SINGLE_HOP = _QA_PROMPT_SHORT_PHRASE_TEMPLATE

QA_PROMPT_ADVERSARIAL = (
    "Based on the context: {context}, answer the following question. {question}\n\n"
    "Select the correct answer: {choice_a} or {choice_b}\n"
    "If the context does not contain evidence for either option, prefer the "
    "\"Not mentioned in the conversation\" option.\n"
    f"{QA_COMMON_ANSWERING_INSTRUCTION}\n\n"
    "Short answer:"
)

QA_PROMPTS_BY_CATEGORY = {
    1: QA_PROMPT_SINGLE_HOP,
    2: QA_PROMPT_MULTI_HOP,
    3: QA_PROMPT_TEMPORAL,
    4: QA_PROMPT_OPEN_DOMAIN,
}

QA_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def extract_token_info(response, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(response, dict) and isinstance(response.get("_usage"), dict):
        usage = response["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def usage_to_token_info(usage: Dict, model_path: str = "") -> Dict:
    return {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "model": model_path,
    }


def token_info_to_usage(token_info: Dict) -> Dict:
    return {
        "prompt_tokens": token_info.get("input", 0),
        "completion_tokens": token_info.get("output", 0),
    }


def format_retrieved_memories(retrieved_pairs: List[Tuple]) -> str:
    if not retrieved_pairs:
        return "No relevant memories found."

    parts = []
    for idx, (memory, _score) in enumerate(retrieved_pairs, start=1):
        parts.append(
            f"[Memory {idx}] Session {memory.session_id} | {memory.date_time}\n"
            f"[{memory.dia_id}] {memory.content}"
        )
    return "\n\n".join(parts)


class BatchedDenseRunner:
    def __init__(self, llm_client, model_path: str, shared_embedding_model, config_metadata: Optional[Dict] = None):
        self.llm_client = llm_client
        self.model_path = model_path
        self.embedding_model = shared_embedding_model
        self.config_metadata = config_metadata or {}

    def _choice_order_is_adversarial_first(self, choice_order_seed: Optional[str]) -> bool:
        if choice_order_seed is None:
            return True
        digest = hashlib.sha256(choice_order_seed.encode("utf-8")).hexdigest()
        return int(digest[-1], 16) % 2 == 0

    def _build_qa_prompt(
        self,
        qa: QAPair,
        context_str: str,
        choice_order_seed: Optional[str] = None,
    ) -> Tuple[str, float]:
        if qa.category == 5:
            not_mentioned = "Not mentioned in the conversation"
            if self._choice_order_is_adversarial_first(choice_order_seed):
                choice_a, choice_b = qa.adversarial_answer or "", not_mentioned
            else:
                choice_a, choice_b = not_mentioned, qa.adversarial_answer or ""
            prompt = QA_PROMPT_ADVERSARIAL.format(
                context=context_str,
                question=qa.question,
                choice_a=choice_a,
                choice_b=choice_b,
            )
            temperature = cfg.TEMPERATURE_C5
        else:
            prompt_template = QA_PROMPTS_BY_CATEGORY.get(
                qa.category,
                QA_PROMPT_SINGLE_HOP,
            )
            prompt = prompt_template.format(context=context_str, question=qa.question)
            temperature = cfg.TEMPERATURE
        return prompt, temperature

    def _generate_qa_raw(self, prompt: str, temperature: float) -> Dict:
        try:
            raw = self.llm_client.generate(
                prompt=prompt,
                guided_json=QA_SCHEMA,
                temperature=temperature,
                max_tokens=cfg.MAX_TOKENS,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
        except json.JSONDecodeError:
            logger.warning("QA: guided_json failed; retrying without guided decoding")
            fallback = self.llm_client.generate(
                prompt=prompt,
                guided_json=None,
                temperature=temperature,
                max_tokens=cfg.MAX_TOKENS,
                json_retry=1,
                return_usage=True,
            )
            text = fallback.get("content", "") if isinstance(fallback, dict) else str(fallback)
            try:
                raw = _parse_json_robust(text)
            except json.JSONDecodeError:
                logger.warning("QA: fallback also failed to produce valid JSON; using empty answer")
                raw = {"answer": ""}
            raw["_usage"] = fallback.get("_usage", {}) if isinstance(fallback, dict) else {}
        return raw

    def _flatten_sample_turns(self, sample: Sample) -> List[Dict]:
        turns = []
        for session in sample.sessions:
            for turn in session.turns:
                turns.append(
                    {
                        "sample_id": sample.sample_id,
                        "session_id": session.session_id,
                        "date_time": session.date_time,
                        "dia_id": turn.dia_id,
                        "speaker": turn.speaker,
                        "text": turn.text,
                        "memory_content": turn.to_memory_content(),
                    }
                )
        return turns

    def run_batch(
        self,
        samples: List[Sample],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        llm_call_log_dirs: List[Path],
    ) -> List[Dict]:
        from dense_store import DenseMemoryStore

        stores = [DenseMemoryStore(k=cfg.RETRIEVE_K) for _ in samples]
        llm_loggers: List[Optional[LLMCallLogger]] = []
        for llm_call_log_dir in llm_call_log_dirs:
            if cfg.ENABLE_LLM_CALL_LOGGING:
                llm_loggers.append(LLMCallLogger(llm_call_log_dir))
            else:
                llm_loggers.append(None)

        for sample in samples:
            total_turns = sum(len(session.turns) for session in sample.sessions)
            logger.info(
                f"Sample {sample.sample_id} "
                f"({len(sample.sessions)} sessions, {total_turns} turns, {len(sample.qa)} QA)"
            )

        self._run_phase1_batched(samples, stores, retrieval_log_paths)
        for sample, store in zip(samples, stores):
            logger.info(f"Phase 1 done: sample {sample.sample_id}, {len(store)} turns stored in memory")

        memory_stats_list = [store.get_memory_stats() for store in stores]
        qa_results_list, phase2_stats = self._run_phase2_batched(
            samples,
            stores,
            retrieval_log_paths,
            llm_loggers,
        )

        results = []
        for sample, store, qa_results, stats, memory_stats in zip(
            samples,
            stores,
            qa_results_list,
            phase2_stats,
            memory_stats_list,
        ):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"sample_{sample.sample_id}"
                store.save_snapshot(snapshot_dir)
                logger.info(f"Memory snapshot saved: sample {sample.sample_id}")

            store.clear()
            logger.info(f"Memory cleared: sample {sample.sample_id}")

            token_stats = {
                "call_1_qa": {
                    "input": stats["qa_input"],
                    "output": stats["qa_output"],
                    "llm_calls": stats["num_qa_calls"],
                },
                "total_input": stats["qa_input"],
                "total_output": stats["qa_output"],
                "total_llm_calls": stats["num_qa_calls"],
            }

            results.append(
                {
                    "sample_id": sample.sample_id,
                    "config_metadata": self.config_metadata,
                    "memory_at_qa_start": memory_stats,
                    "qa_results": qa_results,
                    "token_statistics": token_stats,
                    "memory_snapshot_path": (
                        f"memory_snapshots/sample_{sample.sample_id}/"
                        if cfg.SAVE_MEMORY_SNAPSHOTS else None
                    ),
                }
            )

        return results

    def _run_phase1_batched(
        self,
        samples: List[Sample],
        stores,
        retrieval_log_paths: List[Path],
    ) -> None:
        flat_turns_list = [self._flatten_sample_turns(sample) for sample in samples]
        max_turns = max((len(turns) for turns in flat_turns_list), default=0)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active = [
                (i, samples[i], stores[i], flat_turns_list[i][turn_idx])
                for i in range(len(samples))
                if turn_idx < len(flat_turns_list[i])
            ]
            if not active:
                continue

            memory_contents = [turn_info["memory_content"] for _, _, _, turn_info in active]
            memory_embeddings = self.embedding_model.encode(
                memory_contents,
                batch_size=32,
                show_progress_bar=False,
            )

            for idx, (sample_idx, sample, store, turn_info) in enumerate(active):
                retrieved = store.retrieve(memory_embeddings[idx])
                write_retrieval_log(
                    retrieval_log_paths[sample_idx],
                    build_retrieval_log_entry(
                        phase="memory_construction",
                        sample_id=sample.sample_id,
                        session_id=turn_info["session_id"],
                        dia_id=turn_info["dia_id"],
                        query=turn_info["memory_content"],
                        retrieved_pairs=retrieved,
                        store_size_at_retrieval=len(store),
                    ),
                )

            for idx, (_sample_idx, _sample, store, turn_info) in enumerate(active):
                store.store(
                    content=turn_info["memory_content"],
                    embedding=memory_embeddings[idx],
                    sample_id=turn_info["sample_id"],
                    session_id=turn_info["session_id"],
                    dia_id=turn_info["dia_id"],
                    speaker=turn_info["speaker"],
                    date_time=turn_info["date_time"],
                )

    def _run_phase2_batched(
        self,
        samples: List[Sample],
        stores,
        retrieval_log_paths: List[Path],
        llm_loggers: List[Optional[LLMCallLogger]],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        qa_refs = []
        for sample_idx, sample in enumerate(samples):
            for qa_idx, qa in enumerate(sample.qa):
                qa_refs.append((sample_idx, qa_idx, qa))

        qa_results_per_sample = [[] for _ in samples]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0}
            for _ in samples
        ]

        if not qa_refs:
            return qa_results_per_sample, phase2_stats

        question_embeddings = self.embedding_model.encode(
            [qa.question for _, _, qa in qa_refs],
            batch_size=32,
            show_progress_bar=False,
        )

        qa_jobs = []
        for idx, (sample_idx, qa_idx, qa) in enumerate(qa_refs):
            sample = samples[sample_idx]
            store = stores[sample_idx]
            retrieved = store.retrieve(question_embeddings[idx])

            write_retrieval_log(
                retrieval_log_paths[sample_idx],
                build_retrieval_log_entry(
                    phase="qa",
                    sample_id=sample.sample_id,
                    session_id=None,
                    dia_id=None,
                    query=qa.question,
                    retrieved_pairs=retrieved,
                    store_size_at_retrieval=len(store),
                ),
            )

            retrieved_metadata = [
                {
                    "dia_id": memory.dia_id,
                    "content_preview": memory.content[:120],
                }
                for memory, _ in retrieved
            ]

            context_str = format_retrieved_memories(retrieved)
            choice_seed = f"{sample.sample_id}::{qa_idx}::{qa.question}"
            prompt, temperature = self._build_qa_prompt(
                qa,
                context_str=context_str,
                choice_order_seed=choice_seed,
            )

            qa_jobs.append(
                {
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "sample_id": sample.sample_id,
                    "qa": qa,
                    "prompt": prompt,
                    "temperature": temperature,
                    "retrieved_metadata": retrieved_metadata,
                }
            )

        jobs_by_temperature: Dict[float, List[Dict]] = {}
        for job in qa_jobs:
            jobs_by_temperature.setdefault(job["temperature"], []).append(job)

        all_outputs = []
        for temperature, jobs in jobs_by_temperature.items():
            for chunk_start in tqdm(
                range(0, len(jobs), cfg.QA_BATCH_SIZE),
                desc=f"Phase2 QA temp={temperature}",
            ):
                chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
                chunk_results, chunk_usages = self._batch_generate_with_retry(
                    prompts=[job["prompt"] for job in chunk],
                    temperature=temperature,
                )
                for job, result, usage in zip(chunk, chunk_results, chunk_usages):
                    all_outputs.append((job, result, usage))

        all_outputs.sort(key=lambda item: (item[0]["sample_idx"], item[0]["qa_idx"]))

        for job, result, usage in all_outputs:
            sample_idx = job["sample_idx"]
            if llm_loggers[sample_idx] is not None:
                llm_loggers[sample_idx].log("call_1_qa", "", job["prompt"], result)

            token_info = usage_to_token_info(usage, self.model_path)
            phase2_stats[sample_idx]["qa_input"] += token_info["input"]
            phase2_stats[sample_idx]["qa_output"] += token_info["output"]
            phase2_stats[sample_idx]["num_qa_calls"] += 1

            qa = job["qa"]
            answer = result.get("answer", "") if isinstance(result, dict) else ""
            qa_results_per_sample[sample_idx].append(
                {
                    "question": qa.question,
                    "category": qa.category,
                    "generated_answer": answer,
                    "ground_truth_answer": qa.final_answer,
                    "evidence": qa.evidence,
                    "retrieved_memories": job["retrieved_metadata"],
                    "qa_tokens": token_info,
                }
            )

        return qa_results_per_sample, phase2_stats

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        temperature: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        if not prompts:
            return [], []

        try:
            texts, usages = self.llm_client.generate_batch_raw(
                prompts=prompts,
                system_prompt=None,
                max_tokens=cfg.MAX_TOKENS,
                temperature=temperature,
                guided_json=QA_SCHEMA,
                return_usage=True,
            )
            if len(texts) != len(prompts) or len(usages) != len(prompts):
                raise RuntimeError("Batch generation returned a mismatched number of outputs")
        except Exception as exc:
            logger.warning(f"Batch generation failed; falling back to sequential retry for whole chunk: {exc}")
            results = []
            usages = []
            for prompt in prompts:
                raw = self._generate_qa_raw(prompt, temperature)
                results.append(
                    {
                        key: value
                        for key, value in raw.items()
                        if key != "_usage"
                    }
                    if isinstance(raw, dict) else {}
                )
                usages.append(token_info_to_usage(extract_token_info(raw)))
            return results, usages

        from llm_client import _parse_json_response

        parsed: List[Optional[Dict]] = []
        retry_indices = []
        for idx, text in enumerate(texts):
            try:
                parsed.append(_parse_json_response(text) if isinstance(text, str) else text)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed.append(_parse_json_robust(text) if isinstance(text, str) else text)
                except (json.JSONDecodeError, ValueError):
                    parsed.append(None)
                    retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
            raw = self._generate_qa_raw(prompts[idx], temperature)
            usages[idx] = token_info_to_usage(extract_token_info(raw))
            if isinstance(raw, dict):
                raw.pop("_usage", None)
                parsed[idx] = raw
            else:
                parsed[idx] = {}

        return [item if isinstance(item, dict) else {} for item in parsed], usages


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
    if engine == "together":
        return _create(engine="together", **cfg.TOGETHER_CONFIG)
    if engine == "openai":
        return _create(engine="openai", **cfg.OPENAI_CONFIG)
    raise ValueError(f"Unknown LLM engine: {engine}")


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

    parser = argparse.ArgumentParser(
        description="Dense Retrieval Batch Experiment on LoCoMo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-sample", type=int, required=True, help="First sample index (inclusive)")
    parser.add_argument("--end-sample", type=int, required=True, help="Last sample index (inclusive)")
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None, help="Model max context length.")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="Samples to process in parallel.")
    parser.add_argument("--config", type=str, default="config_0", help="Config file name (without .py).")
    parser.add_argument("--engine", type=str, default=None, help="Override LLM engine (vllm, together, openai).")
    args = parser.parse_args()

    if args.engine is not None:
        cfg.LLM_ENGINE = args.engine

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_sample > args.end_sample:
        parser.error("--start-sample must be <= --end-sample")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    cfg.ensure_directories(args.model, args.start_sample, args.end_sample, config_name)

    sample_dir = cfg.get_sample_dir(args.model, args.start_sample, args.end_sample, config_name)
    global logger
    logger = setup_logging(sample_dir / "logs")

    if cfg.LLM_ENGINE != "vllm":
        logger.warning(
            f"LLM engine is '{cfg.LLM_ENGINE}'. "
            "Batch generate_batch_raw() is expected to work best with vllm."
        )

    results_file = cfg.get_results_file(args.model, args.start_sample, args.end_sample, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.start_sample, args.end_sample, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.start_sample, args.end_sample, config_name)
    snapshots_dir = cfg.get_memory_snapshots_dir(args.model, args.start_sample, args.end_sample, config_name)
    prompt_log_dir = cfg.get_prompt_log_dir(args.model, args.start_sample, args.end_sample, config_name)

    logger.info("=" * 60)
    logger.info("Dense Retrieval Batch Experiment — LoCoMo")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Samples         : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
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

    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer

    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model,
        args.tensor_parallel,
        args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    config_metadata = {
        "config_name": config_name,
        "model": args.model,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "temperature": cfg.TEMPERATURE,
        "temperature_c5": cfg.TEMPERATURE_C5,
        "max_tokens": cfg.MAX_TOKENS,
        "sample_range": [args.start_sample, args.end_sample],
        "retrieve_k": cfg.RETRIEVE_K,
        "batch_size": args.batch_size,
        "qa_batch_size": cfg.QA_BATCH_SIZE,
    }

    runner = BatchedDenseRunner(
        llm_client=llm_client,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
        config_metadata=config_metadata,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print("# Dense Retrieval Batch  |  LoCoMo")
    print(f"# Model      : {cfg.extract_model_name(args.model)}")
    print(f"# Embedding  : {cfg.EMBEDDING_MODEL}  |  K={cfg.RETRIEVE_K}")
    print(f"# Samples [{args.start_sample}, {args.end_sample}]  ({len(pending_samples)} to process)")
    print(f"# Batch size    : {args.batch_size}")
    print(f"# QA batch size : {cfg.QA_BATCH_SIZE}")
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
