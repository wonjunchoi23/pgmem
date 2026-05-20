"""
Dense Retrieval Experiment Runner — PersonaMem (Batched, QA-Only Variant)

Usage:
    python run_experiment.py \\
        --start-session 0 --end-session 36 \\
        --benchmark-size 32k \\
        --model Qwen/Qwen3-1.7B \\
        --tensor-parallel 1 --gpu-memory 0.9 \\
        --batch-size 4 \\
        --config config_0
"""

import os
import re
import sys
import json
import logging
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore

from load_dataset import (
    load_personamem_dataset,
    PersonaMemContext,
    PersonaMemMessage,
    PersonaMemQAPair,
)


# =============================================================================
# MESSAGE PAIRING HELPERS
# =============================================================================

def _get_turn_pairs(context: PersonaMemContext) -> List[Tuple]:
    """
    Pair PersonaMem messages into (user_msg, asst_msg, block_idx, pair_idx_in_block).

    Groups messages by block_idx and pairs consecutive user/assistant messages.
    Mirrors ImplexConv's get_turn_pairs() logic.
    """
    pairs = []
    blocks: Dict[int, List[PersonaMemMessage]] = {}
    for msg in context.messages:
        blocks.setdefault(msg.block_idx, []).append(msg)

    for block_idx in sorted(blocks.keys()):
        block_msgs = blocks[block_idx]
        pair_idx = 0
        i = 0
        while i < len(block_msgs):
            user_msg = asst_msg = None
            if i < len(block_msgs) and block_msgs[i].role == "user":
                user_msg = block_msgs[i]
                i += 1
            if i < len(block_msgs) and block_msgs[i].role == "assistant":
                asst_msg = block_msgs[i]
                i += 1
            if user_msg is not None:
                pairs.append((user_msg, asst_msg, block_idx, pair_idx))
                pair_idx += 1

    return pairs


def _strip_prefix(content: str, role: str) -> str:
    """Remove 'User: ' or 'Assistant: ' prefix embedded in PersonaMem content."""
    prefix = "User: " if role == "user" else "Assistant: "
    return content[len(prefix):] if content.startswith(prefix) else content


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"dense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path) -> Set[int]:
    if not checkpoint_file.exists():
        return set()
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        if "completed_session_ids" in data:
            return set(data["completed_session_ids"])
        last = data.get("last_completed_session_index")
        if last is None:
            return set()
        return set(range(last + 1))
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def save_checkpoint(checkpoint_file: Path, completed_ids: Set[int],
                    model_path: str, benchmark_size: str,
                    start_session: int, end_session: int,
                    config_name: str = "config_0"):
    data = {
        "completed_session_ids": sorted(completed_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name":    config_name,
            "model":          model_path,
            "benchmark_size": benchmark_size,
            "start_session":  start_session,
            "end_session":    end_session,
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


def build_retrieval_log_entry(
    phase: str,
    context_index: int,
    block_idx: int,
    pair_idx: int,
    query: str,
    retrieved_pairs: List[Tuple],
    store_size_at_retrieval: int,
) -> Dict:
    retrieved_items = [
        {
            "content_preview": mem.content[:100],
            "score": score,
            "source_turn": {
                "context_index": mem.session_id,  # internally stores context_index
                "block_idx":     mem.conv_id,     # internally stores block_idx
                "pair_idx":      mem.turn_id,     # internally stores pair_idx_in_block
            },
        }
        for mem, score in retrieved_pairs
    ]
    return {
        "timestamp":      datetime.now().isoformat(),
        "phase":          phase,
        "context_index":  context_index,
        "block_idx":      block_idx,
        "pair_idx":       pair_idx,
        "query":          query[:500],
        "memory_type":    "dense_vector",
        "num_retrieved":  len(retrieved_pairs),
        "retrieved_items": retrieved_items,
        "module_specific": {
            "store_size_at_retrieval": store_size_at_retrieval,
        },
    }


def _remap_retrieved_memories(retrieved_pairs: List[Tuple]) -> List[Dict]:
    """Build retrieved_memories list with PersonaMem field names."""
    return [
        {
            "context_index": mem.session_id,
            "block_idx":     mem.conv_id,
            "pair_idx":      mem.turn_id,
        }
        for mem, _ in retrieved_pairs
    ]


# =============================================================================
# LLM CALL LOGGING
# =============================================================================

class LLMCallLogger:
    CALL_DIRS = ["call_1_qa"]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "timestamp":     datetime.now().isoformat(),
            "call_type":     call_type,
            "system_prompt": system_prompt,
            "user_prompt":   user_prompt,
            "output":        output if not isinstance(output, dict) else {
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# JSON PARSING UTILITIES
# =============================================================================

def _escape_control_chars_in_strings(text: str) -> str:
    result = []
    in_string = False
    prev_backslash = False
    for ch in text:
        if prev_backslash:
            result.append(ch)
            prev_backslash = False
        elif ch == '\\' and in_string:
            result.append(ch)
            prev_backslash = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
    return ''.join(result)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r',(\s*[}\]])', r'\1', text)


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

    def _try_all(t: str):
        for fn in [
            lambda s: json.loads(s),
            lambda s: json.loads(_escape_control_chars_in_strings(s)),
            lambda s: json.loads(_remove_trailing_commas(s)),
            lambda s: json.loads(_remove_trailing_commas(_escape_control_chars_in_strings(s))),
        ]:
            try:
                return fn(t)
            except json.JSONDecodeError:
                pass
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        result = _try_all(text)
        if result is not None:
            return result
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = _try_all(match.group())
            if result is not None:
                return result
        raise original_error


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

QA_PROMPT_MULTICHOICE = """\
You are a helpful assistant. Answer the multiple-choice question based only on the retrieved memories provided.

Retrieved memories:
{context}

Question: {question}

Options:
{options_text}

Choose the single best answer (a, b, c, or d).
Answer in JSON format:
{{"answer": "<a | b | c | d>"}}"""

QA_SCHEMA_MULTICHOICE = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["a", "b", "c", "d"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


# =============================================================================
# TOKEN UTILITIES
# =============================================================================

def extract_token_info(response, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(response, dict) and "_usage" in response:
        usage = response["_usage"] or {}
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def generate_json_with_fallback(client, prompt: str, schema: Dict) -> Tuple[Dict, bool]:
    fallback_used = False
    try:
        raw = client.generate(
            prompt=prompt,
            guided_json=schema,
            temperature=cfg.TEMPERATURE,
            max_tokens=cfg.MAX_TOKENS,
            json_retry=cfg.JSON_RETRY,
            return_usage=True,
        )
    except json.JSONDecodeError:
        fallback_used = True
        logger.warning("QA: guided_json failed; retrying without guided decoding")
        fallback = client.generate(
            prompt=prompt,
            guided_json=None,
            temperature=cfg.TEMPERATURE,
            max_tokens=cfg.MAX_TOKENS,
            json_retry=1,
            return_usage=True,
        )
        text = fallback.get("content", "") if isinstance(fallback, dict) else str(fallback)
        try:
            raw = _parse_json_robust(text)
        except json.JSONDecodeError:
            logger.warning("QA: fallback also failed; using empty answer")
            raw = {"answer": ""}
        raw["_usage"] = fallback.get("_usage") if isinstance(fallback, dict) else None
    return raw, fallback_used


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_retrieved_memories(retrieved_pairs: List[Tuple]) -> str:
    if not retrieved_pairs:
        return "No relevant memories found."
    parts = []
    for idx, (mem, _score) in enumerate(retrieved_pairs, start=1):
        parts.append(f"[Memory {idx}] {mem.content}")
    return "\n\n".join(parts)


# =============================================================================
# BATCHED DENSE RETRIEVAL RUNNER
# =============================================================================

class BatchedDenseRunner:
    """Processes multiple PersonaMem contexts in parallel with batched embedding."""

    QA_SYSTEM_PROMPT = None  # instruction is in the user prompt

    def __init__(
        self,
        llm_client,
        benchmark_size: str,
        model_path: str,
        shared_embedding_model,
    ):
        self.llm_client = llm_client
        self.benchmark_size = benchmark_size
        self.model_path = model_path
        self.embedding_model = shared_embedding_model

    def run_batch(
        self,
        contexts: List[PersonaMemContext],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
        config_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        from dense_store import DenseMemoryStore

        stores = [
            DenseMemoryStore(self.embedding_model, k=cfg.RETRIEVE_K)
            for _ in contexts
        ]

        llm_loggers: List[Optional[LLMCallLogger]] = [
            LLMCallLogger(d) if cfg.ENABLE_LLM_CALL_LOGGING else None
            for d in prompt_log_dirs
        ]

        for context in contexts:
            logger.info(
                f"Context {context.context_index}  "
                f"({len(_get_turn_pairs(context))} turn pairs, "
                f"{len(context.qa_pairs)} QA)"
            )

        # Phase 1: memory construction (embedding only, no LLM)
        self._run_phase1_batched(contexts, stores, retrieval_log_paths)

        memory_stats_list = [store.get_memory_stats() for store in stores]

        # Phase 2: QA answering
        qa_results_list, phase2_stats = self._run_phase2_batched(
            contexts, stores, retrieval_log_paths, llm_loggers
        )

        # Phase 3: cleanup + aggregate results
        results = []
        for i, (context, store) in enumerate(zip(contexts, stores)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"session_{context.context_index}"
                store.save_snapshot(snapshot_dir)
                logger.info(f"Memory snapshot saved: context {context.context_index}")

            store.clear()
            logger.info(f"Memory cleared: context {context.context_index}")

            p2 = phase2_stats[i]
            token_stats = {
                "call_1_qa": {
                    "input":               p2["qa_input"],
                    "output":              p2["qa_output"],
                    "llm_calls":           p2["num_qa_calls"],
                    "parse_fallback_count": p2["qa_parse_fallback_count"],
                },
                "total_input":     p2["qa_input"],
                "total_output":    p2["qa_output"],
                "total_llm_calls": p2["num_qa_calls"],
            }

            result = {
                "context_index":     context.context_index,
                "persona_id":        context.persona_id,
                "shared_context_id": context.shared_context_id,
                "config_metadata":   config_metadata or {},
                "memory_at_qa_start": memory_stats_list[i],
                "qa_results":        qa_results_list[i],
                "token_statistics":  token_stats,
                "memory_snapshot_path": (
                    f"memory_snapshots/session_{context.context_index}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            }
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_batched(
        self,
        contexts: List[PersonaMemContext],
        stores,
        retrieval_log_paths: List[Path],
    ) -> None:
        """Memory construction — batch-embed all pairs, store in dense store.

        For each pair_idx:
          1. Batch-encode user queries → retrieve BEFORE storing
          2. Write retrieval logs
          3. Build pair contents (stripped + re-prefixed)
          4. Batch-encode pair contents → store
        """
        context_pairs = [_get_turn_pairs(ctx) for ctx in contexts]
        max_pairs = max((len(p) for p in context_pairs), default=0)

        for pair_idx in tqdm(range(max_pairs), desc="Phase1 pairs"):
            active = [
                (i, contexts[i], stores[i], context_pairs[i][pair_idx])
                for i in range(len(contexts))
                if pair_idx < len(context_pairs[i])
            ]
            if not active:
                continue

            # Step 1: batch-encode queries (stripped user utterances)
            queries = [
                _strip_prefix(user_msg.content, "user")
                for (_, _, _, (user_msg, _, _, _)) in active
            ]
            query_embeddings = self.embedding_model.encode(
                queries, batch_size=32, show_progress_bar=False
            )

            # Step 2: retrieve BEFORE storing + write logs
            for k, (i, context, store, (user_msg, asst_msg, block_idx, pair_in_block)) in enumerate(active):
                store_size = len(store._memories)
                retrieved = store.retrieve(query_embeddings[k])
                log_entry = build_retrieval_log_entry(
                    phase="prompt_construction",
                    context_index=context.context_index,
                    block_idx=block_idx,
                    pair_idx=pair_in_block,
                    query=queries[k],
                    retrieved_pairs=retrieved,
                    store_size_at_retrieval=store_size,
                )
                write_retrieval_log(retrieval_log_paths[i], log_entry)

            # Step 3: build pair contents (strip prefix, re-prefix cleanly)
            pair_contents = []
            for (_, _, _, (user_msg, asst_msg, _, _)) in active:
                user_text = _strip_prefix(user_msg.content, "user")
                if asst_msg is not None:
                    asst_text = _strip_prefix(asst_msg.content, "assistant")
                    pair_contents.append(f"User: {user_text}\nAssistant: {asst_text}")
                else:
                    pair_contents.append(f"User: {user_text}")

            # Step 4: batch-encode pair contents
            pair_embeddings = self.embedding_model.encode(
                pair_contents, batch_size=32, show_progress_bar=False
            )

            # Step 5: store (context_index→session_id, block_idx→conv_id, pair_idx→turn_id)
            for k, (i, context, store, (user_msg, asst_msg, block_idx, pair_in_block)) in enumerate(active):
                store.store(
                    content=pair_contents[k],
                    embedding=pair_embeddings[k],
                    session_id=context.context_index,
                    conv_id=block_idx,
                    turn_id=pair_in_block,
                )

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        contexts: List[PersonaMemContext],
        stores,
        retrieval_log_paths: List[Path],
        llm_loggers: List[Optional[LLMCallLogger]],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """QA answering — batch-encode all questions, retrieve, batch-generate."""

        # Collect all QA questions
        all_questions = []
        for i, context in enumerate(contexts):
            for qa in context.qa_pairs:
                all_questions.append((i, qa))

        if not all_questions:
            return [[] for _ in contexts], [
                {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0, "qa_parse_fallback_count": 0}
                for _ in contexts
            ]

        # Batch-encode all QA questions
        question_texts = [qa.question for (_, qa) in all_questions]
        qa_embeddings = self.embedding_model.encode(
            question_texts, batch_size=32, show_progress_bar=False
        )

        # Build all QA prompts with retrieval
        qa_jobs = []
        for k, (i, qa) in enumerate(all_questions):
            store = stores[i]
            context = contexts[i]
            qa_emb = qa_embeddings[k]

            store_size = len(store._memories)
            retrieved = store.retrieve(qa_emb)

            write_retrieval_log(
                retrieval_log_paths[i],
                build_retrieval_log_entry(
                    phase="qa",
                    context_index=context.context_index,
                    block_idx=-1,
                    pair_idx=-1,
                    query=qa.question,
                    retrieved_pairs=retrieved,
                    store_size_at_retrieval=store_size,
                ),
            )

            context_str = format_retrieved_memories(retrieved)
            options_text = "\n".join(qa.all_options)
            prompt_content = QA_PROMPT_MULTICHOICE.format(
                context=context_str,
                question=qa.question,
                options_text=options_text,
            )
            retrieved_metadata = _remap_retrieved_memories(retrieved)
            qa_jobs.append((i, qa, retrieved_metadata, prompt_content))

        # Batch generate in QA_BATCH_SIZE chunks
        all_pairs = []

        for chunk_start in tqdm(range(0, len(qa_jobs), cfg.QA_BATCH_SIZE), desc="Phase2 QA chunks"):
            chunk = qa_jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            chunk_prompts = [job[3] for job in chunk]

            chunk_results, chunk_usages, chunk_fallbacks = self._batch_generate_with_retry(
                chunk_prompts, QA_SCHEMA_MULTICHOICE
            )

            for job, result, usage, fallback_used in zip(
                chunk, chunk_results, chunk_usages, chunk_fallbacks
            ):
                all_pairs.append((job, result, usage, fallback_used))

        # Distribute results
        qa_results_per_context: List[List[Dict]] = [[] for _ in contexts]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0, "qa_parse_fallback_count": 0}
            for _ in contexts
        ]

        for (i, qa, retrieved_metadata, prompt_content), result, usage, fallback_used in all_pairs:
            if llm_loggers[i] is not None:
                llm_loggers[i].log("call_1_qa", "", prompt_content, result)

            qa_tokens = {
                "input":  usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "model":  self.model_path,
            }

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            label = answer.strip().lower()
            answer = label if label in ("a", "b", "c", "d") else "unknown"

            qa_results_per_context[i].append({
                "question":                    qa.question,
                "question_type":               qa.question_type,
                "topic":                       qa.topic,
                "all_options":                 qa.all_options,
                "generated_answer":            answer,
                "ground_truth_answer":         qa.correct_answer,
                "end_index_in_shared_context": qa.end_index_in_shared_context,
                "retrieved_memories":          retrieved_metadata,
                "qa_tokens":                   qa_tokens,
            })

            phase2_stats[i]["qa_input"]               += qa_tokens["input"]
            phase2_stats[i]["qa_output"]              += qa_tokens["output"]
            phase2_stats[i]["num_qa_calls"]           += 1
            phase2_stats[i]["qa_parse_fallback_count"] += int(fallback_used)

        return qa_results_per_context, phase2_stats

    # ------------------------------------------------------------------
    # Batch generate with per-item JSON retry fallback
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        schema: Dict,
    ) -> Tuple[List[Dict], List[Dict], List[bool]]:
        if not prompts:
            return [], [], []

        texts, usages = self.llm_client.generate_batch_raw(
            prompts=prompts,
            system_prompt=self.QA_SYSTEM_PROMPT,
            max_tokens=cfg.MAX_TOKENS,
            temperature=cfg.TEMPERATURE,
            guided_json=schema,
            return_usage=True,
        )

        from llm_client import _parse_json_response

        parsed: List[Optional[Dict]] = []
        retry_indices = []
        fallback_flags = [False] * len(prompts)

        for idx, text in enumerate(texts):
            try:
                parsed.append(_parse_json_response(text) if isinstance(text, str) else text)
            except (json.JSONDecodeError, ValueError):
                parsed.append(None)
                retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(f"Batch item {idx} failed JSON parse — retrying sequentially")
            fallback_flags[idx] = True
            try:
                raw, _ = generate_json_with_fallback(self.llm_client, prompts[idx], schema)
                usage_info = (raw.pop("_usage", {}) or {}) if isinstance(raw, dict) else {}
                usages[idx] = {
                    "prompt_tokens":     usage_info.get("prompt_tokens", 0),
                    "completion_tokens": usage_info.get("completion_tokens", 0),
                }
                parsed[idx] = raw if isinstance(raw, dict) else {}
            except Exception as e:
                logger.error(f"Sequential retry for batch item {idx} failed: {e}")
                parsed[idx] = {}

        return [item if isinstance(item, dict) else {} for item in parsed], usages, fallback_flags


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
    # Step 1: load config
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

    # Step 2: parse args
    parser = argparse.ArgumentParser(
        description="Dense Retrieval QA-Only Baseline — PersonaMem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-session",   type=int, required=True)
    parser.add_argument("--end-session",     type=int, required=True)
    parser.add_argument("--benchmark-size",  type=str, required=True,
                        choices=cfg.BENCHMARK_SIZES)
    parser.add_argument("--model",           type=str,
                        default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory",      type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len",   type=int, default=None)
    parser.add_argument("--batch-size",      type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config",          type=str, default="config_0")
    parser.add_argument("--engine",          type=str, default=None,
                        help="Override LLM engine (vllm, together, openai).")
    args = parser.parse_args()

    if args.engine is not None:
        cfg.LLM_ENGINE = args.engine

    # Step 3: validate
    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    # Step 4: directories and logging
    cfg.ensure_directories(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )
    session_dir = cfg.get_session_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file      = cfg.get_results_file(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )
    checkpoint_file   = cfg.get_checkpoint_file(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )
    retrieval_log_dir = cfg.get_retrieval_log_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )
    snapshots_dir     = cfg.get_memory_snapshots_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )
    prompt_log_dir    = cfg.get_prompt_log_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )

    logger.info("=" * 60)
    logger.info("Dense Retrieval Batch QA-Only Baseline — PersonaMem")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Benchmark size  : {args.benchmark_size}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    # Step 5: load dataset
    logger.info("Loading dataset...")
    questions_path, contexts_path = cfg.get_dataset_paths(args.benchmark_size)
    all_contexts = load_personamem_dataset(questions_path, contexts_path)

    if args.end_session >= len(all_contexts):
        logger.error(
            f"end_session={args.end_session} out of range "
            f"(dataset has {len(all_contexts)} contexts)"
        )
        return 1

    target_contexts = all_contexts[args.start_session: args.end_session + 1]

    # Step 6: checkpoint resume
    completed_ids: Set[int] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_ids = load_checkpoint(checkpoint_file)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} contexts already completed")

    pending_contexts = [c for c in target_contexts if c.context_index not in completed_ids]
    if not pending_contexts:
        logger.info("All contexts already completed.")
        return 0

    # Step 7: load embedding model
    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    # Step 8: init LLM
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    runner = BatchedDenseRunner(
        llm_client=llm_client,
        benchmark_size=args.benchmark_size,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
    )

    config_metadata = {
        "config_name":    config_name,
        "model":          args.model,
        "benchmark_size": args.benchmark_size,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "retrieve_k":     cfg.RETRIEVE_K,
        "temperature":    cfg.TEMPERATURE,
        "max_tokens":     cfg.MAX_TOKENS,
        "session_range":  [args.start_session, args.end_session],
    }

    results = load_existing_results(results_file)

    print(f"\n{'#'*60}")
    print(f"# Dense Retrieval Batch QA-Only  |  benchmark_size={args.benchmark_size}")
    print(f"# Model     : {cfg.extract_model_name(args.model)}")
    print(f"# Embedding : {cfg.EMBEDDING_MODEL}  |  K={cfg.RETRIEVE_K}")
    print(f"# Batch size : {args.batch_size}")
    print(f"# Contexts [{args.start_session}, {args.end_session}]  ({len(pending_contexts)} to process)")
    print(f"{'#'*60}\n")

    # Step 9: main loop
    num_batches = (len(pending_contexts) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_contexts[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        context_indices = [c.context_index for c in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: contexts {context_indices}")

        batch_retrieval_log_paths = [
            retrieval_log_dir / f"session_{c.context_index}_retrieval_log.jsonl"
            for c in batch
        ]
        batch_prompt_log_dirs = [
            prompt_log_dir / f"session_{c.context_index}"
            for c in batch
        ]

        try:
            batch_results = runner.run_batch(
                contexts=batch,
                retrieval_log_paths=batch_retrieval_log_paths,
                snapshots_dir=snapshots_dir,
                prompt_log_dirs=batch_prompt_log_dirs,
                config_metadata=config_metadata,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed with hard error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        for result in batch_results:
            results.append(result)
            save_results(results_file, results)

            if cfg.ENABLE_CHECKPOINTING:
                completed_ids.add(result["context_index"])
                save_checkpoint(
                    checkpoint_file, completed_ids,
                    args.model, args.benchmark_size,
                    args.start_session, args.end_session,
                    config_name=config_name,
                )

            logger.info(
                f"Context {result['context_index']} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total LLM calls: {result['token_statistics']['total_llm_calls']}"
            )

    print(f"\n{'#'*60}")
    print(f"# Batch experiment complete!  Results -> {results_file}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
