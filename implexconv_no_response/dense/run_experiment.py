"""
Dense Retrieval Experiment Runner — ImplexConv (Batched, QA-Only Variant)

Usage:
    python run_experiment.py \\
        --start-session 0 --end-session 99 \\
        --subset opposed \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --tensor-parallel 1 --gpu-memory 0.5 \\
        --batch-size 4 \\
        --config config
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

# Suppress noisy logs before any imports
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

# cfg is loaded dynamically in main() based on --config argument.
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
        log_file = log_dir / f"dense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path, target_session_ids: Optional[List[int]] = None) -> Set[int]:
    """Load completed session IDs from checkpoint.

    Supports both the new set-based format and the old sequential format
    storing last_completed_session_index. When old format is detected, the
    index is mapped onto the provided target_session_ids range.
    """
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

        if target_session_ids is None:
            return set(range(last + 1))

        capped = max(0, min(last + 1, len(target_session_ids)))
        return set(target_session_ids[:capped])
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def save_checkpoint(checkpoint_file: Path, completed_ids: Set[int],
                    model_path: str, subset: str,
                    start_session: int, end_session: int,
                    config_name: str = "config"):
    data = {
        "completed_session_ids": sorted(completed_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name":   config_name,
            "model":         model_path,
            "subset":        subset,
            "start_session": start_session,
            "end_session":   end_session,
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
    """Append one retrieval log entry (JSONL) to the session log file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    phase: str,
    session_id: int,
    conv_id: int,
    turn_id: int,
    query: str,
    retrieved_pairs: List[Tuple],  # List of (MemoryUnit, score)
    store_size_at_retrieval: int,
) -> Dict:
    """Build a retrieval log entry for the dense retrieval module.

    Args:
        phase: "prompt_construction" (Phase 1) or "qa" (Phase 2).
        session_id, conv_id, turn_id: Identifying metadata for the turn.
        query: The query string used for retrieval.
        retrieved_pairs: List of (MemoryUnit, cosine_score) from store.retrieve().
        store_size_at_retrieval: Number of memories in the store at retrieval time.
    """
    retrieved_items = [
        {
            "content_preview": mem.content[:100],
            "score": score,
            "source_turn": {
                "session_id": mem.session_id,
                "conv_id":    mem.conv_id,
                "turn_id":    mem.turn_id,
            },
        }
        for mem, score in retrieved_pairs
    ]
    return {
        "timestamp":      datetime.now().isoformat(),
        "phase":          phase,
        "session_id":     session_id,
        "conv_id":        conv_id,
        "turn_id":        turn_id,
        "query":          query[:500],
        "memory_type":    "dense_vector",
        "num_retrieved":  len(retrieved_pairs),
        "retrieved_items": retrieved_items,
        "module_specific": {
            "store_size_at_retrieval": store_size_at_retrieval,
        },
    }


# =============================================================================
# LLM CALL LOGGING
# =============================================================================

class LLMCallLogger:
    """
    Logs all LLM calls (input + output) to call-type-specific folders as JSONL.

    Folder structure:
        {base_dir}/call_1_qa/calls.jsonl
    """

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
    """State-machine: escape literal newlines/tabs inside JSON string values."""
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
    """Remove trailing commas before } or ] (common LLM mistake)."""
    return re.sub(r',(\s*[}\]])', r'\1', text)


def _parse_json_robust(text: str) -> dict:
    """Parse JSON from raw LLM text with multiple fallback strategies."""
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

    def _try_all(t: str):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_escape_control_chars_in_strings(t))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(t))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(_escape_control_chars_in_strings(t)))
        except json.JSONDecodeError:
            pass
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        original_error = e

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

QA_PROMPT_OPPOSED = """\
You are a helpful assistant. Answer the question based only on the retrieved memories provided. Be concise (max 100 words).

Retrieved memories:
{context}

Question: {question}

Answer in JSON format:
{{"answer": "<your answer here>"}}"""

QA_SCHEMA_OPPOSED = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

QA_PROMPT_SUPPORTIVE = """\
You are a helpful assistant. Answer the yes/no question based only on the retrieved memories provided. You MUST answer with exactly one of: "yes" or "no".

Retrieved memories:
{context}

Question: {question}

Answer in JSON format using exactly one of: "yes", "no"
{{"answer": "<yes | no>"}}"""

QA_SCHEMA_SUPPORTIVE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]}
    },
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
    """Run one JSON generation and report whether fallback parsing was used."""
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
            logger.warning("QA: fallback also failed to produce valid JSON; using empty answer")
            raw = {"answer": ""}
        raw["_usage"] = fallback.get("_usage") if isinstance(fallback, dict) else None
    return raw, fallback_used


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_retrieved_memories(retrieved_pairs: List[Tuple]) -> str:
    """Format retrieved (MemoryUnit, score) pairs as a context string.

    Format:
        [Memory 1] User: {utt}
        Assistant: {utt}

        [Memory 2] ...

    Returns "No relevant memories found." if the list is empty.
    """
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
    """Processes multiple dense retrieval sessions in parallel."""

    QA_SYSTEM_PROMPT = None  # No system prompt — instruction is in the user prompt

    def __init__(
        self,
        llm_client,
        subset: str,
        model_path: str,
        shared_embedding_model,
    ):
        self.llm_client = llm_client
        self.subset = subset
        self.model_path = model_path
        self.embedding_model = shared_embedding_model

    def run_batch(
        self,
        sessions: List[Session],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
        config_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        from dense_store import DenseMemoryStore

        # One store per session
        stores = [
            DenseMemoryStore(self.embedding_model, k=cfg.RETRIEVE_K)
            for _ in sessions
        ]

        # One LLM call logger per session (optional)
        llm_loggers: List[Optional[LLMCallLogger]] = []
        for prompt_log_dir in prompt_log_dirs:
            if cfg.ENABLE_LLM_CALL_LOGGING:
                llm_loggers.append(LLMCallLogger(prompt_log_dir))
            else:
                llm_loggers.append(None)

        for session in sessions:
            logger.info(f"{'='*60}")
            logger.info(
                f"Session {session.session_id}  "
                f"({len(session.get_turn_pairs())} turn pairs, "
                f"{len(session.qa)} QA)"
            )
            logger.info(f"{'='*60}")

        # Phase 1: memory construction
        self._run_phase1_batched(sessions, stores, retrieval_log_paths)

        # Capture memory stats (before Phase 2, after Phase 1)
        memory_stats_list = [store.get_memory_stats() for store in stores]

        # Phase 2: QA answering
        qa_results_list, phase2_stats = self._run_phase2_batched(
            sessions, stores, retrieval_log_paths, llm_loggers
        )

        # Phase 3: cleanup + aggregate results
        results = []
        for i, (session, store) in enumerate(zip(sessions, stores)):
            # Save snapshot
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"session_{session.session_id}"
                store.save_snapshot(snapshot_dir)
                logger.info(f"Memory snapshot saved: session {session.session_id}")

            store.clear()
            logger.info(f"Memory cleared: session {session.session_id}")

            p2 = phase2_stats[i]
            qa_input  = p2["qa_input"]
            qa_output = p2["qa_output"]
            num_qa    = p2["num_qa_calls"]
            fallbacks = p2["qa_parse_fallback_count"]

            token_stats = {
                "call_1_qa": {
                    "input":               qa_input,
                    "output":              qa_output,
                    "llm_calls":           num_qa,
                    "parse_fallback_count": fallbacks,
                },
                "total_input":     qa_input,
                "total_output":    qa_output,
                "total_llm_calls": num_qa,
            }

            result = {
                "session_id":         session.session_id,
                "config_metadata":    config_metadata or {},
                "memory_at_qa_start": memory_stats_list[i],
                "qa_results":         qa_results_list[i],
                "token_statistics":   token_stats,
                "memory_snapshot_path": (
                    f"memory_snapshots/session_{session.session_id}/"
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
        sessions: List[Session],
        stores,
        retrieval_log_paths: List[Path],
    ) -> None:
        """Memory construction — interleaved across sessions, turn by turn.

        For each turn_idx:
          1. Batch-encode queries (user utterances) → retrieve BEFORE storing
          2. Write retrieval logs
          3. Batch-encode pair contents → store
        """
        # Pre-compute turn pairs per session once
        turn_pairs_list = [session.get_turn_pairs() for session in sessions]
        max_turns = max((len(tp) for tp in turn_pairs_list), default=0)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            # Gather active sessions at this turn_idx
            active = [
                (i, sessions[i], stores[i], turn_pairs_list[i][turn_idx])
                for i in range(len(sessions))
                if turn_idx < len(turn_pairs_list[i])
            ]

            if not active:
                continue

            # ── Step 1: Batch-encode queries ──────────────────────────
            queries = [user_turn.utterance for (_, _, _, (user_turn, _)) in active]
            query_embeddings = self.embedding_model.encode(
                queries, batch_size=32, show_progress_bar=False
            )  # shape: (len(active), dim)

            # ── Step 2: Retrieve BEFORE storing + write logs ──────────
            for k, (i, session, store, (user_turn, assistant_turn)) in enumerate(active):
                query_emb = query_embeddings[k]
                store_size = len(store._memories)
                retrieved = store.retrieve(query_emb)

                log_entry = build_retrieval_log_entry(
                    phase="prompt_construction",
                    session_id=session.session_id,
                    conv_id=user_turn.conv_id,
                    turn_id=user_turn.turn_id,
                    query=user_turn.utterance,
                    retrieved_pairs=retrieved,
                    store_size_at_retrieval=store_size,
                )
                write_retrieval_log(retrieval_log_paths[i], log_entry)

            # ── Step 3: Build pair contents ───────────────────────────
            pair_contents = []
            for (_, _, _, (user_turn, assistant_turn)) in active:
                if assistant_turn is not None:
                    content = f"User: {user_turn.utterance}\nAssistant: {assistant_turn.utterance}"
                else:
                    content = f"User: {user_turn.utterance}"
                pair_contents.append(content)

            # ── Step 4: Batch-encode pair contents ────────────────────
            pair_embeddings = self.embedding_model.encode(
                pair_contents, batch_size=32, show_progress_bar=False
            )  # shape: (len(active), dim)

            # ── Step 5: Store ─────────────────────────────────────────
            for k, (i, session, store, (user_turn, assistant_turn)) in enumerate(active):
                store.store(
                    content=pair_contents[k],
                    embedding=pair_embeddings[k],
                    session_id=session.session_id,
                    conv_id=user_turn.conv_id,
                    turn_id=user_turn.turn_id,
                )

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        sessions: List[Session],
        stores,
        retrieval_log_paths: List[Path],
        llm_loggers: List[Optional[LLMCallLogger]],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """QA answering — all sessions' QA prompts batched together."""
        schema = QA_SCHEMA_SUPPORTIVE if self.subset == "supportive" else QA_SCHEMA_OPPOSED
        prompt_template = QA_PROMPT_SUPPORTIVE if self.subset == "supportive" else QA_PROMPT_OPPOSED

        # ── Collect all QA questions for batch embedding ──────────────
        all_questions = []  # flat list of (session_idx, qa)
        for i, session in enumerate(sessions):
            for qa in session.qa:
                all_questions.append((i, qa))

        if not all_questions:
            empty_results = [[] for _ in sessions]
            empty_stats = [
                {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0, "qa_parse_fallback_count": 0}
                for _ in sessions
            ]
            return empty_results, empty_stats

        # ── Batch-encode all QA questions ─────────────────────────────
        question_texts = [qa.question for (_, qa) in all_questions]
        qa_embeddings = self.embedding_model.encode(
            question_texts, batch_size=32, show_progress_bar=False
        )  # shape: (total_qa, dim)

        # ── Build all QA prompts with retrieval ───────────────────────
        # qa_jobs: (session_idx, qa, retrieved_metadata, prompt_content)
        qa_jobs = []
        for k, (i, qa) in enumerate(all_questions):
            store = stores[i]
            session = sessions[i]
            qa_emb = qa_embeddings[k]

            store_size = len(store._memories)
            retrieved = store.retrieve(qa_emb)

            # Write retrieval log
            log_entry = build_retrieval_log_entry(
                phase="qa",
                session_id=session.session_id,
                conv_id=-1,
                turn_id=-1,
                query=qa.question,
                retrieved_pairs=retrieved,
                store_size_at_retrieval=store_size,
            )
            write_retrieval_log(retrieval_log_paths[i], log_entry)

            # Retrieved memories metadata for result dict
            retrieved_metadata = [
                {
                    "session_id": mem.session_id,
                    "conv_id":    mem.conv_id,
                    "turn_id":    mem.turn_id,
                }
                for mem, _ in retrieved
            ]

            # Format context and build prompt
            context_str = format_retrieved_memories(retrieved)
            prompt_content = prompt_template.format(
                context=context_str,
                question=qa.question,
            )
            qa_jobs.append((i, qa, retrieved_metadata, prompt_content))

        # ── Batch generate in QA_BATCH_SIZE chunks ────────────────────
        all_pairs = []  # (qa_job, result_dict, usage_dict, fallback_flag)

        for chunk_start in tqdm(range(0, len(qa_jobs), cfg.QA_BATCH_SIZE), desc="Phase2 QA chunks"):
            chunk = qa_jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            chunk_prompts = [job[3] for job in chunk]

            chunk_results, chunk_usages, chunk_fallbacks = self._batch_generate_with_retry(
                chunk_prompts, schema
            )

            for job, result, usage, fallback_used in zip(
                chunk, chunk_results, chunk_usages, chunk_fallbacks
            ):
                all_pairs.append((job, result, usage, fallback_used))

        # ── Distribute results ────────────────────────────────────────
        qa_results_per_session: List[List[Dict]] = [[] for _ in sessions]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0, "qa_parse_fallback_count": 0}
            for _ in sessions
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
            if self.subset == "supportive":
                label = answer.strip().lower()
                answer = label if label in ("yes", "no") else "unknown"

            qa_results_per_session[i].append({
                "question":            qa.question,
                "generated_answer":    answer,
                "ground_truth_answer": qa.answer,
                "retrieved_memories":  retrieved_metadata,
                "qa_tokens":           qa_tokens,
            })

            phase2_stats[i]["qa_input"]               += qa_tokens["input"]
            phase2_stats[i]["qa_output"]              += qa_tokens["output"]
            phase2_stats[i]["num_qa_calls"]           += 1
            phase2_stats[i]["qa_parse_fallback_count"] += int(fallback_used)

        return qa_results_per_session, phase2_stats

    # ------------------------------------------------------------------
    # Batch generate with per-item JSON retry fallback
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        schema: Dict,
    ) -> Tuple[List[Dict], List[Dict], List[bool]]:
        """Batch generate QA answers and retry bad JSON items sequentially.

        Returns:
            (parsed_results, usages, fallback_flags)
            - parsed_results: list of dicts (one per prompt)
            - usages: list of {prompt_tokens, completion_tokens}
            - fallback_flags: list of bool (True if sequential retry was used)
        """
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
    # ── Step 1: load config ───────────────────────────────────────────
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config")
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

    # ── Step 2: parse args ────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Dense Retrieval QA-Only Baseline Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-session", type=int, required=True,
                        help="First session index (inclusive)")
    parser.add_argument("--end-session", type=int, required=True,
                        help="Last session index (inclusive)")
    parser.add_argument("--subset", type=str, required=True,
                        choices=["opposed", "supportive"],
                        help="Dataset subset to use")
    parser.add_argument("--model", type=str,
                        default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Model max context length (passed to vLLM if provided).")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE,
                        help=f"Sessions to process in parallel (default: {cfg.BATCH_SIZE})")
    parser.add_argument("--config", type=str, default="config",
                        help="Config file name (without .py).")
    parser.add_argument("--engine", type=str, default=None,
                        help="Override LLM engine (vllm, together, openai).")
    args = parser.parse_args()

    # Override engine if provided
    if args.engine is not None:
        cfg.LLM_ENGINE = args.engine

    # ── Step 3: validate ──────────────────────────────────────────────
    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    non_vllm_engine = cfg.LLM_ENGINE != "vllm"

    # ── Step 4: setup directories and logging ─────────────────────────
    cfg.ensure_directories(args.model, args.subset, args.start_session, args.end_session, config_name)

    session_dir = cfg.get_session_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    if non_vllm_engine:
        logger.warning(
            f"LLM engine is '{cfg.LLM_ENGINE}'. "
            "Batch generate_batch_raw() requires vllm; falling back to sequential generation "
            "is not implemented. Ensure engine=vllm for optimal throughput."
        )

    results_file      = cfg.get_results_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    checkpoint_file   = cfg.get_checkpoint_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    snapshots_dir     = cfg.get_memory_snapshots_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    prompt_log_dir    = cfg.get_prompt_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)

    logger.info("=" * 60)
    logger.info("Dense Retrieval Batch QA-Only Baseline Experiment")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Subset          : {args.subset}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Retrieve K      : {cfg.RETRIEVE_K}")
    logger.info(f"  Save snapshots  : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call log    : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    # ── Step 5: load dataset ──────────────────────────────────────────
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
    target_session_ids = [s.session_id for s in target_sessions]

    # ── Step 6: checkpoint resume ─────────────────────────────────────
    completed_ids: Set[int] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_ids = load_checkpoint(checkpoint_file, target_session_ids)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} sessions already completed")

    pending_sessions = [s for s in target_sessions if s.session_id not in completed_ids]
    if not pending_sessions:
        logger.info("All sessions already completed.")
        return 0

    # ── Step 7: load embedding model ──────────────────────────────────
    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    # ── Step 8: init LLM ──────────────────────────────────────────────
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    runner = BatchedDenseRunner(
        llm_client=llm_client,
        subset=args.subset,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
    )

    config_metadata = {
        "config_name":    config_name,
        "model":          args.model,
        "subset":         args.subset,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "retrieve_k":     cfg.RETRIEVE_K,
        "temperature":    cfg.TEMPERATURE,
        "max_tokens":     cfg.MAX_TOKENS,
        "session_range":  [args.start_session, args.end_session],
    }

    results = load_existing_results(results_file)

    print(f"\n{'#'*60}")
    print(f"# Dense Retrieval Batch QA-Only  |  subset={args.subset}")
    print(f"# Model  : {cfg.extract_model_name(args.model)}")
    print(f"# Embedding : {cfg.EMBEDDING_MODEL}  |  K={cfg.RETRIEVE_K}")
    print(f"# Batch size : {args.batch_size}")
    print(f"# Sessions [{args.start_session}, {args.end_session}]  ({len(pending_sessions)} to process)")
    print(f"{'#'*60}\n")

    # ── Step 9: main loop ─────────────────────────────────────────────
    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_sessions[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        session_ids = [s.session_id for s in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: sessions {session_ids}")

        batch_retrieval_log_paths = [
            retrieval_log_dir / f"session_{s.session_id}_retrieval_log.jsonl"
            for s in batch
        ]
        batch_prompt_log_dirs = [
            prompt_log_dir / f"session_{s.session_id}"
            for s in batch
        ]

        try:
            batch_results = runner.run_batch(
                sessions=batch,
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
                completed_ids.add(result["session_id"])
                save_checkpoint(
                    checkpoint_file, completed_ids,
                    args.model, args.subset,
                    args.start_session, args.end_session,
                    config_name=config_name,
                )

            logger.info(
                f"Session {result['session_id']} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total LLM calls: {result['token_statistics']['total_llm_calls']}"
            )

    print(f"\n{'#'*60}")
    print(f"# Batch experiment complete!  Results -> {results_file}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
