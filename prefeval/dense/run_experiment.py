"""
Dense Retrieval Experiment Runner — PrefEval (single-chain, cumulative-checkpoint).

Flow per invocation
-------------------
Args: --end-session K, --model, --tensor-parallel, --gpu-memory, --max-model-len, --config

1. Resolve output dir; scan memory_snapshots/m_*/ for max k_existing.
2. If K <= k_existing → no-op, exit.
3. Init shared embedding model + LLM client; create DenseMemoryStore.
   If k_existing >= 0:
     - dense_store.load_snapshot(m_{k_existing})
     - start_k = k_existing + 1
   Else:
     - start_k = 0
4. Load PrefEval samples [0..K] as a chain of Sessions.
5. For k = start_k .. K:
     a. Build pair contents for all turns of session k (user+assistant text).
        Batch encode → store all pairs.
     b. Run batched QA over q_0..q_k:
        - Batch encode (k+1) questions
        - For each j: retrieve top-RETRIEVE_K, build prompt
        - Batch generate (chunked by QA_BATCH_SIZE)
     c. Save snapshot m_k (after QA succeeds).
     d. Update cumulative stats.json.

Q1 (b) policy: NO Phase 1 retrieval / no "prompt_construction" log entries.
Phase 1 is embed + store only.

Resume invariant: snapshot existence implies QA at that checkpoint completed.
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
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore

from load_dataset import load_prefeval_chain, Session, Turn, QAPair
from dense_store  import DenseMemoryStore, MemoryUnit


# =============================================================================
# QA PROMPT (200 words; opposed-style only)
# =============================================================================

QA_PROMPT = """\
You are a helpful assistant. Answer the question based only on the retrieved memories provided. Be concise (max 200 words).

Retrieved memories:
{context}

Question: {question}

Answer in JSON format:
{{"answer": "<your answer here>"}}"""

QA_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


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
# I/O HELPERS
# =============================================================================

def _atomic_write_json(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _append_jsonl(path: Path, entry: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


_M_K_RE = re.compile(r"^m_(\d+)$")


def find_max_existing_k(snapshots_dir: Path) -> int:
    if not snapshots_dir.exists():
        return -1
    max_k = -1
    for p in snapshots_dir.iterdir():
        if not p.is_dir():
            continue
        m = _M_K_RE.match(p.name)
        if m:
            k = int(m.group(1))
            if (p / "memories.json").exists():
                max_k = max(max_k, k)
    return max_k


# =============================================================================
# CUMULATIVE STATS
# =============================================================================

def _empty_stats() -> Dict:
    return {
        "call_1_qa": {"input": 0, "output": 0, "llm_calls": 0},
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


def merge_stats(stats: Dict, qa_tokens: Dict, k: int) -> None:
    stats["call_1_qa"]["input"]     += qa_tokens["input"]
    stats["call_1_qa"]["output"]    += qa_tokens["output"]
    stats["call_1_qa"]["llm_calls"] += qa_tokens["llm_calls"]

    if k not in stats["checkpoints_completed"]:
        stats["checkpoints_completed"].append(k)
        stats["checkpoints_completed"].sort()


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
# RETRIEVAL LOG / PROMPT FORMATTING
# =============================================================================

def _format_pair_content(user_turn: Turn, assistant_turn: Optional[Turn]) -> str:
    if assistant_turn is not None:
        return f"User: {user_turn.utterance}\nAssistant: {assistant_turn.utterance}"
    return f"User: {user_turn.utterance}"


def format_retrieved_memories(retrieved_pairs: List[Tuple[MemoryUnit, float]]) -> str:
    if not retrieved_pairs:
        return "No relevant memories found."
    parts = [f"[Memory {idx}] {mem.content}"
             for idx, (mem, _) in enumerate(retrieved_pairs, start=1)]
    return "\n\n".join(parts)


def build_retrieval_log_entry(
    k: int,
    question_session: int,
    query: str,
    retrieved_pairs: List[Tuple[MemoryUnit, float]],
) -> Dict:
    items = [
        {
            "memory_id":      f"s{m.session_id}-c{m.conv_id}-t{m.turn_id}",
            "content_preview": m.content[:120],
            "score":          float(score),
            "source_turn": {
                "session_id": m.session_id,
                "conv_id":    m.conv_id,
                "turn_id":    m.turn_id,
            },
        }
        for (m, score) in retrieved_pairs
    ]
    return {
        "timestamp":         datetime.now().isoformat(),
        "phase":             "qa",
        "k":                 k,
        "question_session":  question_session,
        "query":             query,
        "retrieved_items":   items,
        "retrieval_scores":  [it["score"] for it in items],
        "module_specific": {
            "module":         "dense",
            "num_retrieved":  len(items),
        },
    }


# =============================================================================
# CHAIN RUNNER
# =============================================================================

class ChainRunner:
    """Cumulative-chain pipeline against a single DenseMemoryStore."""

    def __init__(self, store: DenseMemoryStore, embedding_model, llm_client,
                 model_path: str, llm_logger: Optional[LLMCallLogger] = None):
        self.store = store
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.model_path = model_path
        self.llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Phase 1 — embed + store all turns of one session (Q1 = b: no retrieval)
    # ------------------------------------------------------------------

    def ingest_session(self, session: Session) -> None:
        turn_pairs = session.get_turn_pairs()
        if not turn_pairs:
            return

        contents: List[str] = []
        meta: List[Tuple[int, int, int]] = []  # (session_id, conv_id, turn_id)
        for user_turn, assistant_turn in turn_pairs:
            contents.append(_format_pair_content(user_turn, assistant_turn))
            meta.append((user_turn.session_id, user_turn.conv_id, user_turn.turn_id))

        # Batch encode all pair contents at once (SentenceTransformer handles internal batching)
        embeddings = self.embedding_model.encode(
            contents,
            batch_size=cfg.ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for content, emb, (sid, cid, tid) in zip(contents, embeddings, meta):
            self.store.store(
                content=content,
                embedding=emb.astype(np.float32),
                session_id=sid,
                conv_id=cid,
                turn_id=tid,
            )

    # ------------------------------------------------------------------
    # Phase 2 — batched QA at checkpoint k
    # ------------------------------------------------------------------

    def run_checkpoint_qa(
        self,
        sessions_so_far: List[Session],
        k: int,
        retrieval_log_path: Path,
    ) -> Tuple[List[Dict], Dict]:
        """At checkpoint k, evaluate q_0..q_k against the current store."""
        qas = [sess.qa[0] for sess in sessions_so_far]
        questions = [qa.question for qa in qas]

        # Batch encode all (k+1) questions
        question_embeddings = self.embedding_model.encode(
            questions,
            batch_size=cfg.ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Per-question retrieval + prompt build + retrieval log
        jobs = []  # (j, qa, prompt, retrieved_metadata)
        for j, (qa, q_emb) in enumerate(zip(qas, question_embeddings)):
            retrieved = self.store.retrieve(q_emb.astype(np.float32), k=cfg.RETRIEVE_K)
            _append_jsonl(retrieval_log_path, build_retrieval_log_entry(
                k=k,
                question_session=j,
                query=qa.question,
                retrieved_pairs=retrieved,
            ))
            context = format_retrieved_memories(retrieved)
            prompt = QA_PROMPT.format(context=context, question=qa.question)
            retrieved_metadata = [
                {
                    "session_id": mem.session_id,
                    "conv_id":    mem.conv_id,
                    "turn_id":    mem.turn_id,
                    "score":      float(score),
                }
                for (mem, score) in retrieved
            ]
            jobs.append((j, qa, prompt, retrieved_metadata))

        # Batch generate
        rows: List[Dict] = []
        qa_totals = {"input": 0, "output": 0, "llm_calls": 0}

        for chunk_start in range(0, len(jobs), cfg.QA_BATCH_SIZE):
            chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            chunk_prompts = [c[2] for c in chunk]

            results, usages = self._batch_generate_with_retry(
                prompts=chunk_prompts,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=QA_SCHEMA,
            )

            for (j, qa, prompt, retrieved_metadata), result, usage in zip(chunk, results, usages):
                if self.llm_logger is not None:
                    self.llm_logger.log("call_1_qa", "", prompt, result)

                qa_totals["input"]     += usage["prompt_tokens"]
                qa_totals["output"]    += usage["completion_tokens"]
                qa_totals["llm_calls"] += 1

                answer = result.get("answer", "") if isinstance(result, dict) else ""
                rows.append({
                    "k":                  k,
                    "question_session":   j,
                    "question":           qa.question,
                    "model_answer":       str(answer).strip(),
                    "topic":              qa.topic,
                    "persona":            qa.persona,
                    "preference":         qa.preference,
                    "explanation":        qa.explanation,
                    "retrieved_memories": retrieved_metadata,
                    "qa_tokens": {
                        "input":  usage["prompt_tokens"],
                        "output": usage["completion_tokens"],
                        "model":  self.model_path,
                    },
                })

        return rows, qa_totals

    # ------------------------------------------------------------------
    # Batched generate with per-item JSON retry fallback
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        guided_json=None,
    ) -> Tuple[List, List[Dict]]:
        if not prompts:
            return [], []

        texts, usages = self.llm_client.generate_batch_raw(
            prompts=prompts,
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
            logger.warning(f"Batch item {idx} failed JSON parse — retrying sequentially")
            try:
                retry_result = self.llm_client.generate(
                    prompt=prompts[idx],
                    guided_json=guided_json,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                if isinstance(retry_result, dict):
                    usage_info = retry_result.pop("_usage", {})
                    usages[idx] = {
                        "prompt_tokens":     usage_info.get("prompt_tokens", 0),
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
    # ---- Step 1: load config ----
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

    # ---- Step 2: parse args ----
    parser = argparse.ArgumentParser(
        description="Dense Retrieval PrefEval Chain Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--end-session", type=int, required=True,
                        help="Last sample index in chain (chain = samples[0..K])")
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.end_session < 0:
        parser.error("--end-session must be >= 0")

    cfg.ensure_directories(args.model, config_name)

    snapshots_dir   = cfg.get_memory_snapshots_dir(args.model, config_name)
    results_file    = cfg.get_results_file(args.model, config_name)
    retrieval_log   = cfg.get_retrieval_log_file(args.model, config_name)
    stats_file      = cfg.get_stats_file(args.model, config_name)
    meta_file       = cfg.get_meta_file(args.model, config_name)
    prompt_log_dir  = cfg.get_prompt_log_dir(args.model, config_name)
    run_log_dir     = cfg.get_run_log_dir(args.model, config_name)

    global logger
    logger = setup_logging(run_log_dir)

    # ---- Step 3: resume detection ----
    k_existing = find_max_existing_k(snapshots_dir)
    K = args.end_session

    logger.info("=" * 60)
    logger.info("Dense Retrieval PrefEval Chain")
    logger.info(f"  Config       : {config_name}")
    logger.info(f"  Model        : {args.model}")
    logger.info(f"  end_session  : {K}")
    logger.info(f"  k_existing   : {k_existing}")
    logger.info(f"  Output dir   : {cfg.get_output_dir(args.model, config_name)}")
    logger.info("=" * 60)

    if K <= k_existing:
        logger.info(f"Already complete up to k={k_existing} (>= K={K}). Nothing to do.")
        return 0

    start_k = k_existing + 1

    # ---- Step 4: load chain ----
    sessions = load_prefeval_chain(cfg.DATASET_PATH, end_session=K)

    # ---- Step 5: init embedding + LLM + store ----
    logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )

    store = DenseMemoryStore(embedding_model=shared_embedding_model, k=cfg.RETRIEVE_K)
    llm_logger = LLMCallLogger(prompt_log_dir) if cfg.ENABLE_LLM_CALL_LOGGING else None

    # Resume: load snapshot
    if k_existing >= 0:
        snap_dir = snapshots_dir / f"m_{k_existing}"
        logger.info(f"Loading snapshot {snap_dir}")
        store.load_snapshot(snap_dir)
        logger.info(f"Resumed: {len(store._memories)} memories loaded")

    # ---- Step 6: meta.json on first run ----
    if not meta_file.exists():
        config_metadata = {
            "config_name":        config_name,
            "model":              args.model,
            "embedding_model":    cfg.EMBEDDING_MODEL,
            "retrieve_k":         cfg.RETRIEVE_K,
            "temperature":        cfg.TEMPERATURE,
            "max_tokens":         cfg.MAX_TOKENS,
            "json_retry":         cfg.JSON_RETRY,
            "encode_batch_size":  cfg.ENCODE_BATCH_SIZE,
            "first_run_at":       datetime.now().isoformat(),
        }
        _atomic_write_json(meta_file, config_metadata)

    # ---- Step 7: stats.json (cumulative) ----
    stats = load_stats(stats_file)

    # ---- Step 8: chain loop ----
    runner = ChainRunner(
        store=store,
        embedding_model=shared_embedding_model,
        llm_client=llm_client,
        model_path=args.model,
        llm_logger=llm_logger,
    )

    print(f"\n{'#'*60}")
    print(f"# Dense PrefEval Chain  |  model={cfg.extract_model_name(args.model)}")
    print(f"# Sessions {start_k}..{K}  ({K - start_k + 1} to ingest)")
    print(f"{'#'*60}\n")

    for k in tqdm(range(start_k, K + 1), desc="Chain checkpoints"):
        session = sessions[k]
        logger.info(f"--- checkpoint k={k}: ingesting (conv_id={session.conv_id}, "
                    f"{session.total_turns} turns) ---")

        # (a) Embed + store all turns of session k
        runner.ingest_session(session)

        # (b) Batched QA over q_0..q_k
        logger.info(f"k={k}: running QA over {k + 1} questions "
                    f"(store size={len(store._memories)})")
        rows, qa_totals = runner.run_checkpoint_qa(
            sessions_so_far=sessions[:k + 1],
            k=k,
            retrieval_log_path=retrieval_log,
        )

        # Append rows
        for row in rows:
            _append_jsonl(results_file, row)

        # (c) Save snapshot AFTER QA succeeds
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snap_dir = snapshots_dir / f"m_{k}"
            store.save_snapshot(snap_dir)
            logger.info(f"Saved snapshot {snap_dir}")

        # (d) Update cumulative stats
        merge_stats(stats, qa_totals, k)
        _atomic_write_json(stats_file, stats)

        logger.info(
            f"k={k} done. mem={len(store._memories)}, "
            f"qa_in={qa_totals['input']}, qa_out={qa_totals['output']}, "
            f"qa_calls={qa_totals['llm_calls']}"
        )

    print(f"\n{'#'*60}")
    print(f"# Chain complete!")
    print(f"# Results -> {results_file}")
    print(f"# Snapshots -> {snapshots_dir}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
