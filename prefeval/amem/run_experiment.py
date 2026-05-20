"""
A-MEM Experiment Runner — PrefEval (single-chain, cumulative-checkpoint).

Flow per invocation
-------------------
Args: --end-session K, --model, --tensor-parallel, --gpu-memory, --max-model-len, --config

1. Resolve output dir; scan memory_snapshots/m_*/ for max k_existing.
2. If K <= k_existing → no-op, exit.
3. Init shared embedding model + LLM client; create BaseAgent.
   If k_existing >= 0: load_snapshot(m_{k_existing}); start_k = k_existing+1.
   Else:               start_k = 0.
4. Load PrefEval samples [0..K] as a chain of Sessions.
5. For k = start_k .. K:
     a. Ingest session k turn-by-turn (sequential add_memory), drain tokens.
     b. Run QA at checkpoint k: batch all q_0..q_k together (chunked by QA_BATCH_SIZE).
     c. Append rows to results.jsonl + retrieval_log.jsonl.
     d. Save snapshot m_k (after QA — so snapshot existence ⇔ QA done).
     e. Update cumulative stats.json + meta.json (first run only).

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

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore

from load_dataset import load_prefeval_chain, Session, Turn, QAPair


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"amem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    """Return the largest k for which snapshots_dir/m_k/ exists. -1 if none."""
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
        "call_2_note_construction": {"input": 0, "output": 0, "llm_calls": 0,
                                     "parse_fallback_count": 0},
        "call_3_evolution":         {"input": 0, "output": 0, "llm_calls": 0},
        "call_4_qa":                {"input": 0, "output": 0, "llm_calls": 0},
        "evolution": {
            "evo_triggered_count": 0,
            "actions_taken": {"strengthen": 0, "update_neighbor": 0},
        },
        "checkpoints_completed": [],
    }


def load_stats(stats_file: Path) -> Dict:
    if not stats_file.exists():
        return _empty_stats()
    try:
        with open(stats_file) as f:
            data = json.load(f)
        # Backfill missing keys for safety
        empty = _empty_stats()
        for k, v in empty.items():
            if k not in data:
                data[k] = v
        return data
    except Exception as e:
        logger.warning(f"Could not load stats.json ({e}); starting fresh")
        return _empty_stats()


def merge_stats(stats: Dict, mem_tokens: Dict, qa_tokens: Dict,
                internal_stats: Dict, k: int) -> None:
    """Mutate `stats` in place by adding the deltas from one checkpoint."""
    for ct in ("call_2_note_construction", "call_3_evolution"):
        stats[ct]["input"]     += mem_tokens.get(ct, {}).get("input", 0)
        stats[ct]["output"]    += mem_tokens.get(ct, {}).get("output", 0)
        stats[ct]["llm_calls"] += mem_tokens.get(ct, {}).get("llm_calls", 0)
    stats["call_2_note_construction"]["parse_fallback_count"] += \
        internal_stats.get("note_parse_fallback_count", 0)

    stats["call_4_qa"]["input"]     += qa_tokens["input"]
    stats["call_4_qa"]["output"]    += qa_tokens["output"]
    stats["call_4_qa"]["llm_calls"] += qa_tokens["llm_calls"]

    stats["evolution"]["evo_triggered_count"] += internal_stats.get("evo_triggered_count", 0)
    actions = internal_stats.get("actions_taken", {})
    for a in ("strengthen", "update_neighbor"):
        stats["evolution"]["actions_taken"][a] += actions.get(a, 0)

    if k not in stats["checkpoints_completed"]:
        stats["checkpoints_completed"].append(k)
        stats["checkpoints_completed"].sort()


# =============================================================================
# LLM CALL LOGGING
# =============================================================================

class LLMCallLogger:
    CALL_DIRS = [
        "call_2_note_construction",
        "call_3_evolution",
        "call_4_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "call_type":     call_type,
            "system_prompt": system_prompt,
            "user_prompt":   user_prompt,
            "output":        output if not isinstance(output, dict) else {
                kk: vv for kk, vv in output.items() if kk != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# CHAIN RUNNER
# =============================================================================

class ChainRunner:
    """Runs the cumulative-chain pipeline against a single BaseAgent."""

    QA_SYSTEM_PROMPT = (
        "You are a helpful assistant answering a question about a user "
        "based on their conversation history stored in memory. "
        "Respond in JSON format with an 'answer' field."
    )

    def __init__(self, agent, llm_client, model_path: str):
        self.agent = agent
        self.llm_client = llm_client
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Phase 1 — sequential ingestion of one session
    # ------------------------------------------------------------------

    def ingest_session(self, session: Session) -> None:
        """Ingest all turns of a session via the sequential add_note path."""
        for turn in tqdm(session.turns, desc=f"Ingest k={session.conv_id}", leave=False):
            ts = f"0000_{turn.conv_id:04d}_{turn.turn_id:04d}"
            self.agent.add_memory(turn.to_memory_content(), time=ts)

    # ------------------------------------------------------------------
    # Phase 2 — batched QA at checkpoint k
    # ------------------------------------------------------------------

    def run_checkpoint_qa(
        self,
        sessions_so_far: List[Session],
        k: int,
        retrieval_log_path: Path,
    ) -> Tuple[List[Dict], Dict]:
        """
        At checkpoint k, evaluate q_0..q_k against the current memory state.

        Returns: (rows_for_results_jsonl, qa_token_totals)
        """
        from agent import QA_PROMPT, QA_SCHEMA

        # Build retrieval + prompts for each q_j (j = 0..k)
        jobs = []  # (j, qa, prompt, retrieved_metadata)
        for j, sess in enumerate(sessions_so_far):
            qa = sess.qa[0]

            retrieved_str, retrieved_metadata = self.agent.retrieve_memory_with_metadata(qa.question)

            log_items, num_linked = self.agent.retrieve_for_log(qa.question)
            _append_jsonl(retrieval_log_path, {
                "timestamp":         datetime.now().isoformat(),
                "phase":             "qa",
                "k":                 k,
                "question_session":  j,
                "query":             qa.question,
                "memory_type":       ["direct", "linked"],
                "num_retrieved":     [len(log_items), num_linked],
                "retrieved_items":   log_items,
                "retrieval_scores":  [it["score"] for it in log_items],
                "module_specific":   {
                    "module": "amem",
                    "num_direct_hits":      len(log_items),
                    "num_linked_neighbors": num_linked,
                },
            })

            memory_str = retrieved_str if retrieved_str else "No memory available."
            prompt = QA_PROMPT.format(retrieved_memory=memory_str, question=qa.question)
            jobs.append((j, qa, prompt, retrieved_metadata))

        # Batch generate in chunks
        rows: List[Dict] = []
        qa_totals = {"input": 0, "output": 0, "llm_calls": 0}

        for chunk_start in range(0, len(jobs), cfg.QA_BATCH_SIZE):
            chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            chunk_prompts = [c[2] for c in chunk]

            results, usages = self._batch_generate_with_retry(
                prompts=chunk_prompts,
                system_prompt=self.QA_SYSTEM_PROMPT,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=QA_SCHEMA,
            )

            for (j, qa, prompt, retrieved_metadata), result, usage in zip(chunk, results, usages):
                # Per-call prompt log
                if self.agent._llm_logger is not None:
                    self.agent._llm_logger.log("call_4_qa", "", prompt, result)

                qa_totals["input"]     += usage["prompt_tokens"]
                qa_totals["output"]    += usage["completion_tokens"]
                qa_totals["llm_calls"] += 1

                answer = result.get("answer", "") if isinstance(result, dict) else ""
                rows.append({
                    "k":                  k,
                    "question_session":   j,
                    "question":           qa.question,
                    "model_answer":       answer,
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
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        guided_json=None,
    ) -> Tuple[List, List[Dict]]:
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
            logger.warning(f"Batch item {idx} failed JSON parse — retrying sequentially")
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
    elif engine == "together":
        return _create(engine="together", **cfg.TOGETHER_CONFIG)
    elif engine == "openai":
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
        description="A-MEM PrefEval Chain Experiment",
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
    logger.info("A-MEM PrefEval Chain")
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

    # ---- Step 5: init embedding + LLM + agent ----
    logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )

    from agent import BaseAgent
    agent = BaseAgent(llm_client, args.model, embedding_model=shared_embedding_model)
    if cfg.ENABLE_LLM_CALL_LOGGING:
        agent.set_llm_logger(LLMCallLogger(prompt_log_dir))

    # Resume: load last snapshot
    if k_existing >= 0:
        snap_dir = snapshots_dir / f"m_{k_existing}"
        logger.info(f"Loading snapshot {snap_dir}")
        agent.load_memory_snapshot(snap_dir)
        # Drain any stray counters before main loop (load_snapshot doesn't touch them,
        # but a fresh agent has zero counters so this is just defensive)
        agent.get_and_reset_memory_tokens()
        agent.get_and_reset_internal_stats()
        logger.info(f"Resumed with {agent.get_memory_count()} memories in state")

    # ---- Step 6: meta.json on first run ----
    if not meta_file.exists():
        config_metadata = {
            "config_name":         config_name,
            "model":               args.model,
            "embedding_model":     cfg.EMBEDDING_MODEL,
            "retrieve_k":          cfg.RETRIEVE_K,
            "evolution_threshold": cfg.EVOLUTION_THRESHOLD,
            "temperature":         cfg.TEMPERATURE,
            "max_tokens":          cfg.MAX_TOKENS,
            "conv_ids_per_day":    cfg.CONV_IDS_PER_DAY,
            "minutes_per_turn":    cfg.MINUTES_PER_TURN,
            "first_run_at":        datetime.now().isoformat(),
        }
        _atomic_write_json(meta_file, config_metadata)

    # ---- Step 7: stats.json (cumulative) ----
    stats = load_stats(stats_file)

    # ---- Step 8: chain loop ----
    runner = ChainRunner(agent, llm_client, args.model)

    print(f"\n{'#'*60}")
    print(f"# A-MEM PrefEval Chain  |  model={cfg.extract_model_name(args.model)}")
    print(f"# Sessions {start_k}..{K}  ({K - start_k + 1} to ingest)")
    print(f"{'#'*60}\n")

    for k in tqdm(range(start_k, K + 1), desc="Chain checkpoints"):
        session = sessions[k]
        logger.info(f"--- checkpoint k={k}: ingesting session (conv_id={session.conv_id}, "
                    f"{session.total_turns} turns) ---")

        # (a) Ingest session k
        runner.ingest_session(session)
        mem_tokens     = agent.get_and_reset_memory_tokens()
        internal_stats = agent.get_and_reset_internal_stats()

        # (b) QA at checkpoint k (q_0..q_k)
        logger.info(f"k={k}: running QA over {k + 1} questions")
        rows, qa_totals = runner.run_checkpoint_qa(
            sessions_so_far=sessions[:k + 1],
            k=k,
            retrieval_log_path=retrieval_log,
        )

        # (c) Append rows
        for row in rows:
            _append_jsonl(results_file, row)

        # (d) Save snapshot AFTER QA succeeds (snapshot existence ⇔ QA done)
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snap_dir = snapshots_dir / f"m_{k}"
            agent.save_memory_snapshot(snap_dir)
            logger.info(f"Saved snapshot {snap_dir}")

        # (e) Update cumulative stats
        merge_stats(stats, mem_tokens, qa_totals, internal_stats, k)
        _atomic_write_json(stats_file, stats)

        logger.info(f"k={k} done. mem={agent.get_memory_count()}, "
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
