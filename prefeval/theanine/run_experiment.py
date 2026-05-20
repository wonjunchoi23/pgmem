"""
Theanine Experiment Runner — PrefEval (single-chain, cumulative-checkpoint).

Flow per invocation
-------------------
Args: --end-session K, --model, --tensor-parallel, --gpu-memory, --max-model-len, --config

1. Resolve output dir; scan memory_snapshots/m_*/ for max k_existing.
2. If K <= k_existing → no-op, exit.
3. Init shared embedding model + LLM client; create TheanineModule.
   If k_existing >= 0:
     - load_memory_snapshot(m_{k_existing})
     - reconstruct runner-side state from sessions[0..k_existing]:
         current_conv_id, current_day, current_dialogue, finalize_turns, pending_count
     - start_k = k_existing + 1
   Else:
     - start_k = 0; runner state = empty/initial
4. Load PrefEval samples [0..K] as a chain of Sessions.
5. For k = start_k .. K:
     a. Boundary check: if k > 0, pending_count += 1; if pending >= FINALIZE_EVERY_N_CONVS
        and finalize_turns non-empty → run natural finalize for [finalize_turns]
        with conv_id=current_conv_id (the LAST conv before crossing).
     b. Day boundary: if k // CONV_IDS_PER_DAY != current_day → reset current_dialogue.
     c. Ingest session k turns (no LLM calls; just accumulate dialogue + finalize_turns).
     d. NO force-flush at checkpoint (Q1=b: natural-boundary policy).
     e. Run QA over q_0..q_k:
         - retrieve timeline paths for each question
         - batch refine prompts across all (k+1) questions
         - batch QA prompts across all (k+1) questions
     f. Save snapshot m_k (after QA succeeds).
     g. Update cumulative stats.json.

Resume invariant: snapshot existence implies QA at that checkpoint completed.

Q1 (b) policy implications:
- At checkpoint k, the graph contains nodes only from natural finalize batches
  triggered during ingestion. Sessions whose turns are still pending in
  finalize_turns are NOT in the graph — they show up only in current_dialogue
  (current virtual day).
- For default CONV_IDS_PER_DAY=2 / FINALIZE_EVERY_N_CONVS=2:
    k=0,1: graph empty
    k=2,3: graph has [s0+s1]
    k=4,5: graph has [s0+s1, s2+s3]
    ...
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
        log_file = log_dir / f"theanine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            if (p / "nodes.json").exists():
                max_k = max(max_k, k)
    return max_k


# =============================================================================
# CUMULATIVE STATS
# =============================================================================

def _empty_stats() -> Dict:
    return {
        "call_2_refinement":    {"input": 0, "output": 0, "llm_calls": 0},
        "call_3_summarization": {"input": 0, "output": 0, "llm_calls": 0,
                                 "parse_fallback_count": 0},
        "call_4_relation":      {"input": 0, "output": 0, "llm_calls": 0},
        "call_5_qa":            {"input": 0, "output": 0, "llm_calls": 0},
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


def merge_stats(stats: Dict, mem_tokens: Dict, refine_tokens: Dict,
                qa_tokens: Dict, k: int) -> None:
    """Add deltas from one checkpoint into cumulative stats."""
    summ_b = mem_tokens.get("call_3_summarization", {})
    rel_b  = mem_tokens.get("call_4_relation", {})
    stats["call_3_summarization"]["input"]                += summ_b.get("input", 0)
    stats["call_3_summarization"]["output"]               += summ_b.get("output", 0)
    stats["call_3_summarization"]["llm_calls"]            += summ_b.get("llm_calls", 0)
    stats["call_3_summarization"]["parse_fallback_count"] += summ_b.get("parse_fallback_count", 0)

    stats["call_4_relation"]["input"]     += rel_b.get("input", 0)
    stats["call_4_relation"]["output"]    += rel_b.get("output", 0)
    stats["call_4_relation"]["llm_calls"] += rel_b.get("llm_calls", 0)

    stats["call_2_refinement"]["input"]     += refine_tokens["input"]
    stats["call_2_refinement"]["output"]    += refine_tokens["output"]
    stats["call_2_refinement"]["llm_calls"] += refine_tokens["llm_calls"]

    stats["call_5_qa"]["input"]     += qa_tokens["input"]
    stats["call_5_qa"]["output"]    += qa_tokens["output"]
    stats["call_5_qa"]["llm_calls"] += qa_tokens["llm_calls"]

    if k not in stats["checkpoints_completed"]:
        stats["checkpoints_completed"].append(k)
        stats["checkpoints_completed"].sort()


# =============================================================================
# DIALOGUE FORMATTING
# =============================================================================

def _format_finalize_dialogue(turns: List[Turn]) -> str:
    return "\n".join(t.to_message() for t in turns)


def _format_session_dialogue(turn_pairs) -> str:
    lines = []
    for user_turn, assistant_turn in turn_pairs:
        if user_turn:
            lines.append(user_turn.to_message())
        if assistant_turn:
            lines.append(assistant_turn.to_message())
    return "\n".join(lines)


# =============================================================================
# RETRIEVAL LOG
# =============================================================================

def build_retrieval_log_entry(
    k: int,
    question_session: int,
    query: str,
    retrieved_items: List[Dict],
    use_timelines: List,
    timeline_info: List,
    current_dialogue_turns: int,
) -> Dict:
    return {
        "timestamp":         datetime.now().isoformat(),
        "phase":             "qa",
        "k":                 k,
        "question_session":  question_session,
        "query":             query,
        "retrieved_items":   retrieved_items,
        "retrieval_scores":  [it.get("score", 0.0) for it in retrieved_items],
        "module_specific": {
            "module":                  "theanine",
            "use_timelines":           use_timelines,
            "timeline_info":           timeline_info,
            "current_dialogue_turns":  current_dialogue_turns,
        },
    }


# =============================================================================
# CHAIN RUNNER
# =============================================================================

class ChainRunner:
    """Cumulative-chain pipeline against a single TheanineModule."""

    def __init__(self, module, llm_client, model_path: str):
        self.module = module
        self.llm_client = llm_client
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Phase 2 — batched QA at checkpoint k
    # ------------------------------------------------------------------

    def run_checkpoint_qa(
        self,
        sessions_so_far: List[Session],
        k: int,
        current_dialogue: str,
        retrieval_log_path: Path,
    ) -> Tuple[List[Dict], Dict, Dict]:
        """
        At checkpoint k, evaluate q_0..q_k against current memory state.

        Pipeline:
          1. retrieve timelines for each q_j
          2. build refine prompts (RETRIEVE_TOP_K * TIMELINE_SAMPLE_N per QA)
          3. batch-generate refine results
          4. build QA prompts using refined memory + current_dialogue
          5. batch-generate QA results

        Returns: (rows, refine_token_totals, qa_token_totals)
        """
        from generator import QA_SCHEMA
        from timeline import _REFINEMENT_GUIDED_JSON

        qa_dialogue = current_dialogue.strip()
        dialogue_turn_count = sum(1 for line in qa_dialogue.split("\n") if line.strip())

        # ----- Step 1: retrieve + collect refine items -----
        qa_jobs = []          # (j, qa, retrieved_metadata, qa_dialogue)
        refine_items = []     # (qa_job_index, prompt, path_text)

        for j, sess in enumerate(sessions_so_far):
            qa = sess.qa[0]
            timelines, log_items = self.module.retrieve(qa.question)

            retrieved_metadata = [
                {
                    "session_id": item["source_turn"]["session_id"],
                    "conv_id":    item["source_turn"]["conv_id"],
                    "turn_id":    item["source_turn"]["turn_id_start"],
                    "score":      item.get("score", 0.0),
                }
                for item in log_items
            ]

            _append_jsonl(retrieval_log_path, build_retrieval_log_entry(
                k=k,
                question_session=j,
                query=qa.question,
                retrieved_items=log_items,
                use_timelines=timelines.get("use_timeline", []),
                timeline_info=timelines.get("timeline", []),
                current_dialogue_turns=dialogue_turn_count,
            ))

            qa_jobs.append({
                "j":                  j,
                "qa":                 qa,
                "retrieved_metadata": retrieved_metadata,
                "refined_texts":      [],
            })
            qa_idx = len(qa_jobs) - 1

            for path in timelines.get("use_timeline", []):
                path_text = self.module.timeline.get_path_text(
                    path, self.module.memory_graph,
                )
                prompt = self.module.timeline.build_refine_prompt(
                    path_text, qa_dialogue, qa.question,
                )
                refine_items.append({
                    "qa_idx":    qa_idx,
                    "prompt":    prompt,
                    "path_text": path_text,
                })

        # ----- Step 2: batch refine -----
        refine_totals = {"input": 0, "output": 0, "llm_calls": 0}
        if refine_items:
            refine_prompts = [it["prompt"] for it in refine_items]
            refine_results, refine_usages = self._batch_generate_with_retry(
                prompts=refine_prompts,
                max_tokens=cfg.MAX_TOKENS,
                guided_json=_REFINEMENT_GUIDED_JSON,
                batch_limit=cfg.REFINE_BATCH_SIZE,
            )
            for it, result, usage in zip(refine_items, refine_results, refine_usages):
                if self.module.timeline._llm_logger is not None:
                    self.module.timeline._llm_logger.log(
                        "call_2_refinement", "", it["prompt"], result,
                    )
                refine_totals["input"]     += usage["prompt_tokens"]
                refine_totals["output"]    += usage["completion_tokens"]
                refine_totals["llm_calls"] += 1
                refined_text = self.module.timeline.parse_refine_result(result, it["path_text"])
                qa_jobs[it["qa_idx"]]["refined_texts"].append(refined_text)

        # ----- Step 3: build QA prompts -----
        qa_prompts = []
        for job in qa_jobs:
            prompt, _ = self.module.generator.build_qa_prompt(
                question=job["qa"].question,
                retrieved_summaries=job["refined_texts"],
                current_dialogue=qa_dialogue,
            )
            job["prompt"] = prompt
            qa_prompts.append(prompt)

        # ----- Step 4: batch QA -----
        qa_results, qa_usages = self._batch_generate_with_retry(
            prompts=qa_prompts,
            max_tokens=cfg.MAX_TOKENS,
            guided_json=QA_SCHEMA,
            batch_limit=cfg.QA_BATCH_SIZE,
        )

        # ----- Step 5: assemble rows -----
        rows: List[Dict] = []
        qa_totals = {"input": 0, "output": 0, "llm_calls": 0}
        for job, result, usage in zip(qa_jobs, qa_results, qa_usages):
            if self.module.generator._llm_logger is not None:
                self.module.generator._llm_logger.log(
                    "call_5_qa", "", job["prompt"], result,
                )
            qa_totals["input"]     += usage["prompt_tokens"]
            qa_totals["output"]    += usage["completion_tokens"]
            qa_totals["llm_calls"] += 1

            answer = self.module.generator.parse_qa_result(result)
            qa = job["qa"]
            rows.append({
                "k":                  k,
                "question_session":   job["j"],
                "question":           qa.question,
                "model_answer":       answer,
                "topic":              qa.topic,
                "persona":            qa.persona,
                "preference":         qa.preference,
                "explanation":        qa.explanation,
                "retrieved_memories": job["retrieved_metadata"],
                "qa_tokens": {
                    "input":  usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "model":  self.model_path,
                },
            })

        return rows, refine_totals, qa_totals

    # ------------------------------------------------------------------
    # Batched generate with per-item JSON retry fallback
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        max_tokens: int,
        guided_json=None,
        batch_limit: Optional[int] = None,
    ) -> Tuple[List, List[Dict]]:
        if not prompts:
            return [], []

        from llm_client import _parse_json_response

        batch_limit = batch_limit or len(prompts)
        all_results = []
        all_usages = []

        for start in range(0, len(prompts), batch_limit):
            chunk = prompts[start:start + batch_limit]
            texts, usages = self.llm_client.generate_batch_raw(
                prompts=chunk,
                max_tokens=max_tokens,
                temperature=cfg.TEMPERATURE,
                guided_json=guided_json,
                return_usage=True,
            )

            if guided_json is None:
                all_results.extend(texts)
                all_usages.extend(usages)
                continue

            parsed_chunk = []
            for idx, text in enumerate(texts):
                try:
                    parsed_chunk.append(_parse_json_response(text) if isinstance(text, str) else text)
                except (json.JSONDecodeError, ValueError):
                    try:
                        retry_result = self.llm_client.generate(
                            prompt=chunk[idx],
                            guided_json=guided_json,
                            temperature=cfg.TEMPERATURE,
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
                            parsed_chunk.append(retry_result)
                        else:
                            parsed_chunk.append({})
                    except Exception as e:
                        logger.warning(f"Sequential retry failed for batch item {start + idx}: {e}")
                        parsed_chunk.append({})

            all_results.extend(parsed_chunk)
            all_usages.extend(usages)

        return all_results, all_usages


# =============================================================================
# RUNNER STATE RECONSTRUCTION FOR RESUME
# =============================================================================

def reconstruct_runner_state(
    sessions: List[Session],
    k_existing: int,
) -> Dict:
    """
    Reconstruct runner-side state at end-of-checkpoint k_existing,
    consistent with natural-boundary policy (Q1 = b, no force flush).

    State:
      current_conv_id : k_existing
      current_day     : k_existing // CONV_IDS_PER_DAY
      current_dialogue: concat of sessions in current virtual day = sessions
                        [day*CONV_IDS_PER_DAY .. k_existing] (Q2 = a)
      pending_count   : k_existing % FINALIZE_EVERY_N_CONVS    (k_existing > 0)
                        0                                      (k_existing == 0)
      finalize_turns  : turns of sessions still un-finalized = the most recent
                        (pending_count + 1) sessions for k > 0; sessions[0..0]
                        for k == 0
    """
    if k_existing < 0:
        return {
            "current_conv_id": None,
            "current_day":     -1,
            "current_dialogue": "",
            "finalize_turns":  [],
            "pending_count":   0,
        }

    cpd = cfg.CONV_IDS_PER_DAY
    fen = cfg.FINALIZE_EVERY_N_CONVS

    # Day & current_dialogue (Q2 = a)
    day = k_existing // cpd
    day_start = day * cpd
    day_pairs = []
    for j in range(day_start, k_existing + 1):
        day_pairs.extend(sessions[j].get_turn_pairs())
    current_dialogue = _format_session_dialogue(day_pairs) + ("\n" if day_pairs else "")

    # Finalize state
    if k_existing == 0:
        # No boundary crossed yet — finalize_turns = session 0's turns, pending = 0
        pending_count = 0
        unfinalized_sessions = [0]
    else:
        # Last natural finalize was at boundary entering session
        # last_natural = floor(k_existing / fen) * fen   (largest multiple of fen <= k_existing)
        # finalize_turns now contains turns from sessions [last_natural .. k_existing]
        # pending_count = k_existing - last_natural  (1..fen-1) or 0 if last_natural == k_existing
        last_natural = (k_existing // fen) * fen
        pending_count = k_existing - last_natural
        unfinalized_sessions = list(range(last_natural, k_existing + 1))

    finalize_turns: List[Turn] = []
    for j in unfinalized_sessions:
        for user_turn, asst_turn in sessions[j].get_turn_pairs():
            finalize_turns.append(user_turn)
            if asst_turn:
                finalize_turns.append(asst_turn)

    return {
        "current_conv_id": k_existing,
        "current_day":     day,
        "current_dialogue": current_dialogue,
        "finalize_turns":  finalize_turns,
        "pending_count":   pending_count,
    }


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
        description="Theanine PrefEval Chain Experiment",
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
    logger.info("Theanine PrefEval Chain")
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

    # ---- Step 5: init embedding + LLM + module ----
    logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )

    from theanine_module import TheanineModule, LLMCallLogger
    module = TheanineModule(llm_client, args.model, embedding_model=shared_embedding_model)
    if cfg.ENABLE_LLM_CALL_LOGGING:
        module.set_llm_logger(LLMCallLogger(prompt_log_dir))

    # ---- Step 6: resume state ----
    state = reconstruct_runner_state(sessions, k_existing)
    if k_existing >= 0:
        snap_dir = snapshots_dir / f"m_{k_existing}"
        logger.info(f"Loading snapshot {snap_dir}")
        module.load_memory_snapshot(snap_dir)
        # Drain stray counters defensively
        module.get_and_reset_memory_tokens()
        module.timeline.get_and_reset_token_counts()
        logger.info(
            f"Resumed: {module.get_memory_count()} nodes, "
            f"finalize_turns={len(state['finalize_turns'])}, "
            f"pending_count={state['pending_count']}, "
            f"current_day={state['current_day']}"
        )

    # ---- Step 7: meta.json on first run ----
    if not meta_file.exists():
        config_metadata = {
            "config_name":           config_name,
            "model":                 args.model,
            "embedding_model":       cfg.EMBEDDING_MODEL,
            "linking_top_j":         cfg.LINKING_TOP_J,
            "retrieve_top_k":        cfg.RETRIEVE_TOP_K,
            "timeline_sample_n":     cfg.TIMELINE_SAMPLE_N,
            "conv_ids_per_day":      cfg.CONV_IDS_PER_DAY,
            "minutes_per_turn":      cfg.MINUTES_PER_TURN,
            "finalize_every_n_convs": cfg.FINALIZE_EVERY_N_CONVS,
            "temperature":           cfg.TEMPERATURE,
            "max_tokens":            cfg.MAX_TOKENS,
            "summarize_max_tokens":  cfg.SUMMARIZE_MAX_TOKENS,
            "first_run_at":          datetime.now().isoformat(),
        }
        _atomic_write_json(meta_file, config_metadata)

    # ---- Step 8: stats.json (cumulative) ----
    stats = load_stats(stats_file)

    # ---- Step 9: chain loop ----
    runner = ChainRunner(module, llm_client, args.model)

    print(f"\n{'#'*60}")
    print(f"# Theanine PrefEval Chain  |  model={cfg.extract_model_name(args.model)}")
    print(f"# Sessions {start_k}..{K}  ({K - start_k + 1} to ingest)")
    print(f"# Policy: natural-boundary finalize (FINALIZE_EVERY_N_CONVS={cfg.FINALIZE_EVERY_N_CONVS}, no force flush)")
    print(f"{'#'*60}\n")

    for k in tqdm(range(start_k, K + 1), desc="Chain checkpoints"):
        session = sessions[k]
        logger.info(f"--- checkpoint k={k}: ingesting (conv_id={session.conv_id}, "
                    f"{session.total_turns} turns) ---")

        # (a) Boundary check (entering new conv_id k)
        if state["current_conv_id"] is not None:
            state["pending_count"] += 1
            if (state["pending_count"] >= cfg.FINALIZE_EVERY_N_CONVS
                and state["finalize_turns"]):
                # Natural finalize for accumulated turns
                ft = state["finalize_turns"]
                logger.info(
                    f"natural finalize at boundary {state['current_conv_id']}->{k}: "
                    f"{len(ft)} turns"
                )
                module.finalize_conv(
                    conv_id=state["current_conv_id"],
                    session_id=0,
                    full_conv_dialogue=_format_finalize_dialogue(ft),
                    turn_id_start=ft[0].turn_id          if ft else -1,
                    turn_id_end=ft[-1].turn_id           if ft else -1,
                    global_turn_id_start=ft[0].global_turn_id  if ft else -1,
                    global_turn_id_end=ft[-1].global_turn_id   if ft else -1,
                )
                state["finalize_turns"] = []
                state["pending_count"]  = 0

        # (b) Day boundary
        new_day = k // cfg.CONV_IDS_PER_DAY
        if new_day != state["current_day"]:
            state["current_dialogue"] = ""
            state["current_day"] = new_day

        # (c) Ingest session k (no LLM calls)
        for user_turn, assistant_turn in session.get_turn_pairs():
            state["current_conv_id"] = user_turn.conv_id
            state["current_dialogue"] += user_turn.to_message() + "\n"
            if assistant_turn:
                state["current_dialogue"] += assistant_turn.to_message() + "\n"
            state["finalize_turns"].append(user_turn)
            if assistant_turn:
                state["finalize_turns"].append(assistant_turn)

        # Drain memory tokens accumulated during finalize (if any)
        mem_tokens = module.get_and_reset_memory_tokens()

        # (e) QA at checkpoint k
        logger.info(f"k={k}: running QA over {k + 1} questions "
                    f"(graph nodes={module.get_memory_count()})")
        rows, refine_totals, qa_totals = runner.run_checkpoint_qa(
            sessions_so_far=sessions[:k + 1],
            k=k,
            current_dialogue=state["current_dialogue"],
            retrieval_log_path=retrieval_log,
        )
        # Drain refine tokens (already collected via runner)
        # but reset module-side counter for cleanliness
        module.timeline.get_and_reset_token_counts()

        # Append rows
        for row in rows:
            _append_jsonl(results_file, row)

        # (f) Save snapshot AFTER QA succeeds
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snap_dir = snapshots_dir / f"m_{k}"
            module.save_memory_snapshot(snap_dir)
            logger.info(f"Saved snapshot {snap_dir}")

        # (g) Update cumulative stats
        merge_stats(stats, mem_tokens, refine_totals, qa_totals, k)
        _atomic_write_json(stats_file, stats)

        logger.info(
            f"k={k} done. nodes={module.get_memory_count()}, "
            f"summ_calls={mem_tokens.get('call_3_summarization', {}).get('llm_calls', 0)}, "
            f"rel_calls={mem_tokens.get('call_4_relation', {}).get('llm_calls', 0)}, "
            f"refine_calls={refine_totals['llm_calls']}, "
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
