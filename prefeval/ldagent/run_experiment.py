"""
LD-Agent Experiment Runner — PrefEval (single-chain, cumulative-checkpoint).

Flow per invocation
-------------------
Args: --end-session K, --model, --tensor-parallel, --gpu-memory, --max-model-len, --config

1. Resolve output dir; scan memory_snapshots/m_*/ for max k_existing.
2. If K <= k_existing → no-op, exit.
3. Init shared encoder + LLM client; create LDAgentModule.
   If k_existing >= 0:
     - load_snapshot(m_{k_existing})  — restores STM, LTM, personas, last_conv_id,
       pending_conv_count
     - start_k = k_existing + 1
   Else:
     - start_k = 0
4. Load PrefEval samples [0..K] as a chain of Sessions.
5. For k = start_k .. K:
     a. For each (user_turn, asst_turn) in session k:
         - module.process_turn(...)  (sequential, Q2 = a)
         - drain per-call tokens from per.last_user_token_info,
           per.last_agent_token_info, mb.last_summarize_token_info
     b. Run batched QA over q_0..q_k:
         - retrieve LTM per question, build prompts (single shared system prompt)
         - batch generate (chunked by QA_BATCH_SIZE)
     c. Save snapshot m_k (after QA succeeds).
     d. Update cumulative stats.json.

Q1 (a) policy: STM clears at natural FINALIZE_EVERY_N_CONVS=2 boundary inside
context_retrieve. No force flush at checkpoints.

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

from load_dataset import load_prefeval_chain, Session, Turn, QAPair, compute_virtual_seconds


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"ldagent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            # Validate by core file presence
            if (p / "memory_state.json").exists():
                max_k = max(max_k, k)
    return max_k


# =============================================================================
# CUMULATIVE STATS
# =============================================================================

def _empty_stats() -> Dict:
    return {
        "call_2_user_persona":  {"input": 0, "output": 0, "llm_calls": 0},
        "call_3_agent_persona": {"input": 0, "output": 0, "llm_calls": 0},
        "call_4_summarization": {"input": 0, "output": 0, "llm_calls": 0},
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


def merge_stats(stats: Dict, ck_tokens: Dict, k: int) -> None:
    """Add deltas from one checkpoint into cumulative stats."""
    for ct in ("call_2_user_persona", "call_3_agent_persona",
               "call_4_summarization", "call_5_qa"):
        bucket = ck_tokens.get(ct, {})
        stats[ct]["input"]     += bucket.get("input", 0)
        stats[ct]["output"]    += bucket.get("output", 0)
        stats[ct]["llm_calls"] += bucket.get("llm_calls", 0)

    if k not in stats["checkpoints_completed"]:
        stats["checkpoints_completed"].append(k)
        stats["checkpoints_completed"].sort()


# =============================================================================
# RETRIEVAL LOG
# =============================================================================

def build_retrieval_log_entry(
    k: int,
    question_session: int,
    query: str,
    relevant_memories: List[Dict],
    ltm_count: int,
    stm_turns: int,
    user_trait_count: int,
    agent_trait_count: int,
) -> Dict:
    return {
        "timestamp":         datetime.now().isoformat(),
        "phase":             "qa",
        "k":                 k,
        "question_session":  question_session,
        "query":             query,
        "retrieved_items":   relevant_memories,
        "retrieval_scores":  [m.get("score", 0.0) for m in relevant_memories],
        "module_specific": {
            "module":            "ldagent",
            "ltm_entry_count":   ltm_count,
            "stm_context_turns": stm_turns,
            "user_trait_count":  user_trait_count,
            "agent_trait_count": agent_trait_count,
        },
    }


# =============================================================================
# CHAIN RUNNER
# =============================================================================

def _count_traits(traits_str: str) -> int:
    return len([t for t in traits_str.split("\n") if t.strip()]) if traits_str else 0


class ChainRunner:
    """Cumulative-chain pipeline against a single LDAgentModule."""

    def __init__(self, module, llm_client, model_path: str):
        self.module = module
        self.llm_client = llm_client
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Phase 1 — sequential ingestion of one session (Q2 = a)
    # ------------------------------------------------------------------

    def ingest_session(self, session: Session) -> Dict[str, Dict]:
        """
        Sequential per-turn ingestion. After each turn, drain per-call tokens.

        Returns per-call token totals for this session:
          {
            "call_2_user_persona":  {"input", "output", "llm_calls"},
            "call_3_agent_persona": {"input", "output", "llm_calls"},
            "call_4_summarization": {"input", "output", "llm_calls"},
          }
        """
        per_call = {
            "call_2_user_persona":  {"input": 0, "output": 0, "llm_calls": 0},
            "call_3_agent_persona": {"input": 0, "output": 0, "llm_calls": 0},
            "call_4_summarization": {"input": 0, "output": 0, "llm_calls": 0},
        }

        mb  = self.module.memory_bank
        per = self.module.personas

        for user_turn, assistant_turn in tqdm(
            session.get_turn_pairs(),
            desc=f"Ingest k={session.conv_id}",
            leave=False,
        ):
            if assistant_turn is None:
                # Skip turns without assistant (data quirk)
                continue
            self.module.process_turn(
                user_utterance=user_turn.utterance,
                gt_response=assistant_turn.utterance,
                conv_id=user_turn.conv_id,
                turn_id=user_turn.turn_id,
                session_id=user_turn.session_id,
            )

            # Drain per-call tokens (set inside process_turn)
            summ = mb.last_summarize_token_info
            if summ.get("input", 0) > 0 or summ.get("output", 0) > 0:
                per_call["call_4_summarization"]["input"]     += summ["input"]
                per_call["call_4_summarization"]["output"]    += summ["output"]
                per_call["call_4_summarization"]["llm_calls"] += 1

            up = per.last_user_token_info
            per_call["call_2_user_persona"]["input"]     += up.get("input", 0)
            per_call["call_2_user_persona"]["output"]    += up.get("output", 0)
            per_call["call_2_user_persona"]["llm_calls"] += 1

            ap = per.last_agent_token_info
            per_call["call_3_agent_persona"]["input"]     += ap.get("input", 0)
            per_call["call_3_agent_persona"]["output"]    += ap.get("output", 0)
            per_call["call_3_agent_persona"]["llm_calls"] += 1

        return per_call

    # ------------------------------------------------------------------
    # Phase 2 — batched QA at checkpoint k
    # ------------------------------------------------------------------

    def run_checkpoint_qa(
        self,
        sessions_so_far: List[Session],
        k: int,
        retrieval_log_path: Path,
    ) -> Tuple[List[Dict], Dict]:
        """At checkpoint k, evaluate q_0..q_k against current memory state."""
        from generator import format_memories_for_prompt, QA_RESPONSE_SCHEMA

        mb  = self.module.memory_bank
        per = self.module.personas
        gen = self.module.generator

        user_traits, agent_traits = per.get_current_traits()
        u_count = _count_traits(user_traits)
        a_count = _count_traits(agent_traits)

        # The system prompt only depends on agent_traits → identical across all (k+1) items
        # at this checkpoint, so we can submit it once via generate_batch_raw.
        # Build per-item user prompts.
        jobs = []
        shared_system_prompt = None

        for j, sess in enumerate(sessions_so_far):
            qa = sess.qa[0]
            relevant_memories = mb.relevance_retrieve(
                ori_query=qa.question,
                n_results=cfg.RETRIEVE_K,
                current_virtual_seconds=mb.current_virtual_seconds,
            )
            memories_str = format_memories_for_prompt(
                relevant_memories, mb.current_virtual_seconds,
            )
            context_str = self.module._format_stm_context_for_qa()

            sys_prompt, user_prompt, prompt_snapshot, _ = gen.build_qa_prompt(
                question=qa.question,
                memories=memories_str,
                user_traits=user_traits,
                agent_traits=agent_traits,
                context=context_str,
            )
            if shared_system_prompt is None:
                shared_system_prompt = sys_prompt

            retrieved_metadata = [
                {
                    "session_id":      m.get("session_id", 0),
                    "conv_id":         m.get("conv_id", 0),
                    "virtual_seconds": m.get("virtual_seconds", 0.0),
                    "score":           m.get("score", 0.0),
                }
                for m in relevant_memories
            ]

            _append_jsonl(retrieval_log_path, build_retrieval_log_entry(
                k=k,
                question_session=j,
                query=qa.question,
                relevant_memories=relevant_memories,
                ltm_count=mb.get_memory_count(),
                stm_turns=len(mb.short_term_memory),
                user_trait_count=u_count,
                agent_trait_count=a_count,
            ))

            jobs.append({
                "j":                  j,
                "qa":                 qa,
                "user_prompt":        user_prompt,
                "system_prompt":      sys_prompt,
                "retrieved_metadata": retrieved_metadata,
                "prompt_snapshot":    prompt_snapshot,
            })

        # Batch generate
        rows: List[Dict] = []
        qa_totals = {"input": 0, "output": 0, "llm_calls": 0}

        for chunk_start in range(0, len(jobs), cfg.QA_BATCH_SIZE):
            chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            chunk_prompts = [c["user_prompt"] for c in chunk]

            results, usages = self._batch_generate_with_retry(
                prompts=chunk_prompts,
                system_prompt=shared_system_prompt or "",
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=QA_RESPONSE_SCHEMA,
            )

            for c, result, usage in zip(chunk, results, usages):
                if gen.llm_logger is not None:
                    gen.llm_logger.log(
                        "call_5_qa",
                        c["system_prompt"],
                        c["user_prompt"],
                        result,
                    )

                qa_totals["input"]     += usage["prompt_tokens"]
                qa_totals["output"]    += usage["completion_tokens"]
                qa_totals["llm_calls"] += 1

                answer = result.get("answer", "") if isinstance(result, dict) else ""
                qa = c["qa"]
                rows.append({
                    "k":                  k,
                    "question_session":   c["j"],
                    "question":           qa.question,
                    "model_answer":       str(answer).strip(),
                    "topic":              qa.topic,
                    "persona":            qa.persona,
                    "preference":         qa.preference,
                    "explanation":        qa.explanation,
                    "retrieved_memories": c["retrieved_metadata"],
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
        description="LD-Agent PrefEval Chain Experiment",
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
    logger.info("LD-Agent PrefEval Chain")
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

    # ---- Step 5: init shared encoder + LLM + module ----
    logger.info("Loading shared encoder for LTM...")
    from sentence_transformers import SentenceTransformer
    shared_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )

    from ldagent_module import LDAgentModule, LLMCallLogger
    module = LDAgentModule(
        llm_client=llm_client,
        config=cfg,
        logger=logger,
        sample_id="prefeval_chain",
        shared_encoder=shared_encoder,
    )
    if cfg.ENABLE_LLM_CALL_LOGGING:
        module.set_llm_logger(LLMCallLogger(prompt_log_dir))

    # ---- Step 6: resume ----
    if k_existing >= 0:
        snap_dir = snapshots_dir / f"m_{k_existing}"
        logger.info(f"Loading snapshot {snap_dir}")
        module.load_snapshot(snap_dir)
        logger.info(
            f"Resumed: ltm={module.memory_bank.get_memory_count()}, "
            f"stm={len(module.memory_bank.short_term_memory)}, "
            f"user_traits={module.personas.get_user_trait_count()}, "
            f"agent_traits={module.personas.get_agent_trait_count()}, "
            f"last_conv_id={module.memory_bank.last_conv_id}, "
            f"pending_count={module.memory_bank._pending_conv_count}"
        )

    # ---- Step 7: meta.json on first run ----
    if not meta_file.exists():
        config_metadata = {
            "config_name":              config_name,
            "model":                    args.model,
            "embedding_model":          "sentence-transformers/all-MiniLM-L6-v2",
            "relevance_memory_number":  cfg.RELEVANCE_MEMORY_NUMBER,
            "retrieve_k":               cfg.RETRIEVE_K,
            "dist_threshold":           cfg.DIST_THRESHOLD,
            "decay_temp":               cfg.DECAY_TEMP,
            "conv_ids_per_day":         cfg.CONV_IDS_PER_DAY,
            "minutes_per_turn":         cfg.MINUTES_PER_TURN,
            "finalize_every_n_convs":   cfg.FINALIZE_EVERY_N_CONVS,
            "max_user_personas":        cfg.MAX_USER_PERSONAS,
            "max_agent_personas":       cfg.MAX_AGENT_PERSONAS,
            "temperature":              cfg.TEMPERATURE,
            "max_tokens":               cfg.MAX_TOKENS,
            "json_retry":               cfg.JSON_RETRY,
            "first_run_at":             datetime.now().isoformat(),
        }
        _atomic_write_json(meta_file, config_metadata)

    # ---- Step 8: stats.json (cumulative) ----
    stats = load_stats(stats_file)

    # ---- Step 9: chain loop ----
    runner = ChainRunner(module, llm_client, args.model)

    print(f"\n{'#'*60}")
    print(f"# LD-Agent PrefEval Chain  |  model={cfg.extract_model_name(args.model)}")
    print(f"# Sessions {start_k}..{K}  ({K - start_k + 1} to ingest)")
    print(f"# Policy: natural-boundary STM clear (FINALIZE_EVERY_N_CONVS={cfg.FINALIZE_EVERY_N_CONVS})")
    print(f"{'#'*60}\n")

    for k in tqdm(range(start_k, K + 1), desc="Chain checkpoints"):
        session = sessions[k]
        logger.info(f"--- checkpoint k={k}: ingesting (conv_id={session.conv_id}, "
                    f"{session.total_turns} turns) ---")

        # (a) Sequential ingestion
        per_call = runner.ingest_session(session)

        # (b) Batched QA at checkpoint k
        logger.info(
            f"k={k}: running QA over {k + 1} questions "
            f"(ltm={module.memory_bank.get_memory_count()}, "
            f"stm={len(module.memory_bank.short_term_memory)})"
        )
        rows, qa_totals = runner.run_checkpoint_qa(
            sessions_so_far=sessions[:k + 1],
            k=k,
            retrieval_log_path=retrieval_log,
        )
        per_call["call_5_qa"] = qa_totals

        # Append rows
        for row in rows:
            _append_jsonl(results_file, row)

        # (c) Save snapshot AFTER QA succeeds
        if cfg.SAVE_MEMORY_SNAPSHOTS:
            snap_dir = snapshots_dir / f"m_{k}"
            module.save_snapshot(snap_dir)
            logger.info(f"Saved snapshot {snap_dir}")

        # (d) Update cumulative stats
        merge_stats(stats, per_call, k)
        _atomic_write_json(stats_file, stats)

        logger.info(
            f"k={k} done. ltm={module.memory_bank.get_memory_count()}, "
            f"summ_calls={per_call['call_4_summarization']['llm_calls']}, "
            f"user_p={per_call['call_2_user_persona']['llm_calls']}, "
            f"agent_p={per_call['call_3_agent_persona']['llm_calls']}, "
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
