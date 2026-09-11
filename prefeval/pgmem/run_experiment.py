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
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

cfg = None  # type: ignore

from load_dataset import load_prefeval_chain, Session


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"pgmem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    """Return the largest k for which snapshots_dir/m_k/module_state.json exists. -1 if none."""
    if not snapshots_dir.exists():
        return -1
    max_k = -1
    for p in snapshots_dir.iterdir():
        if not p.is_dir():
            continue
        m = _M_K_RE.match(p.name)
        if m and (p / "module_state.json").exists():
            max_k = max(max_k, int(m.group(1)))
    return max_k


# =============================================================================
# CUMULATIVE STATS
# =============================================================================

def _empty_stats() -> Dict:
    return {
        "call_tokens": {},          # {call_type: {input, output, llm_calls}}
        "internal_counters": {},    # flat dict of summed counters
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


def merge_stats(
    stats: Dict,
    mem_tokens: Dict[str, Dict[str, int]],
    qa_tokens: Dict[str, int],
    internal_stats: Dict[str, int],
    k: int,
) -> None:
    """Mutate `stats` in place by adding the deltas from one checkpoint."""
    call_tokens = stats["call_tokens"]
    for call_type, counts in mem_tokens.items():
        bucket = call_tokens.setdefault(call_type, {"input": 0, "output": 0, "llm_calls": 0})
        bucket["input"]     += counts.get("input", 0)
        bucket["output"]    += counts.get("output", 0)
        bucket["llm_calls"] += counts.get("llm_calls", 0)

    qa_bucket = call_tokens.setdefault("call_6_qa", {"input": 0, "output": 0, "llm_calls": 0})
    qa_bucket["input"]     += qa_tokens.get("input", 0)
    qa_bucket["output"]    += qa_tokens.get("output", 0)
    qa_bucket["llm_calls"] += qa_tokens.get("llm_calls", 0)

    counters = stats["internal_counters"]
    for name, val in internal_stats.items():
        counters[name] = counters.get(name, 0) + val

    if k not in stats["checkpoints_completed"]:
        stats["checkpoints_completed"].append(k)
        stats["checkpoints_completed"].sort()


# =============================================================================
# CHAIN RUNNER
# =============================================================================

class ChainRunner:

    def __init__(self, module, llm_client, model_path: str):
        self.module = module
        self.llm_client = llm_client
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Phase 1 — per-session ingestion (drain wraps each turn boundary)
    # ------------------------------------------------------------------

    def ingest_session(self, session: Session, session_id: int = 0) -> None:
        pairs = session.get_turn_pairs()
        for user_turn, asst_turn in tqdm(
            pairs, desc=f"Ingest k={session.conv_id}", leave=False
        ):
            conv_id = user_turn.conv_id
            turn_id = user_turn.turn_id
            gt_response = asst_turn.utterance if asst_turn is not None else ""

            # (1) pre-turn (chunk-boundary episode/trait extraction)
            call = self.module.prepare_pre_turn_call(
                conv_id=conv_id, turn_id=turn_id, session_id=session_id,
            )
            if call is not None:
                self._drain_pending_calls([(0, self.module, call)])

            # (2) process turn core (no LLM call inside)
            self.module.process_turn_core(
                user_utterance=user_turn.utterance,
                gt_response=gt_response,
                conv_id=conv_id,
                turn_id=turn_id,
                session_id=session_id,
            )

            # (3) post-turn (state extraction)
            call = self.module.prepare_post_turn_call(session_id=session_id)
            if call is not None:
                self._drain_pending_calls([(0, self.module, call)])

            self.module.advance_turn()

        # (4) finalize — flush last sub-chunk's episode/trait extraction so it
        # becomes visible to the next session in the chain.
        call = self.module.prepare_finalize_call(session_id=session_id)
        if call is not None:
            self._drain_pending_calls([(0, self.module, call)])

    # Batched-drain
    def _drain_pending_calls(self, pending_jobs: List[Tuple]) -> None:
        current_jobs = pending_jobs
        while current_jobs:
            next_jobs = []
            groups: Dict[str, List[Tuple]] = {}
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
                results = list(results)
                usages = list(usages)
                for _attempt in range(1, cfg.JUDGMENT_RETRY + 1):
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
                    if call.expected_judgment_count > 0:
                        judgments = result.get("judgments", []) if isinstance(result, dict) else []
                        if isinstance(judgments, list) and len(judgments) == 0:
                            module.apply_irrelevant_fallback(call)
                    next_call = module.apply_pending_call(call, result if isinstance(result, dict) else {})
                    if next_call is not None:
                        next_jobs.append((idx, module, next_call))

            current_jobs = next_jobs

    # ------------------------------------------------------------------
    # Phase 2 — batched QA at checkpoint k
    # ------------------------------------------------------------------

    def run_checkpoint_qa(
        self,
        sessions_so_far: List[Session],
        k: int,
        retrieval_log_path: Path,
    ) -> Tuple[List[Dict], Dict]:
        # Build retrieval + prompts for each q_j (j = 0..k)
        jobs = []
        for j, sess in enumerate(sessions_so_far):
            qa_pair = sess.qa[0]
            prepared = self.module.prepare_qa(qa_pair.question)

            rr = prepared.retrieval_result
            _append_jsonl(retrieval_log_path, {
                "timestamp":        datetime.now().isoformat(),
                "phase":            "qa",
                "k":                k,
                "question_session": j,
                "query":            qa_pair.question,
                "memory_type":      "graph_retrieval",
                "num_retrieved":    len(rr.all_final_nodes),
                "retrieved_items":  [
                    {
                        "node_id":         n.node_id,
                        "node_type":       n.node_type,
                        "content_preview": n.content[:120],
                        "score":           rr.node_scores.get(n.node_id, 0.0),
                        "source_turn": {
                            "session_id": n.session_id,
                            "conv_id":    n.conv_id,
                            "turn_id":    n.turn_id,
                        },
                    }
                    for n in rr.all_final_nodes
                ],
                "retrieval_scores": [rr.node_scores.get(n.node_id, 0.0) for n in rr.all_final_nodes],
                "module_specific": {
                    "module": "pgmem",
                    "num_by_slot": {
                        "aps":                len(rr.active_persona),
                        "traits_stable":      len(rr.traits_stable),
                        "traits_challenged":  len(rr.traits_challenged),
                        "states_conflict":    len(rr.states_conflict),
                        "episodes_conflict":  len(rr.episodes_conflict),
                        "states_relevant":    len(rr.states_relevant),
                        "episodes_relevant":  len(rr.episodes_relevant),
                    },
                    "seed_counts": {
                        "context": len(rr.seed_contexts),
                        "episode": len(rr.seed_episodes),
                        "state":   len(rr.seed_states),
                        "trait":   len(rr.seed_traits),
                    },
                    "pool_size": len(rr.pool_nodes),
                },
            })
            jobs.append((j, qa_pair, prepared))

        rows: List[Dict] = []
        qa_totals = {"input": 0, "output": 0, "llm_calls": 0}

        for chunk_start in tqdm(
            range(0, len(jobs), cfg.QA_BATCH_SIZE),
            desc=f"QA k={k} ({len(jobs)} Q)",
            leave=False,
        ):
            chunk = jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            if not chunk:
                continue
            prompts = [prepared.prompt for _, _, prepared in chunk]
            system_prompt = chunk[0][2].system_prompt
            schema = chunk[0][2].schema
            results, usages = self._batch_generate_with_retry(
                prompts=prompts,
                system_prompt=system_prompt,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=schema,
            )

            for (j, qa_pair, prepared), result, usage in zip(chunk, results, usages):
                self.module.log_call("call_6_qa", prepared.system_prompt, prepared.prompt, result)
                answer = result.get("answer", "") if isinstance(result, dict) else ""

                qa_totals["input"]     += usage.get("prompt_tokens", 0)
                qa_totals["output"]    += usage.get("completion_tokens", 0)
                qa_totals["llm_calls"] += 1

                rows.append({
                    "k":                  k,
                    "question_session":   j,
                    "question":           qa_pair.question,
                    "model_answer":       answer,
                    "topic":              qa_pair.topic,
                    "persona":            qa_pair.persona,
                    "preference":         qa_pair.preference,
                    "explanation":        qa_pair.explanation,
                    "retrieved_memories": prepared.retrieved_memories,
                    "qa_tokens": {
                        "input":  usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "model":  self.model_path,
                    },
                })

        return rows, qa_totals

    # ------------------------------------------------------------------
    # Batch generate with per-item JSON retry fallback
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
                        "prompt_tokens":     usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    })
                except Exception as exc:
                    logger.error(f"Sequential fallback generation failed: {exc}")
                    texts.append({} if guided_json is not None else "")
                    usages.append({"prompt_tokens": 0, "completion_tokens": 0})

        if guided_json is None:
            return texts, usages

        parsed = []
        for idx, text in enumerate(texts):
            if isinstance(text, dict):
                parsed.append({k: v for k, v in text.items() if k != "_usage"})
                continue
            try:
                parsed.append(_try_parse_json(text))
            except Exception:
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
                        "prompt_tokens":     usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    }
                    parsed.append({k: v for k, v in retry.items() if k != "_usage"}
                                  if isinstance(retry, dict) else {})
                except Exception as exc:
                    logger.error(f"JSON retry failed: {exc}")
                    parsed.append({})

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


def create_llm_client(model_path: str, tensor_parallel: int, gpu_memory: float,
                     max_model_len: Optional[int] = None):
    from llm_client import create_llm_client as _create

    engine = cfg.LLM_ENGINE
    if engine == "vllm":
        kwargs = {
            "engine": "vllm",
            "model_path": model_path,
            "tensor_parallel_size": tensor_parallel,
            "gpu_memory_utilization": gpu_memory,
            "download_dir": None,
            "enable_prefix_caching": True,
        }
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
    # ---- Step 1: load config ----
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_8")
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

    # ---- Step 2: parse args ----
    parser = argparse.ArgumentParser(description="PGMem PrefEval Chain Experiment")
    parser.add_argument("--end-session", type=int, required=True,
                        help="Last sample index in chain (chain = samples[0..K])")
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=30000)
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.end_session < 0:
        parser.error("--end-session must be >= 0")

    cfg.ensure_directories(args.model, config_name)

    snapshots_dir  = cfg.get_memory_snapshots_dir(args.model, config_name)
    results_file   = cfg.get_results_file(args.model, config_name)
    retrieval_log  = cfg.get_retrieval_log_file(args.model, config_name)
    stats_file     = cfg.get_stats_file(args.model, config_name)
    meta_file      = cfg.get_meta_file(args.model, config_name)
    prompt_log_dir = cfg.get_prompt_log_dir(args.model, config_name)
    run_log_dir    = cfg.get_run_log_dir(args.model, config_name)

    global logger
    logger = setup_logging(run_log_dir)

    # ---- Step 3: resume detection ----
    k_existing = find_max_existing_k(snapshots_dir)
    K = args.end_session

    logger.info("=" * 60)
    logger.info("PGMem PrefEval Chain")
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

    # ---- Step 5: init embedding + spaCy + LLM + module ----
    logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

    logger.info(f"Loading spaCy model: {cfg.SPACY_MODEL}")
    import spacy
    try:
        shared_nlp = spacy.load(cfg.SPACY_MODEL)
    except OSError:
        spacy.cli.download(cfg.SPACY_MODEL)
        shared_nlp = spacy.load(cfg.SPACY_MODEL)

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )

    from graphmem_module import GraphMemModule, LLMCallLogger
    module = GraphMemModule(
        llm_client,
        model_path=args.model,
        config=cfg,
        embed_model=shared_embedding_model,
        nlp=shared_nlp,
    )
    if cfg.ENABLE_LLM_CALL_LOGGING:
        module.set_llm_logger(LLMCallLogger(prompt_log_dir))

    # Resume: load last snapshot
    if k_existing >= 0:
        snap_dir = snapshots_dir / f"m_{k_existing}"
        logger.info(f"Loading snapshot {snap_dir}")
        module.load_snapshot(snap_dir)
        # Drain stray counters before main loop.
        module.get_and_reset_internal_tokens()
        module.get_and_reset_internal_stats()
        node_counts = module.get_node_counts()
        logger.info(f"Resumed with node counts: {node_counts}")

    # ---- Step 6: meta.json on first run ----
    if not meta_file.exists():
        config_metadata = {
            "config_name":                     config_name,
            "model":                           args.model,
            "embedding_model":                 cfg.EMBEDDING_MODEL,
            "spacy_model":                     cfg.SPACY_MODEL,
            "temperature":                     cfg.TEMPERATURE,
            "max_tokens":                      cfg.MAX_TOKENS,
            "conv_ids_per_day":                cfg.CONV_IDS_PER_DAY,
            "minutes_per_turn":                cfg.TIME_PER_TURN_MINUTES,
            "time_per_conv_id_hours":          cfg.TIME_PER_CONV_ID_HOURS,
            "chunk_factor":                    cfg.CHUNK_FACTOR,
            "state_extraction_h":              cfg.STATE_EXTRACTION_H,
            "trait_extraction_chunks":         cfg.TRAIT_EXTRACTION_CHUNKS,
            "k_context":                       cfg.K_CONTEXT,
            "k_episode":                       cfg.K_EPISODE,
            "k_episode_final":                 cfg.K_EPISODE_FINAL,
            "k_state":                         cfg.K_STATE,
            "k_trait":                         cfg.K_TRAIT,
            "k_sf":                            cfg.K_SF,
            "k_aps":                           cfg.K_APS,
            "enable_extra_relation_extraction": cfg.ENABLE_EXTRA_RELATION_EXTRACTION,
            "enable_shift_chain_pruning":      cfg.ENABLE_SHIFT_CHAIN_PRUNING,
            "aps_exclude_shift_source":        cfg.APS_EXCLUDE_SHIFT_SOURCE,
            "strict_high_default_low":         cfg.STRICT_HIGH_DEFAULT_LOW,
            "first_run_at":                    datetime.now().isoformat(),
        }
        _atomic_write_json(meta_file, config_metadata)

    # ---- Step 7: stats.json (cumulative) ----
    stats = load_stats(stats_file)

    # ---- Step 8: chain loop ----
    runner = ChainRunner(module, llm_client, args.model)

    print(f"\n{'#'*60}")
    print(f"# PGMem PrefEval Chain  |  model={cfg.extract_model_name(args.model)}")
    print(f"# Sessions {start_k}..{K}  ({K - start_k + 1} to ingest)")
    print(f"{'#'*60}\n")

    for k in tqdm(range(start_k, K + 1), desc="Chain checkpoints"):
        session = sessions[k]
        logger.info(f"--- checkpoint k={k}: ingesting session (conv_id={session.conv_id}, "
                    f"{session.total_turns} turns) ---")

        # (a) Ingest session k
        runner.ingest_session(session, session_id=0)
        mem_tokens     = module.get_and_reset_internal_tokens()
        internal_stats = module.get_and_reset_internal_stats()

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
            module.save_snapshot(snap_dir)
            logger.info(f"Saved snapshot {snap_dir}")

        # (e) Update cumulative stats
        merge_stats(stats, mem_tokens, qa_totals, internal_stats, k)
        _atomic_write_json(stats_file, stats)

        node_counts = module.get_node_counts()
        logger.info(f"k={k} done. nodes={node_counts}, "
                    f"qa_in={qa_totals['input']}, qa_out={qa_totals['output']}, "
                    f"qa_calls={qa_totals['llm_calls']}")

    print(f"\n{'#'*60}")
    print("# Chain complete!")
    print(f"# Results -> {results_file}")
    print(f"# Snapshots -> {snapshots_dir}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
