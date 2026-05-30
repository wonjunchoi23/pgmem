"""
PGMem Experiment Runner — PersonaMem

Usage:
    python run_experiment.py \\
        --start-session 0 --end-session 36 \\
        --benchmark-size 32k \\
        --model Qwen/Qwen3-1.7B \\
        --tensor-parallel 1 --gpu-memory 0.9 \\
        --batch-size 4 \\
        --config config_0
"""

import argparse
import importlib.util
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

cfg = None  # type: ignore

from load_dataset import (
    load_personamem_dataset,
    PersonaMemContext,
    PersonaMemMessage,
)


# =============================================================================
# MESSAGE PAIRING HELPERS
# =============================================================================

def _get_turn_pairs(context: PersonaMemContext, chunk_factor: int = 1) -> List[Tuple]:
    """
    Pair PersonaMem messages into (user_msg, asst_msg, virtual_conv_id, turn_id_in_chunk).

    When chunk_factor > 1, each block is split into chunk_factor equal sub-blocks so
    that episode extraction fires chunk_factor times per block instead of once.
    virtual_conv_id = block_idx * chunk_factor + sub_block_index.
    turn_id is reset to 0 at each sub-block boundary.
    """
    pairs = []
    blocks: Dict[int, List[PersonaMemMessage]] = {}
    for msg in context.messages:
        blocks.setdefault(msg.block_idx, []).append(msg)

    for block_idx in sorted(blocks.keys()):
        block_msgs = blocks[block_idx]
        raw_pairs: List[Tuple] = []
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
                raw_pairs.append((user_msg, asst_msg))

        total = len(raw_pairs)
        for local_idx, (user_msg, asst_msg) in enumerate(raw_pairs):
            if chunk_factor <= 1 or total == 0:
                sub = 0
            else:
                sub = min((local_idx * chunk_factor) // total, chunk_factor - 1)
            virtual_conv_id = block_idx * chunk_factor + sub
            sub_start = (sub * total) // chunk_factor if chunk_factor > 1 else 0
            turn_id = local_idx - sub_start
            pairs.append((user_msg, asst_msg, virtual_conv_id, turn_id))

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
        handlers.append(logging.FileHandler(
            log_dir / f"pgmem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ))
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
        return set(range(last + 1)) if last is not None else set()
    except Exception as exc:
        logger.warning(f"Could not load checkpoint: {exc}")
        return set()


def save_checkpoint(
    checkpoint_file: Path,
    completed_ids: Set[int],
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str,
) -> None:
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
            return json.load(f)
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


# =============================================================================
# RETRIEVAL LOGGING
# =============================================================================

def write_retrieval_log(log_path: Path, entry: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    context_index: int,
    block_idx: int,
    pair_idx: int,
    query: str,
    retrieval_result,
) -> Dict:
    rr = retrieval_result
    return {
        "phase": "qa",
        "context_index": context_index,
        "block_idx": block_idx,
        "pair_idx": pair_idx,
        "query": query,
        "memory_type": "graph_retrieval",
        "num_retrieved": len(rr.all_final_nodes),
        "retrieved_items": [
            {
                "node_id": n.node_id,
                "node_type": n.node_type,
                "content_preview": n.content[:120],
                "score": rr.node_scores.get(n.node_id, 0.0),
                "source_turn": {
                    "context_index": n.session_id,
                    "block_idx":     n.conv_id,
                    "pair_idx":      n.turn_id,
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
    }


def _remap_retrieved_memories(memories: List[Dict]) -> List[Dict]:
    """Remap internal field names to PersonaMem result schema names."""
    return [
        {
            "context_index": m.get("session_id", m.get("context_index", -1)),
            "block_idx":     m.get("conv_id",    m.get("block_idx", -1)),
            "pair_idx":      m.get("turn_id",    m.get("pair_idx", -1)),
            "node_type":     m.get("node_type", ""),
        }
        for m in memories
    ]


# =============================================================================
# BATCHED PGMEM RUNNER
# =============================================================================

class GraphMemBatchRunner:
    """Processes PersonaMem contexts in batches while batching LLM calls."""

    def __init__(
        self,
        llm_client,
        benchmark_size: str,
        model_path: str,
        shared_embed_model,
        shared_nlp,
        config_metadata: Dict,
    ):
        self.llm_client = llm_client
        self.benchmark_size = benchmark_size
        self.model_path = model_path
        self.shared_embed_model = shared_embed_model
        self.shared_nlp = shared_nlp
        self.config_metadata = config_metadata

    def run_batch(
        self,
        contexts: List[PersonaMemContext],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
        qa_only: bool = False,
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
            for _ in contexts
        ]

        for context, module, prompt_log_dir in zip(contexts, modules, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING and context.context_index < cfg.LLM_CALL_LOG_FIRST_N_SESSIONS:
                module.set_llm_logger(LLMCallLogger(prompt_log_dir))

        if qa_only:
            for context, module in zip(contexts, modules):
                snapshot_dir = snapshots_dir / f"session_{context.context_index}"
                if not snapshot_dir.exists():
                    raise FileNotFoundError(
                        f"qa-only mode: snapshot not found for context {context.context_index} at {snapshot_dir}"
                    )
                module.load_snapshot(snapshot_dir)
            phase1_token_stats = [{} for _ in contexts]
            phase1_internal_stats = [{} for _ in contexts]
        else:
            phase1_token_stats, phase1_internal_stats = self._run_phase1_batched(contexts, modules)

        memory_stats = [module.get_memory_stats() for module in modules]
        qa_results_list, phase2_stats = self._run_phase2_batched(contexts, modules, retrieval_log_paths)

        results = []
        for i, (context, module) in enumerate(zip(contexts, modules)):
            if cfg.SAVE_MEMORY_SNAPSHOTS and not qa_only:
                snapshot_dir = snapshots_dir / f"session_{context.context_index}"
                module.save_snapshot(snapshot_dir)

            module.clear()

            token_stats = {}
            total_input = total_output = total_llm_calls = 0
            for call_type, counts in phase1_token_stats[i].items():
                token_stats[call_type] = counts
                total_input     += counts["input"]
                total_output    += counts["output"]
                total_llm_calls += counts["llm_calls"]
            qa_stat = phase2_stats[i]
            token_stats["call_6_qa"] = {
                "input":     qa_stat["qa_input"],
                "output":    qa_stat["qa_output"],
                "llm_calls": qa_stat["num_qa_calls"],
            }
            total_input     += qa_stat["qa_input"]
            total_output    += qa_stat["qa_output"]
            total_llm_calls += qa_stat["num_qa_calls"]
            token_stats["total_input"]     = total_input
            token_stats["total_output"]    = total_output
            token_stats["total_llm_calls"] = total_llm_calls

            results.append({
                "context_index":     context.context_index,
                "persona_id":        context.persona_id,
                "shared_context_id": context.shared_context_id,
                "config_metadata":   self.config_metadata,
                "memory_at_qa_start": memory_stats[i],
                "qa_results":        qa_results_list[i],
                "token_statistics":  token_stats,
                "phase1_statistics": phase1_internal_stats[i],
                "memory_snapshot_path": (
                    f"memory_snapshots/session_{context.context_index}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            })

        return results

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_batched(
        self,
        contexts: List[PersonaMemContext],
        modules,
    ) -> Tuple[List[Dict], List[Dict]]:
        chunk_factor = getattr(cfg, "CHUNK_FACTOR", {}).get(self.benchmark_size, 1)
        context_pairs = [_get_turn_pairs(ctx, chunk_factor) for ctx in contexts]
        max_pairs = max((len(p) for p in context_pairs), default=0)

        for pair_idx in tqdm(range(max_pairs), desc="Phase1 pairs"):
            active = [
                (i, contexts[i], modules[i], context_pairs[i][pair_idx])
                for i in range(len(contexts))
                if pair_idx < len(context_pairs[i])
            ]
            if not active:
                continue

            # pre-turn calls (chunk-boundary episode/trait extraction)
            pending = []
            for i, context, module, (user_msg, asst_msg, block_idx, pair_in_block) in active:
                call = module.prepare_pre_turn_call(
                    conv_id=block_idx,
                    turn_id=pair_in_block,
                    session_id=context.context_index,
                )
                if call is not None:
                    pending.append((i, module, call))
            self._drain_pending_calls(pending)

            # process_turn_core for each active context
            turn_results = []
            for i, context, module, (user_msg, asst_msg, block_idx, pair_in_block) in active:
                user_utterance = _strip_prefix(user_msg.content, "user")
                gt_response = _strip_prefix(asst_msg.content, "assistant") if asst_msg else ""
                module.process_turn_core(
                    user_utterance=user_utterance,
                    gt_response=gt_response,
                    conv_id=block_idx,
                    turn_id=pair_in_block,
                    session_id=context.context_index,
                )
                turn_results.append((i, context, module))

            # post-turn calls (state extraction)
            pending = []
            for i, context, module in turn_results:
                call = module.prepare_post_turn_call(session_id=context.context_index)
                if call is not None:
                    pending.append((i, module, call))
            self._drain_pending_calls(pending)

            for _, _, module in turn_results:
                module.advance_turn()

        # finalize: last chunk episode/trait extraction
        pending = []
        for i, (context, module) in enumerate(zip(contexts, modules)):
            call = module.prepare_finalize_call(session_id=context.context_index)
            if call is not None:
                pending.append((i, module, call))
        self._drain_pending_calls(pending)

        phase1_token_stats   = [module.get_and_reset_internal_tokens() for module in modules]
        phase1_internal_stats = [module.get_and_reset_internal_stats() for module in modules]
        return phase1_token_stats, phase1_internal_stats

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
                # Mirrors updater.py's sequential JUDGMENT_RETRY policy on the
                # batched path, which previously bypassed it entirely.
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
                        # Accumulate retry token usage on top of original.
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
                    # Exhaustion fallback: if the call had expected judgments
                    # but the parsed array is still empty, store IRRELEVANT
                    # edges for every expected pair so future ⑤b/⑤c/⑤d can skip.
                    if call.expected_judgment_count > 0:
                        judgments = result.get("judgments", []) if isinstance(result, dict) else []
                        if isinstance(judgments, list) and len(judgments) == 0:
                            module.apply_irrelevant_fallback(call)
                    next_call = module.apply_pending_call(call, result if isinstance(result, dict) else {})
                    if next_call is not None:
                        next_jobs.append((idx, module, next_call))

            current_jobs = next_jobs

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        contexts: List[PersonaMemContext],
        modules,
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        qa_jobs = []

        for i, (context, module) in enumerate(zip(contexts, modules)):
            for qa in context.qa_pairs:
                prepared = module.prepare_qa(qa.question, options=qa.all_options)
                write_retrieval_log(
                    retrieval_log_paths[i],
                    build_retrieval_log_entry(
                        context_index=context.context_index,
                        block_idx=-1,
                        pair_idx=-1,
                        query=qa.question,
                        retrieval_result=prepared.retrieval_result,
                    ),
                )
                qa_jobs.append((i, qa, prepared))

        qa_results_per_context = [[] for _ in contexts]
        phase2_stats = [{"qa_input": 0, "qa_output": 0, "num_qa_calls": 0} for _ in contexts]

        for start in tqdm(range(0, len(qa_jobs), cfg.QA_BATCH_SIZE), desc="Phase2 QA chunks"):
            chunk = qa_jobs[start:start + cfg.QA_BATCH_SIZE]
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

            for (i, qa, prepared), result, usage in zip(chunk, results, usages):
                modules[i].log_call("call_6_qa", prepared.system_prompt, prepared.prompt, result)
                raw_answer = result.get("answer", "") if isinstance(result, dict) else ""
                label = raw_answer.strip().lower()
                answer = label if label in {"a", "b", "c", "d"} else "unknown"

                phase2_stats[i]["qa_input"]    += usage.get("prompt_tokens", 0)
                phase2_stats[i]["qa_output"]   += usage.get("completion_tokens", 0)
                phase2_stats[i]["num_qa_calls"] += 1

                qa_results_per_context[i].append({
                    "question":                    qa.question,
                    "question_type":               qa.question_type,
                    "topic":                       qa.topic,
                    "all_options":                 qa.all_options,
                    "generated_answer":            answer,
                    "ground_truth_answer":         qa.correct_answer,
                    "end_index_in_shared_context": qa.end_index_in_shared_context,
                    "retrieved_memories":          _remap_retrieved_memories(prepared.retrieved_memories),
                    "qa_tokens": {
                        "input":  usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "model":  self.model_path,
                    },
                })

        return qa_results_per_context, phase2_stats

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
                        texts.append(
                            result.get("content", "") if isinstance(result, dict) else str(result)
                        )
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
                    parsed.append(
                        {k: v for k, v in retry.items() if k != "_usage"}
                        if isinstance(retry, dict) else {}
                    )
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


def create_llm_client(
    model_path: str, tensor_parallel: int, gpu_memory: float,
    max_model_len: Optional[int] = None,
):
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

    parser = argparse.ArgumentParser(description="PGMem v6 Experiment on PersonaMem")
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
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Skip Phase1 and load memory snapshots from disk to run QA only. Requires --results-suffix.",
    )
    parser.add_argument(
        "--results-suffix",
        type=str,
        default=None,
        help="Suffix tag for results/checkpoint/retrieval_logs/prompt_log so QA-only runs don't overwrite originals. Required with --qa-only.",
    )
    args = parser.parse_args()

    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")

    if args.qa_only and not args.results_suffix:
        parser.error("--qa-only requires --results-suffix")
    if args.results_suffix and not args.qa_only:
        parser.error("--results-suffix requires --qa-only")

    suffix = args.results_suffix

    cfg.ensure_directories(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name, suffix
    )
    session_dir = cfg.get_session_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file = cfg.get_results_file(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name, suffix
    )
    checkpoint_file = cfg.get_checkpoint_file(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name, suffix
    )
    retrieval_log_dir = cfg.get_retrieval_log_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name, suffix
    )
    prompt_log_dir = cfg.get_prompt_log_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name, suffix
    )
    snapshots_dir = cfg.get_memory_snapshots_dir(
        args.model, args.benchmark_size, args.start_session, args.end_session, config_name
    )

    logger.info("=" * 60)
    logger.info("PGMem v6 Experiment — PersonaMem")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Mode            : {'QA-only (snapshots → QA)' if args.qa_only else 'full (Phase1 + QA)'}")
    if suffix:
        logger.info(f"  Results suffix  : {suffix}")
    logger.info(f"  Benchmark size  : {args.benchmark_size}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info("=" * 60)

    chunk_factor = getattr(cfg, "CHUNK_FACTOR", {}).get(args.benchmark_size, 1)
    cfg.TIME_PER_CONV_ID_HOURS = 24 // chunk_factor
    cfg.CONV_IDS_PER_DAY = chunk_factor
    logger.info(f"  Chunk factor    : {chunk_factor}x (TIME_PER_CONV_ID_HOURS={cfg.TIME_PER_CONV_ID_HOURS}h)")

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
    completed_ids = load_checkpoint(checkpoint_file) if cfg.ENABLE_CHECKPOINTING else set()
    existing_results = {
        item["context_index"]: item for item in load_existing_results(results_file)
    }

    pending_contexts = [c for c in target_contexts if c.context_index not in completed_ids]
    if not pending_contexts:
        logger.info("No pending contexts to run.")
        return 0

    logger.info("Initializing LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")

    logger.info("Loading shared embedding/spaCy models...")
    from sentence_transformers import SentenceTransformer
    import spacy

    shared_embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    try:
        shared_nlp = spacy.load(cfg.SPACY_MODEL)
    except OSError:
        spacy.cli.download(cfg.SPACY_MODEL)
        shared_nlp = spacy.load(cfg.SPACY_MODEL)

    config_metadata = {
        "config_name":                     config_name,
        "model":                           args.model,
        "benchmark_size":                  args.benchmark_size,
        "embedding_model":                 cfg.EMBEDDING_MODEL,
        "temperature":                     cfg.TEMPERATURE,
        "max_tokens":                      cfg.MAX_TOKENS,
        "session_range":                   [args.start_session, args.end_session],
        "state_extraction_h":              cfg.STATE_EXTRACTION_H,
        "chunk_size_conv":                 cfg.CHUNK_SIZE_CONV,
        "chunk_factor":                    chunk_factor,
        "trait_extraction_chunks":         cfg.TRAIT_EXTRACTION_CHUNKS,
        "k_sf":                            cfg.K_SF,
        "k_episode_final":                 cfg.K_EPISODE_FINAL,
        "k_aps":                           cfg.K_APS,
        "enable_extra_relation_extraction": cfg.ENABLE_EXTRA_RELATION_EXTRACTION,
        "enable_shift_chain_pruning":       cfg.ENABLE_SHIFT_CHAIN_PRUNING,
        "aps_exclude_shift_source":         cfg.APS_EXCLUDE_SHIFT_SOURCE,
        "strict_high_default_low":          cfg.STRICT_HIGH_DEFAULT_LOW,
        "batch_size":                      args.batch_size,
    }

    runner = GraphMemBatchRunner(
        llm_client=llm_client,
        benchmark_size=args.benchmark_size,
        model_path=args.model,
        shared_embed_model=shared_embed_model,
        shared_nlp=shared_nlp,
        config_metadata=config_metadata,
    )

    for batch_start in range(0, len(pending_contexts), args.batch_size):
        batch = pending_contexts[batch_start:batch_start + args.batch_size]
        retrieval_log_paths = [
            retrieval_log_dir / f"session_{c.context_index}_retrieval_log.jsonl"
            for c in batch
        ]
        prompt_log_dirs_batch = [
            prompt_log_dir / f"session_{c.context_index}"
            for c in batch
        ]

        logger.info(f"Running batch: context_indices={[c.context_index for c in batch]}")
        batch_results = runner.run_batch(
            contexts=batch,
            retrieval_log_paths=retrieval_log_paths,
            snapshots_dir=snapshots_dir,
            prompt_log_dirs=prompt_log_dirs_batch,
            qa_only=args.qa_only,
        )

        for result in batch_results:
            existing_results[result["context_index"]] = result
            completed_ids.add(result["context_index"])

        save_results(results_file, [existing_results[k] for k in sorted(existing_results)])
        if cfg.ENABLE_CHECKPOINTING:
            save_checkpoint(
                checkpoint_file,
                completed_ids,
                args.model,
                args.benchmark_size,
                args.start_session,
                args.end_session,
                config_name,
            )

    logger.info("All requested contexts completed.")
    print(f"\n{'#'*60}")
    print(f"# Experiment complete — {len(target_contexts)} contexts processed")
    print(f"# Results: {results_file}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
