"""GraphMem experiment runner — batch-capable single entry point.

Phases:
  - memory: Phase 1 only (build memory + save snapshot + memory_build_stats).
  - qa:     Phase 2 only (load snapshot from BASE_MEMORY_CONFIG, run QA,
            merge base's Phase 1 stats into results).
  - all:    Both phases (current behavior; also writes memory_build_stats so
            future QA-variant configs can reuse the snapshot).

If --phase is omitted, the config name decides:
  - matches `config_<N>_q<M>` → qa
  - otherwise               → all
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

cfg = None  # type: ignore

from load_dataset import Session, load_implexconv_dataset


_QA_VARIANT_RE = re.compile(r"^config_\d+_q\d+$")


def load_config_module(config_name: str):
    """Load a config module by name (without .py). Does not touch sys.modules['config']."""
    config_file = _MODULE_DIR / f"{config_name}.py"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    spec = importlib.util.spec_from_file_location(config_name, config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_phase(config_name: str) -> str:
    """Infer phase from config name. Variants like `config_0_q1` run QA; everything else runs all."""
    return "qa" if _QA_VARIANT_RE.match(config_name) else "all"


def validate_memory_params(variant_cfg, base_cfg, variant_name: str, base_name: str) -> None:
    """Raise ValueError if any MEMORY_AFFECTING_PARAMS differ between variant and base."""
    params = getattr(base_cfg, "MEMORY_AFFECTING_PARAMS", None)
    if params is None:
        raise ValueError(f"{base_name} is missing MEMORY_AFFECTING_PARAMS — cannot validate.")
    mismatches = []
    for name in params:
        if not hasattr(base_cfg, name):
            mismatches.append(f"  {name}: missing in base ({base_name})")
            continue
        if not hasattr(variant_cfg, name):
            mismatches.append(f"  {name}: missing in variant ({variant_name})")
            continue
        bv, vv = getattr(base_cfg, name), getattr(variant_cfg, name)
        if bv != vv:
            mismatches.append(f"  {name}: base={bv!r} variant={vv!r}")
    if mismatches:
        raise ValueError(
            f"Memory-affecting parameters differ between {variant_name} and base {base_name}:\n"
            + "\n".join(mismatches)
            + f"\n\nQA-variant configs must inherit memory-affecting params from {base_name} unchanged."
        )


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        handlers.append(logging.FileHandler(log_dir / f"gmem4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


logger: logging.Logger = logging.getLogger(__name__)


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
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str,
) -> None:
    data = {
        "completed_session_ids": sorted(completed_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name": config_name,
            "model": model_path,
            "subset": subset,
            "start_session": start_session,
            "end_session": end_session,
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


def print_completion_banner(results_file: Path, total_sessions: int) -> None:
    banner = "#" * 60
    print(f"\n{banner}")
    print(f"# Experiment completed - {total_sessions} sessions processed")
    print(f"# Results: {results_file}")
    print(banner)
    print(banner)
    print()


def write_retrieval_log(log_path: Path, entry: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    session_id: int,
    conv_id: int,
    turn_id: int,
    query: str,
    retrieval_result,
) -> Dict:
    rr = retrieval_result
    return {
        "phase": "qa",
        "session_id": session_id,
        "conv_id": conv_id,
        "turn_id": turn_id,
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
                    "session_id": n.session_id,
                    "conv_id": n.conv_id,
                    "turn_id": n.turn_id,
                },
            }
            for n in rr.all_final_nodes
        ],
        "retrieval_scores": [rr.node_scores.get(n.node_id, 0.0) for n in rr.all_final_nodes],
        "module_specific": {
            "module": "gmem4",
            "num_by_slot": {
                "aps": len(rr.active_persona),
                "traits_stable": len(rr.traits_stable),
                "traits_challenged": len(rr.traits_challenged),
                "states_conflict": len(rr.states_conflict),
                "episodes_conflict": len(rr.episodes_conflict),
                "states_relevant": len(rr.states_relevant),
                "episodes_relevant": len(rr.episodes_relevant),
            },
            "seed_counts": {
                "context": len(rr.seed_contexts),
                "episode": len(rr.seed_episodes),
                "state": len(rr.seed_states),
                "trait": len(rr.seed_traits),
            },
            "pool_size": len(rr.pool_nodes),
        },
    }


class GraphMemBatchRunner:
    """Processes sessions in batches while batching LLM calls across modules."""

    def __init__(self, llm_client, subset: str, model_path: str, shared_embed_model, shared_nlp, config_metadata: Dict):
        self.llm_client = llm_client
        self.subset = subset
        self.model_path = model_path
        self.shared_embed_model = shared_embed_model
        self.shared_nlp = shared_nlp
        self.config_metadata = config_metadata

    def run_batch(
        self,
        sessions: List[Session],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
        phase: str = "all",
        base_snapshots_dir: Optional[Path] = None,
        base_phase1_data: Optional[Dict[int, Dict]] = None,
        base_memory_config: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """Run one batch under the requested phase.

        Returns {"results": [...], "memory_build_stats": [...]}:
          - phase=memory: results=[], memory_build_stats has Phase 1 info.
          - phase=qa:     results has QA results with base Phase 1 stats merged,
                          memory_build_stats=[].
          - phase=all:    results has full results, memory_build_stats also
                          populated so future QA variants can reuse.
        """
        if phase not in {"memory", "qa", "all"}:
            raise ValueError(f"Unknown phase: {phase!r}")
        if phase == "qa" and base_snapshots_dir is None:
            raise ValueError("phase=qa requires base_snapshots_dir")

        from graphmem_module import GraphMemModule, LLMCallLogger

        modules = [
            GraphMemModule(
                self.llm_client,
                model_path=self.model_path,
                config=cfg,
                embed_model=self.shared_embed_model,
                nlp=self.shared_nlp,
            )
            for _ in sessions
        ]

        for session, module, prompt_log_dir in zip(sessions, modules, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING and session.session_id < cfg.LLM_CALL_LOG_FIRST_N_SESSIONS:
                module.set_llm_logger(LLMCallLogger(prompt_log_dir))

        memory_build_stats: List[Dict] = []
        results: List[Dict] = []

        if phase in {"memory", "all"}:
            phase1_token_stats, phase1_internal_stats = self._run_phase1_batched(sessions, modules)
            memory_stats = [module.get_memory_stats() for module in modules]

            for i, session in enumerate(sessions):
                p1_breakdown = {}
                p1_total_in = p1_total_out = p1_total_calls = 0
                for call_type, counts in phase1_token_stats[i].items():
                    p1_breakdown[call_type] = counts
                    p1_total_in += counts["input"]
                    p1_total_out += counts["output"]
                    p1_total_calls += counts["llm_calls"]
                memory_build_stats.append({
                    "session_id": session.session_id,
                    "config_metadata": self.config_metadata,
                    "memory_at_qa_start": memory_stats[i],
                    "phase1_statistics": phase1_internal_stats[i],
                    "phase1_token_statistics": {
                        "per_call_type": p1_breakdown,
                        "total_input": p1_total_in,
                        "total_output": p1_total_out,
                        "total_llm_calls": p1_total_calls,
                    },
                    "memory_snapshot_path": (
                        f"memory_snapshots/session_{session.session_id}/" if cfg.SAVE_MEMORY_SNAPSHOTS else None
                    ),
                })

            if cfg.SAVE_MEMORY_SNAPSHOTS:
                for session, module in zip(sessions, modules):
                    snapshot_dir = snapshots_dir / f"session_{session.session_id}"
                    module.save_snapshot(snapshot_dir)

        elif phase == "qa":
            for session, module in zip(sessions, modules):
                snapshot_dir = base_snapshots_dir / f"session_{session.session_id}"
                if not snapshot_dir.exists():
                    raise FileNotFoundError(
                        f"Memory snapshot not found for session_id={session.session_id}: {snapshot_dir}\n"
                        f"Run base config ({base_memory_config}) with --phase memory or --phase all first."
                    )
                module.load_snapshot(snapshot_dir)

        if phase in {"qa", "all"}:
            qa_results_list, phase2_stats = self._run_phase2_batched(sessions, modules, retrieval_log_paths)

            for i, session in enumerate(sessions):
                if phase == "all":
                    p1_info = memory_build_stats[i]
                else:
                    p1_info = (base_phase1_data or {}).get(session.session_id)
                    if p1_info is None:
                        logger.warning(
                            f"No base Phase 1 stats for session_id={session.session_id}; "
                            "Phase 1 tokens will be zero in this session's result."
                        )
                        p1_info = {
                            "memory_at_qa_start": modules[i].get_memory_stats(),
                            "phase1_statistics": {},
                            "phase1_token_statistics": {
                                "per_call_type": {},
                                "total_input": 0,
                                "total_output": 0,
                                "total_llm_calls": 0,
                            },
                            "memory_snapshot_path": None,
                        }

                token_stats: Dict[str, Dict] = {}
                total_input = 0
                total_output = 0
                total_llm_calls = 0
                p1_tok = p1_info["phase1_token_statistics"]
                for call_type, counts in p1_tok.get("per_call_type", {}).items():
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

                result = {
                    "session_id": session.session_id,
                    "config_metadata": self.config_metadata,
                    "memory_at_qa_start": p1_info["memory_at_qa_start"],
                    "qa_results": qa_results_list[i],
                    "token_statistics": token_stats,
                    "phase1_statistics": p1_info["phase1_statistics"],
                    "memory_snapshot_path": p1_info.get("memory_snapshot_path"),
                }
                if phase == "qa":
                    result["base_memory_config"] = base_memory_config
                results.append(result)

        for module in modules:
            module.clear()

        return {"results": results, "memory_build_stats": memory_build_stats}

    def _run_phase1_batched(
        self, sessions: List[Session], modules
    ) -> Tuple[List[Dict], List[Dict]]:
        max_turns = max(len(s.get_turn_pairs()) for s in sessions) if sessions else 0

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active = [
                (i, sessions[i], modules[i])
                for i in range(len(sessions))
                if turn_idx < len(sessions[i].get_turn_pairs())
            ]
            if not active:
                continue

            pending = []
            for i, session, module in active:
                user_turn, _ = session.get_turn_pairs()[turn_idx]
                if user_turn is None:
                    continue
                call = module.prepare_pre_turn_call(user_turn.conv_id, user_turn.turn_id, session.session_id)
                if call is not None:
                    pending.append((i, module, call))
            self._drain_pending_calls(pending)

            turn_results = []
            for i, session, module in active:
                user_turn, assistant_turn = session.get_turn_pairs()[turn_idx]
                gt_response = assistant_turn.utterance if assistant_turn else ""
                module.process_turn_core(
                    user_utterance=user_turn.utterance,
                    gt_response=gt_response,
                    conv_id=user_turn.conv_id,
                    turn_id=user_turn.turn_id,
                    session_id=session.session_id,
                )
                turn_results.append((i, session, module))

            pending = []
            for i, session, module in turn_results:
                call = module.prepare_post_turn_call(session.session_id)
                if call is not None:
                    pending.append((i, module, call))
            self._drain_pending_calls(pending)

            for _, _, module in turn_results:
                module.advance_turn()

        pending = []
        for i, (session, module) in enumerate(zip(sessions, modules)):
            call = module.prepare_finalize_call(session.session_id)
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
            groups: Dict[str, List[Tuple[int, object, object]]] = {}
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

    def _run_phase2_batched(
        self,
        sessions: List[Session],
        modules,
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        qa_jobs = []

        for i, (session, module) in enumerate(zip(sessions, modules)):
            for qa in session.qa:
                prepared = module.prepare_qa(qa.question, subset=self.subset)
                write_retrieval_log(
                    retrieval_log_paths[i],
                    build_retrieval_log_entry(
                        session_id=session.session_id,
                        conv_id=-1,
                        turn_id=-1,
                        query=qa.question,
                        retrieval_result=prepared.retrieval_result,
                    ),
                )
                qa_jobs.append((i, qa, prepared))

        qa_results_per_session = [[] for _ in sessions]
        phase2_stats = [{"qa_input": 0, "qa_output": 0, "num_qa_calls": 0} for _ in sessions]

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
                answer = raw_answer
                if self.subset == "supportive":
                    label = answer.strip().lower()
                    answer = label if label in {"yes", "no"} else "no"

                phase2_stats[i]["qa_input"] += usage.get("prompt_tokens", 0)
                phase2_stats[i]["qa_output"] += usage.get("completion_tokens", 0)
                if raw_answer:
                    phase2_stats[i]["num_qa_calls"] += 1
                qa_results_per_session[i].append({
                    "question": qa.question,
                    "generated_answer": answer,
                    "ground_truth_answer": qa.answer,
                    "retrieved_memories": prepared.retrieved_memories,
                    "qa_tokens": {
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "model": self.model_path,
                    },
                })

        return qa_results_per_session, phase2_stats

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
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    }
                    parsed.append({k: v for k, v in retry.items() if k != "_usage"} if isinstance(retry, dict) else {})
                except Exception as exc:
                    logger.error(f"JSON retry failed: {exc}")
                    parsed.append({})

        return parsed, usages


def _try_parse_json(text: str) -> dict:
    """Best-effort JSON parser for raw LLM batch output text."""
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def create_llm_client(model_path: str, tensor_parallel: int, gpu_memory: float, max_model_len: Optional[int] = None):
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


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_0")
    pre_parser.add_argument("--phase", type=str, default=None, choices=["memory", "qa", "all"])
    pre_args, _ = pre_parser.parse_known_args()
    config_name = Path(pre_args.config).stem

    try:
        global cfg
        cfg = load_config_module(config_name)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1
    sys.modules["config"] = cfg

    phase = pre_args.phase or infer_phase(config_name)

    # Resolve base config and validate when running QA-only.
    base_cfg = None
    base_memory_config = None
    if phase == "qa":
        base_memory_config = getattr(cfg, "BASE_MEMORY_CONFIG", None)
        if not base_memory_config:
            print(
                f"[error] phase=qa requires BASE_MEMORY_CONFIG in {config_name}.py.\n"
                f"        Add e.g. `BASE_MEMORY_CONFIG = \"config_0\"` after `from config_0 import *`."
            )
            return 1
        try:
            base_cfg = load_config_module(base_memory_config)
        except FileNotFoundError as exc:
            print(f"[error] base config '{base_memory_config}' not found: {exc}")
            return 1
        try:
            validate_memory_params(cfg, base_cfg, config_name, base_memory_config)
        except ValueError as exc:
            print(f"[error] {exc}")
            return 1

    parser = argparse.ArgumentParser(description="GraphMem v4 Experiment on ImplexConv")
    parser.add_argument("--start-session", type=int, required=True)
    parser.add_argument("--end-session", type=int, required=True)
    parser.add_argument("--subset", type=str, required=True, choices=["opposed", "supportive"])
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config", type=str, default="config_0")
    parser.add_argument("--phase", type=str, default=None, choices=["memory", "qa", "all"])
    args = parser.parse_args()

    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")

    cfg.ensure_directories(args.model, args.subset, args.start_session, args.end_session, config_name)
    session_dir = cfg.get_session_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file = cfg.get_results_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    memory_build_stats_file = cfg.get_memory_build_stats_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    prompt_log_dir = cfg.get_prompt_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    snapshots_dir = cfg.get_memory_snapshots_dir(args.model, args.subset, args.start_session, args.end_session, config_name)

    # Phase 2 reads snapshots and Phase 1 stats from the base config's output dir.
    base_snapshots_dir = None
    base_phase1_data: Dict[int, Dict] = {}
    if phase == "qa":
        base_snapshots_dir = base_cfg.get_memory_snapshots_dir(
            args.model, args.subset, args.start_session, args.end_session, base_memory_config
        )
        if not base_snapshots_dir.exists():
            logger.error(
                f"Base memory snapshots not found: {base_snapshots_dir}\n"
                f"Build memory first: python run_experiment.py --config {base_memory_config} "
                f"--phase memory --subset {args.subset} --start-session {args.start_session} --end-session {args.end_session}"
            )
            return 1
        base_stats_file = base_cfg.get_memory_build_stats_file(
            args.model, args.subset, args.start_session, args.end_session, base_memory_config
        )
        if base_stats_file.exists():
            with open(base_stats_file) as f:
                base_phase1_data = {item["session_id"]: item for item in json.load(f)}
        else:
            logger.warning(
                f"Base memory_build_stats not found at {base_stats_file}; "
                "Phase 1 token stats will be zero in this run's results."
            )

    logger.info("=" * 60)
    logger.info("GraphMem v4 Experiment")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Phase           : {phase}")
    if phase == "qa":
        logger.info(f"  Base config     : {base_memory_config}")
        logger.info(f"  Base snapshots  : {base_snapshots_dir}")
    logger.info(f"  Subset          : {args.subset}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  Embedding model : {cfg.EMBEDDING_MODEL}")
    logger.info("=" * 60)

    dataset_path = cfg.get_dataset_path(args.subset)
    sessions = load_implexconv_dataset(dataset_path, args.subset)
    if args.end_session >= len(sessions):
        logger.error(f"end_session={args.end_session} out of range (dataset has {len(sessions)} sessions)")
        return 1

    target_sessions = sessions[args.start_session: args.end_session + 1]
    completed_ids = load_checkpoint(checkpoint_file) if cfg.ENABLE_CHECKPOINTING else set()
    existing_results = {item["session_id"]: item for item in load_existing_results(results_file)}
    existing_build_stats = (
        {item["session_id"]: item for item in load_existing_results(memory_build_stats_file)}
        if phase in {"memory", "all"} else {}
    )

    pending_sessions = [s for s in target_sessions if s.session_id not in completed_ids]
    if not pending_sessions:
        logger.info("No pending sessions to run.")
        return 0

    logger.info("Initializing LLM client...")
    llm_client = create_llm_client(
        args.model,
        args.tensor_parallel,
        args.gpu_memory,
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
        "config_name": config_name,
        "model": args.model,
        "subset": args.subset,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "temperature": cfg.TEMPERATURE,
        "max_tokens": cfg.MAX_TOKENS,
        "session_range": [args.start_session, args.end_session],
        "state_extraction_h": cfg.STATE_EXTRACTION_H,
        "chunk_size_conv": cfg.CHUNK_SIZE_CONV,
        "trait_extraction_chunks": cfg.TRAIT_EXTRACTION_CHUNKS,
        "k_sf": cfg.K_SF,
        "k_episode_final": cfg.K_EPISODE_FINAL,
        "k_t_final": cfg.K_T_FINAL,
        "k_aps": cfg.K_APS,
        "w_sr": cfg.W_SR,
        "enable_extra_relation_extraction": cfg.ENABLE_EXTRA_RELATION_EXTRACTION,
        "enable_shift_chain_pruning": cfg.ENABLE_SHIFT_CHAIN_PRUNING,
        "aps_exclude_shift_source": cfg.APS_EXCLUDE_SHIFT_SOURCE,
        "strict_high_default_low": cfg.STRICT_HIGH_DEFAULT_LOW,
        "batch_size": args.batch_size,
    }

    runner = GraphMemBatchRunner(
        llm_client=llm_client,
        subset=args.subset,
        model_path=args.model,
        shared_embed_model=shared_embed_model,
        shared_nlp=shared_nlp,
        config_metadata=config_metadata,
    )

    for batch_start in range(0, len(pending_sessions), args.batch_size):
        batch_sessions = pending_sessions[batch_start:batch_start + args.batch_size]
        retrieval_log_paths = [retrieval_log_dir / f"session_{s.session_id}_retrieval_log.jsonl" for s in batch_sessions]
        prompt_log_dirs = [prompt_log_dir / f"session_{s.session_id}" for s in batch_sessions]

        logger.info(f"Running batch: session_ids={[s.session_id for s in batch_sessions]}")
        batch_output = runner.run_batch(
            sessions=batch_sessions,
            retrieval_log_paths=retrieval_log_paths,
            snapshots_dir=snapshots_dir,
            prompt_log_dirs=prompt_log_dirs,
            phase=phase,
            base_snapshots_dir=base_snapshots_dir,
            base_phase1_data=base_phase1_data,
            base_memory_config=base_memory_config,
        )

        for stats in batch_output["memory_build_stats"]:
            existing_build_stats[stats["session_id"]] = stats
        for result in batch_output["results"]:
            existing_results[result["session_id"]] = result

        # Checkpoint policy: a session is "complete" when its primary output
        # for the current phase has been written. memory → snapshot saved;
        # qa → QA results written; all → both.
        for s in batch_sessions:
            sid = s.session_id
            if phase == "memory" and sid in existing_build_stats:
                completed_ids.add(sid)
            elif phase == "qa" and sid in existing_results:
                completed_ids.add(sid)
            elif phase == "all" and sid in existing_results and sid in existing_build_stats:
                completed_ids.add(sid)

        if phase in {"memory", "all"} and existing_build_stats:
            save_results(memory_build_stats_file, [existing_build_stats[k] for k in sorted(existing_build_stats)])
        if phase in {"qa", "all"} and existing_results:
            save_results(results_file, [existing_results[k] for k in sorted(existing_results)])
        if cfg.ENABLE_CHECKPOINTING:
            save_checkpoint(
                checkpoint_file,
                completed_ids,
                args.model,
                args.subset,
                args.start_session,
                args.end_session,
                config_name,
            )

    logger.info("All requested sessions completed.")
    primary_output = memory_build_stats_file if phase == "memory" else results_file
    print_completion_banner(primary_output, len(target_sessions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
