"""
Theanine Batch Experiment Runner — LoComo
"""

import argparse
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
MODULE_SLUG = _MODULE_DIR.name

cfg = None  # type: ignore

from load_dataset import LoCoMoSession, QAPair, Sample, Turn, load_locomo_dataset

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# DATE PARSING
# =============================================================================

def _parse_date(date_time_str: str) -> Optional[str]:
    if " on " in date_time_str:
        return date_time_str.split(" on ", 1)[1].strip()
    try:
        from dateutil.parser import parse as _parse
        return str(_parse(date_time_str).date())
    except Exception:
        return None


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"{MODULE_SLUG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path) -> Tuple[Set[str], Optional[int]]:
    if not checkpoint_file.exists():
        return set(), None
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        if "completed_sample_ids" in data:
            return {str(x) for x in data["completed_sample_ids"]}, None
        return set(), data.get("last_completed_sample_index")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set(), None


def save_checkpoint(
    checkpoint_file: Path,
    completed_sample_ids: Set[str],
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
):
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
    sample_id: str,
    query: str,
    retrieved_items: List[Dict],
    use_timelines: Optional[List] = None,
    timeline_info: Optional[List[Dict]] = None,
    current_dialogue_turns: int = 0,
) -> Dict:
    if use_timelines:
        seed_ids = {item["memory_id"] for item in retrieved_items}
        path_node_ids = {
            elem
            for path in use_timelines
            for i, elem in enumerate(path)
            if i % 2 == 0
        }
        num_path_linked = len(path_node_ids - seed_ids)
    else:
        num_path_linked = 0

    return {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "sample_id": sample_id,
        "query": query[:500],
        "memory_type": ["seed", "path_linked", "current_dialogue"],
        "num_retrieved": [len(retrieved_items), num_path_linked, current_dialogue_turns],
        "retrieved_items": retrieved_items,
        "retrieval_scores": [item["score"] for item in retrieved_items],
        "module_specific": {
            "module": MODULE_SLUG,
            "num_paths_used": len(use_timelines) if use_timelines else 0,
            "use_timelines": [list(p) for p in use_timelines] if use_timelines else [],
            "timeline_info": timeline_info or [],
        },
    }


# =============================================================================
# HELPERS
# =============================================================================

class BatchedTheanineRunner:
    def __init__(self, llm_client, model_path: str, shared_embedding_model,
                 config_metadata: Optional[Dict] = None):
        self.llm_client = llm_client
        self.model_path = model_path
        self.shared_embedding_model = shared_embedding_model
        self.config_metadata = config_metadata or {}

    def run_batch(
        self,
        samples: List[Sample],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
    ) -> List[Dict]:
        from theanine_module import LLMCallLogger, TheanineModule

        modules = [
            TheanineModule(
                self.llm_client,
                self.model_path,
                embedding_model=self.shared_embedding_model,
            )
            for _ in samples
        ]

        for module, prompt_log_dir in zip(modules, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                module.set_llm_logger(LLMCallLogger(prompt_log_dir))

        phase1_stats = self._run_phase1_batched(samples, modules, retrieval_log_paths)

        memory_at_qa_start_list = [module.get_memory_stats() for module in modules]

        qa_results_list, phase2_stats = self._run_phase2_batched(samples, modules, retrieval_log_paths)

        results = []
        for i, (sample, module) in enumerate(zip(samples, modules)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"sample_{sample.sample_id}"
                module.save_memory_snapshot(snapshot_dir)
                logger.info(f"Memory snapshot saved: sample {sample.sample_id}")

            module.clear()
            logger.info(f"Memory cleared: sample {sample.sample_id}")

            p1 = phase1_stats[i]
            p2 = phase2_stats[i]
            qa_input = p2["refine_input"] + p2["qa_input"]
            token_stats = {
                "call_2_summarization": {
                    "input": p1["summ_input"],
                    "output": p1["summ_output"],
                    "llm_calls": p1["summ_calls"],
                    "parse_fallback_count": p1["summ_fallback"],
                },
                "call_3_relation": {
                    "input": p1["rel_input"],
                    "output": p1["rel_output"],
                    "llm_calls": p1["rel_calls"],
                },
                "call_1_refinement": {
                    "input": p2["refine_input"],
                    "output": p2["refine_output"],
                    "llm_calls": p2["num_refine_calls"],
                },
                "call_4_qa": {
                    "input": p2["qa_input"],
                    "output": p2["qa_output"],
                    "llm_calls": p2["num_qa_calls"],
                },
                "qa_input": qa_input,
                "total_input": p1["summ_input"] + p1["rel_input"] + qa_input,
                "total_output": (p1["summ_output"] + p1["rel_output"]
                                 + p2["refine_output"] + p2["qa_output"]),
                "total_llm_calls": (p1["summ_calls"] + p1["rel_calls"]
                                    + p2["num_refine_calls"] + p2["num_qa_calls"]),
            }
            results.append({
                "sample_id": sample.sample_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": memory_at_qa_start_list[i],
                "qa_results": qa_results_list[i],
                "token_statistics": token_stats,
                "memory_snapshot_path": (
                    f"memory_snapshots/sample_{sample.sample_id}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            })
        return results

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_batched(self, samples: List[Sample], modules, retrieval_log_paths: List[Path]) -> List[Dict]:
        phase1_stats = [
            {
                "summ_input": 0, "summ_output": 0, "summ_calls": 0, "summ_fallback": 0,
                "rel_input": 0, "rel_output": 0, "rel_calls": 0,
            }
            for _ in samples
        ]
        states = [
            {
                "prev_date": None,
                "finalize_idx": 0,
                "accumulated_turns": [],
                "accumulated_sessions": [],
            }
            for _ in samples
        ]

        max_sessions = max((len(sample.sessions) for sample in samples), default=0)
        for session_idx in tqdm(range(max_sessions), desc="Phase1 sessions"):
            finalize_jobs = []
            active_sessions = []

            for sample_idx, sample in enumerate(samples):
                if session_idx >= len(sample.sessions):
                    continue

                session = sample.sessions[session_idx]
                state = states[sample_idx]
                current_date = _parse_date(session.date_time)
                if current_date is None:
                    logger.warning(
                        f"Could not parse date_time='{session.date_time}' "
                        f"for sample {sample.sample_id}, session {session.session_id}. "
                        "Using force-finalize semantics."
                    )

                date_changed = (
                    state["prev_date"] is not None
                    and (current_date is None or current_date != state["prev_date"])
                )
                if date_changed and state["accumulated_turns"]:
                    finalize_jobs.append(self._build_finalize_job(
                        sample_idx=sample_idx,
                        sample=sample,
                        module=modules[sample_idx],
                        state=state,
                    ))
                    state["finalize_idx"] += 1
                    state["accumulated_turns"] = []
                    state["accumulated_sessions"] = []

                active_sessions.append((sample_idx, sample, modules[sample_idx], session, current_date))

            if finalize_jobs:
                self._run_finalize_jobs(finalize_jobs, phase1_stats)

            for sample_idx, sample, module, session, current_date in active_sessions:
                state = states[sample_idx]
                for turn in session.turns:
                    state["accumulated_turns"].append(turn)

                state["accumulated_sessions"].append(session.session_id)
                state["prev_date"] = current_date if current_date is not None else state["prev_date"]

        final_jobs = []
        for sample_idx, sample in enumerate(samples):
            state = states[sample_idx]
            if state["accumulated_turns"]:
                final_jobs.append(self._build_finalize_job(
                    sample_idx=sample_idx,
                    sample=sample,
                    module=modules[sample_idx],
                    state=state,
                ))

        if final_jobs:
            self._run_finalize_jobs(final_jobs, phase1_stats)

        for sample_idx, module in enumerate(modules):
            phase1_stats[sample_idx]["summ_fallback"] = (
                module.memory_graph.get_and_reset_summarize_fallback_count()
            )

        for sample_idx, sample in enumerate(samples):
            logger.info(
                f"Phase 1 done: sample {sample.sample_id}, "
                f"memory nodes={modules[sample_idx].get_memory_count()}, "
                f"finalize batches={states[sample_idx]['finalize_idx'] + (1 if sample.sessions else 0)}"
            )

        return phase1_stats

    def _build_finalize_job(self, sample_idx: int, sample: Sample, module, state: Dict) -> Dict:
        turns = state["accumulated_turns"]
        return {
            "sample_idx": sample_idx,
            "sample_id": sample.sample_id,
            "module": module,
            "finalize_idx": state["finalize_idx"],
            "full_dialogue": "\n".join(f"{t.speaker}: {t.text}" for t in turns),
            "dia_ids": [t.dia_id for t in turns],
            "session_ids": list(state["accumulated_sessions"]),
            "date": state["prev_date"] or "",
            "relation_results": [],
            "new_nodes": [],
        }

    def _run_finalize_jobs(self, finalize_jobs: List[Dict], phase1_stats: List[Dict]) -> None:
        from memory_graph import RELATION_GUIDED_JSON, SUMMARIZATION_GUIDED_JSON

        if not finalize_jobs:
            return

        for start in tqdm(range(0, len(finalize_jobs), cfg.SUMMARIZE_BATCH_SIZE), desc="Phase1 summarize"):
            chunk = finalize_jobs[start:start + cfg.SUMMARIZE_BATCH_SIZE]
            prompts = [job["module"].memory_graph.build_summarize_prompt(job["full_dialogue"]) for job in chunk]

            def retry_single(local_idx: int, _prompt: str):
                job = chunk[local_idx]
                sentences, token_info = job["module"].memory_graph._summarize_conv(job["full_dialogue"])
                return {"sentences": sentences}, {
                    "prompt_tokens": token_info.get("input", 0),
                    "completion_tokens": token_info.get("output", 0),
                }, True

            results, usages, already_logged = self._batch_generate_with_retry(
                prompts=prompts,
                system_prompt=None,
                max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=SUMMARIZATION_GUIDED_JSON,
                sequential_retry_fn=retry_single,
            )

            for job, prompt, result, usage, logged in zip(chunk, prompts, results, usages, already_logged):
                if not logged and job["module"].memory_graph._llm_logger is not None:
                    job["module"].memory_graph._llm_logger.log("call_2_summarization", "", prompt, result)

                phase1_stats[job["sample_idx"]]["summ_input"] += usage["prompt_tokens"]
                phase1_stats[job["sample_idx"]]["summ_output"] += usage["completion_tokens"]
                phase1_stats[job["sample_idx"]]["summ_calls"] += 1

                job["new_nodes"] = job["module"].memory_graph.apply_summarize_result(
                    summarize_result=result,
                    finalize_idx=job["finalize_idx"],
                    sample_id=job["sample_id"],
                    session_ids=job["session_ids"],
                    dia_ids=job["dia_ids"],
                    date=job["date"],
                    source_session_dialogue=job["full_dialogue"],
                )
                job["module"].memory_graph.embed_nodes(job["new_nodes"])

        relation_jobs = []
        for job in finalize_jobs:
            if not job["new_nodes"]:
                continue
            for rel_job in job["module"].memory_graph.build_relation_jobs(job["new_nodes"]):
                rel_job["finalize_job"] = job
                relation_jobs.append(rel_job)

        for start in tqdm(range(0, len(relation_jobs), cfg.RELATION_BATCH_SIZE), desc="Phase1 relations"):
            chunk = relation_jobs[start:start + cfg.RELATION_BATCH_SIZE]
            prompts = [job["prompt"] for job in chunk]

            def retry_single(local_idx: int, _prompt: str):
                rel_job = chunk[local_idx]
                relation, token_info = rel_job["finalize_job"]["module"].memory_graph._extract_relation(
                    sentence1=rel_job["past_node"].summary,
                    sentence2=rel_job["new_node"].summary,
                    dialogue1=rel_job["past_node"].source_session_dialogue,
                    dialogue2=rel_job["new_node"].source_session_dialogue,
                )
                return {"relation": relation}, {
                    "prompt_tokens": token_info.get("input", 0),
                    "completion_tokens": token_info.get("output", 0),
                }, True

            results, usages, already_logged = self._batch_generate_with_retry(
                prompts=prompts,
                system_prompt=None,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=RELATION_GUIDED_JSON,
                sequential_retry_fn=retry_single,
            )

            for rel_job, prompt, result, usage, logged in zip(chunk, prompts, results, usages, already_logged):
                finalize_job = rel_job["finalize_job"]
                if not logged and finalize_job["module"].memory_graph._llm_logger is not None:
                    finalize_job["module"].memory_graph._llm_logger.log("call_3_relation", "", prompt, result)

                phase1_stats[finalize_job["sample_idx"]]["rel_input"] += usage["prompt_tokens"]
                phase1_stats[finalize_job["sample_idx"]]["rel_output"] += usage["completion_tokens"]
                phase1_stats[finalize_job["sample_idx"]]["rel_calls"] += 1

                finalize_job["relation_results"].append({
                    "new_node_id": rel_job["new_node_id"],
                    "past_node": rel_job["past_node"],
                    "relation": result.get("relation", "None") if isinstance(result, dict) else "None",
                })

        for job in finalize_jobs:
            if job["new_nodes"]:
                relation_meta = job["module"].memory_graph.apply_relation_results(
                    job["new_nodes"], job["relation_results"]
                )
                job["module"].memory_graph.register_new_nodes(job["new_nodes"])
            else:
                relation_meta = {"linked_edges": 0}

            logger.info(
                f"finalize_conv: finalize_idx={job['finalize_idx']}, sample_id={job['sample_id']}, "
                f"sessions={job['session_ids']}, nodes_created={len(job['new_nodes'])}, "
                f"linked_edges={relation_meta['linked_edges']}"
            )

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        samples: List[Sample],
        modules,
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        from timeline import REFINEMENT_SCHEMA

        qa_jobs = []
        phase2_stats = [
            {
                "refine_input": 0,
                "refine_output": 0,
                "num_refine_calls": 0,
                "qa_input": 0,
                "qa_output": 0,
                "num_qa_calls": 0,
            }
            for _ in samples
        ]

        for sample_idx, (sample, module) in enumerate(zip(samples, modules)):
            qa_current_dialogue = "\n".join(
                f"{turn.speaker}: {turn.text}"
                for turn in (sample.sessions[-1].turns if sample.sessions else [])
            )
            qa_dialogue_turns = sum(1 for line in qa_current_dialogue.split("\n") if line.strip())

            for qa_idx, qa in enumerate(sample.qa):
                timelines, log_items = module.retrieve_for_response(qa.question)

                retrieved_memories = [
                    {
                        "dia_ids": item["source"]["dia_ids"],
                        "content_preview": item["content_preview"],
                    }
                    for item in log_items
                ]
                use_timelines = timelines.get("use_timeline", [])
                path_texts = [
                    module.timeline.get_path_text(path, module.memory_graph)
                    for path in use_timelines
                ]

                qa_jobs.append({
                    "sample_idx": sample_idx,
                    "sample_id": sample.sample_id,
                    "qa_idx": qa_idx,
                    "qa": qa,
                    "module": module,
                    "timelines": timelines,
                    "log_items": log_items,
                    "retrieved_memories": retrieved_memories,
                    "path_texts": path_texts,
                    "refined_texts": [None] * len(path_texts),
                    "qa_current_dialogue": qa_current_dialogue,
                    "qa_dialogue_turns": qa_dialogue_turns,
                })

        qa_results_per_sample = [[] for _ in samples]
        if not qa_jobs:
            return qa_results_per_sample, phase2_stats

        refine_jobs = []
        for qa_job in qa_jobs:
            for path_idx, path_text in enumerate(qa_job["path_texts"]):
                refine_jobs.append({
                    "qa_job": qa_job,
                    "path_idx": path_idx,
                    "path_text": path_text,
                    "prompt": qa_job["module"].timeline.build_refine_prompt(
                        path_text, qa_job["qa_current_dialogue"], question=qa_job["qa"].question
                    ),
                })

        for start in tqdm(range(0, len(refine_jobs), cfg.REFINE_BATCH_SIZE), desc="Phase2 refine"):
            chunk = refine_jobs[start:start + cfg.REFINE_BATCH_SIZE]
            prompts = [job["prompt"] for job in chunk]

            def retry_single(local_idx: int, _prompt: str):
                refine_job = chunk[local_idx]
                text, token_info = refine_job["qa_job"]["module"].timeline.refine_timeline(
                    refine_job["path_text"],
                    refine_job["qa_job"]["qa_current_dialogue"],
                    question=refine_job["qa_job"]["qa"].question,
                )
                return {"refined_text": text}, {
                    "prompt_tokens": token_info.get("input", 0),
                    "completion_tokens": token_info.get("output", 0),
                }, True

            results, usages, already_logged = self._batch_generate_with_retry(
                prompts=prompts,
                system_prompt=None,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=REFINEMENT_SCHEMA,
                sequential_retry_fn=retry_single,
            )

            for refine_job, result, usage, logged in zip(chunk, results, usages, already_logged):
                qa_job = refine_job["qa_job"]
                if not logged and qa_job["module"].timeline._llm_logger is not None:
                    qa_job["module"].timeline._llm_logger.log(
                        "call_1_refinement",
                        "",
                        refine_job["prompt"],
                        result,
                    )

                qa_job["refined_texts"][refine_job["path_idx"]] = (
                    result.get("refined_text", refine_job["path_text"])
                    if isinstance(result, dict) else refine_job["path_text"]
                )

                phase2_stats[qa_job["sample_idx"]]["refine_input"] += usage["prompt_tokens"]
                phase2_stats[qa_job["sample_idx"]]["refine_output"] += usage["completion_tokens"]
                phase2_stats[qa_job["sample_idx"]]["num_refine_calls"] += 1

        grouped_jobs: Dict[Tuple[float, str], List[Dict]] = {}
        for qa_job in qa_jobs:
            qa_job["refined_texts"] = [
                text for text in qa_job["refined_texts"]
                if text is not None
            ]
            prompt, temperature, schema = qa_job["module"].generator.build_qa_prompt(
                question=qa_job["qa"].question,
                refined_texts=qa_job["refined_texts"],
                category=qa_job["qa"].category,
                current_dialogue=qa_job["qa_current_dialogue"],
                adversarial_answer=qa_job["qa"].adversarial_answer or "",
                choice_order_seed=f"{qa_job['sample_id']}::{qa_job['qa_idx']}::{qa_job['qa'].question}",
            )
            qa_job["qa_prompt"] = prompt
            qa_job["qa_temperature"] = temperature
            qa_job["qa_schema"] = schema
            key = (temperature, json.dumps(schema, sort_keys=True))
            grouped_jobs.setdefault(key, []).append(qa_job)

        qa_outputs = []
        for (temperature, _schema_key), jobs in grouped_jobs.items():
            for start in tqdm(range(0, len(jobs), cfg.QA_BATCH_SIZE), desc=f"Phase2 QA temp={temperature}"):
                chunk = jobs[start:start + cfg.QA_BATCH_SIZE]
                prompts = [job["qa_prompt"] for job in chunk]
                schema = chunk[0]["qa_schema"]

                def retry_single(local_idx: int, _prompt: str):
                    qa_job = chunk[local_idx]
                    answer, token_info, prompt_used = qa_job["module"].generator.generate_qa_answer(
                        question=qa_job["qa"].question,
                        refined_texts=qa_job["refined_texts"],
                        category=qa_job["qa"].category,
                        current_dialogue=qa_job["qa_current_dialogue"],
                        adversarial_answer=qa_job["qa"].adversarial_answer or "",
                        choice_order_seed=f"{qa_job['sample_id']}::{qa_job['qa_idx']}::{qa_job['qa'].question}",
                    )
                    return {"answer": answer}, {
                        "prompt_tokens": token_info.get("input", 0),
                        "completion_tokens": token_info.get("output", 0),
                    }, True

                results, usages, already_logged = self._batch_generate_with_retry(
                    prompts=prompts,
                    system_prompt=None,
                    max_tokens=cfg.MAX_TOKENS,
                    temperature=temperature,
                    guided_json=schema,
                    sequential_retry_fn=retry_single,
                )

                for qa_job, result, usage, logged in zip(chunk, results, usages, already_logged):
                    qa_outputs.append((qa_job, result, usage, logged))

        qa_outputs.sort(key=lambda item: (item[0]["sample_idx"], item[0]["qa_idx"]))

        for qa_job, result, usage, logged in qa_outputs:
            sample_idx = qa_job["sample_idx"]
            if not logged and qa_job["module"].generator._llm_logger is not None:
                qa_job["module"].generator._llm_logger.log(
                    "call_4_qa",
                    "",
                    qa_job["qa_prompt"],
                    result,
                )

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            phase2_stats[sample_idx]["qa_input"] += usage["prompt_tokens"]
            phase2_stats[sample_idx]["qa_output"] += usage["completion_tokens"]
            phase2_stats[sample_idx]["num_qa_calls"] += 1

            write_retrieval_log(
                retrieval_log_paths[sample_idx],
                build_retrieval_log_entry(
                    phase="qa",
                    sample_id=qa_job["sample_id"],
                    query=qa_job["qa"].question,
                    retrieved_items=qa_job["log_items"],
                    use_timelines=qa_job["timelines"].get("use_timeline", []),
                    timeline_info=qa_job["timelines"].get("timeline", []),
                    current_dialogue_turns=qa_job["qa_dialogue_turns"],
                ),
            )

            qa_results_per_sample[sample_idx].append({
                "question": qa_job["qa"].question,
                "category": qa_job["qa"].category,
                "generated_answer": answer,
                "ground_truth_answer": qa_job["qa"].final_answer,
                "evidence": qa_job["qa"].evidence,
                "retrieved_memories": qa_job["retrieved_memories"],
                "qa_tokens": {
                    "input": usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "model": self.model_path,
                },
            })

        return qa_results_per_sample, phase2_stats

    # ------------------------------------------------------------------
    # Batch LLM helper
    # ------------------------------------------------------------------

    def _batch_generate_with_retry(
        self,
        prompts: List[str],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        guided_json=None,
        sequential_retry_fn=None,
    ) -> Tuple[List, List[Dict], List[bool]]:
        if not prompts:
            return [], [], []

        try:
            texts, usages = self.llm_client.generate_batch_raw(
                prompts=prompts,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                guided_json=guided_json,
                return_usage=True,
            )
        except Exception as e:
            if sequential_retry_fn is None:
                raise
            logger.warning(f"Batch generation failed; falling back to sequential retry for chunk: {e}")
            results = []
            usages = []
            already_logged = []
            for idx, prompt in enumerate(prompts):
                result, usage, logged = sequential_retry_fn(idx, prompt)
                results.append(result)
                usages.append(usage)
                already_logged.append(logged)
            return results, usages, already_logged

        already_logged = [False] * len(prompts)
        if guided_json is None:
            return texts, usages, already_logged

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
            logger.warning(f"Batch item {idx} failed JSON parse; retrying sequentially")
            if sequential_retry_fn is not None:
                result, usage, logged = sequential_retry_fn(idx, prompts[idx])
                parsed[idx] = result
                usages[idx] = usage
                already_logged[idx] = logged
                continue
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
                usage_info = retry_result.pop("_usage", {}) if isinstance(retry_result, dict) else {}
                usages[idx] = {
                    "prompt_tokens": usage_info.get("prompt_tokens", 0),
                    "completion_tokens": usage_info.get("completion_tokens", 0),
                }
                parsed[idx] = retry_result if isinstance(retry_result, dict) else {}
            except Exception as e:
                logger.error(f"Sequential retry for batch item {idx} failed: {e}")
                parsed[idx] = {}

        return parsed, usages, already_logged


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

    from theanine_module import TheanineModule  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Theanine Batch Experiment on LoComo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-sample", type=int, required=True)
    parser.add_argument("--end-sample", type=int, required=True)
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int, default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float, default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

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

    results_file = cfg.get_results_file(args.model, args.start_sample, args.end_sample, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.start_sample, args.end_sample, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.start_sample, args.end_sample, config_name)
    snapshots_dir = cfg.get_memory_snapshots_dir(args.model, args.start_sample, args.end_sample, config_name)
    prompt_log_dir = cfg.get_prompt_log_dir(args.model, args.start_sample, args.end_sample, config_name)

    logger.info("=" * 60)
    logger.info("Theanine Batch Experiment — LoComo")
    logger.info(f"  Config               : {config_name}")
    logger.info(f"  Model                : {args.model}")
    logger.info(f"  Samples              : [{args.start_sample}, {args.end_sample}]")
    logger.info(f"  Batch size           : {args.batch_size}")
    logger.info(f"  Summarize batch size : {cfg.SUMMARIZE_BATCH_SIZE}")
    logger.info(f"  Relation batch size  : {cfg.RELATION_BATCH_SIZE}")
    logger.info(f"  Refine batch size    : {cfg.REFINE_BATCH_SIZE}")
    logger.info(f"  QA batch size        : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model      : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Linking top_j        : {cfg.LINKING_TOP_J}")
    logger.info(f"  Retrieve top_k       : {cfg.RETRIEVE_TOP_K}")
    logger.info(f"  Save snapshots       : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  LLM call logging     : {cfg.ENABLE_LLM_CALL_LOGGING}")
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

    pending_samples = [sample for sample in target_samples if str(sample.sample_id) not in completed_sample_ids]
    if not pending_samples:
        logger.info("All samples already completed.")
        return 0

    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer

    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(args.model, args.tensor_parallel, args.gpu_memory, max_model_len=args.max_model_len)
    logger.info("LLM client ready.")

    config_metadata = {
        "config_name": config_name,
        "model": args.model,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "temperature": cfg.TEMPERATURE,
        "temperature_c5": cfg.TEMPERATURE_C5,
        "max_tokens": cfg.MAX_TOKENS,
        "sample_range": [args.start_sample, args.end_sample],
        "retrieve_top_k": cfg.RETRIEVE_TOP_K,
        "linking_top_j": cfg.LINKING_TOP_J,
        "timeline_sample_n": cfg.TIMELINE_SAMPLE_N,
    }

    runner = BatchedTheanineRunner(
        llm_client=llm_client,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
        config_metadata=config_metadata,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# Theanine Batch  |  LoComo")
    print(f"# Model   : {cfg.extract_model_name(args.model)}")
    print(f"# Samples [{args.start_sample}, {args.end_sample}]  ({len(pending_samples)} to process)")
    print(f"# Batch size : {args.batch_size}")
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
                prompt_log_dirs=batch_prompt_log_dirs,
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
