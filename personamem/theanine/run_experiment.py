"""
Theanine Experiment Runner — PersonaMem (Batched, QA-Only)

Runs multiple PersonaMem shared contexts in parallel within a single GPU by
batching their LLM calls together while preserving each context's internal
ordering.

Internal field names (`session_id`, `conv_id`, `turn_id`, `c{conv_id}-m{idx}`)
inside memory_graph / timeline / theanine_module are retained from the
ImplexConv variant. The runner injects PersonaMem identifiers into those slots
(`context_index` → session_id, `block_idx` → conv_id, `local_msg_idx` →
turn_id) and remaps result metadata via `_remap_retrieved_metadata`.

Boundary detection: PersonaMem messages already carry `block_idx`. We trigger
finalize whenever block_idx changes (with `FINALIZE_EVERY_N_CONVS=1`).
"""

import os
import sys
import json
import logging
import argparse
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
    PersonaMemQAPair,
)


# =============================================================================
# LOGGING
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
        if last is not None:
            return set(range(last + 1))
        return set()
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def save_checkpoint(
    checkpoint_file: Path,
    completed_ids: Set[int],
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config",
):
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
    local_msg_idx: int,
    query: str,
    retrieved_items: List[Dict],
    use_timelines: Optional[List] = None,
    timeline_info: Optional[List[Dict]] = None,
    current_dialogue_messages: int = 0,
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
        "context_index": context_index,
        "block_idx": block_idx,
        "local_msg_idx": local_msg_idx,
        "query": query[:500],
        "memory_type": ["seed", "path_linked", "current_dialogue"],
        "num_retrieved": [len(retrieved_items), num_path_linked, current_dialogue_messages],
        "retrieved_items": retrieved_items,
        "retrieval_scores": [item["score"] for item in retrieved_items],
        "module_specific": {
            "module": "theanine",
            "num_paths_used": len(use_timelines) if use_timelines else 0,
            "use_timelines": [list(p) for p in use_timelines] if use_timelines else [],
            "timeline_info": timeline_info or [],
        },
    }


def _remap_retrieved_memories(log_items: List[Dict]) -> List[Dict]:
    """Remap memory_layer field keys (session_id/conv_id/turn_id_start) to
    PersonaMem keys (context_index/block_idx/local_msg_idx)."""
    remapped = []
    for item in log_items:
        src = item.get("source_turn", {})
        remapped.append({
            "context_index": src.get("session_id", -1),
            "block_idx":     src.get("conv_id",    -1),
            "local_msg_idx": src.get("turn_id_start", -1),
        })
    return remapped


# =============================================================================
# DIALOGUE / BLOCK HELPERS
# =============================================================================

def build_block_dialogue(messages: List[PersonaMemMessage]) -> str:
    """Concatenate block messages — content already includes 'User: '/'Assistant: '."""
    return "\n".join(m.content for m in messages)


# =============================================================================
# BATCHED RUNNER
# =============================================================================

class BatchedTheanineRunner:
    def __init__(self, llm_client, benchmark_size: str, model_path: str,
                 shared_embedding_model, config_metadata: Optional[Dict] = None):
        self.llm_client = llm_client
        self.benchmark_size = benchmark_size
        self.model_path = model_path
        self.shared_embedding_model = shared_embedding_model
        self.config_metadata = config_metadata or {}

    def run_batch(
        self,
        contexts: List[PersonaMemContext],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
    ) -> List[Dict]:
        from theanine_module import TheanineModule, LLMCallLogger

        modules = [
            TheanineModule(
                self.llm_client,
                model_path=self.model_path,
                embedding_model=self.shared_embedding_model,
            )
            for _ in contexts
        ]

        for module, prompt_log_dir in zip(modules, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                module.set_llm_logger(LLMCallLogger(prompt_log_dir))

        phase1_stats, final_dialogues = self._run_phase1_batched(contexts, modules)

        memory_at_qa_start_list = [module.get_memory_stats() for module in modules]

        qa_results_list, phase2_stats = self._run_phase2_batched(
            contexts, modules, retrieval_log_paths, final_dialogues
        )

        results = []
        for i, (context, module) in enumerate(zip(contexts, modules)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"session_{context.context_index}"
                module.save_memory_snapshot(snapshot_dir)

            module.clear()

            p1 = phase1_stats[i]
            p2 = phase2_stats[i]
            token_stats = {
                "call_3_summarization": {
                    "input":               p1["summ_input"],
                    "output":              p1["summ_output"],
                    "llm_calls":           p1["summ_calls"],
                    "parse_fallback_count": p1["summ_fallback"],
                },
                "call_4_relation": {
                    "input":     p1["rel_input"],
                    "output":    p1["rel_output"],
                    "llm_calls": p1["rel_calls"],
                },
                "call_2_refinement": {
                    "input":     p2["refine_input"],
                    "output":    p2["refine_output"],
                    "llm_calls": p2["num_refine_calls"],
                },
                "call_5_qa": {
                    "input":     p2["qa_prompt_input"],
                    "output":    p2["qa_output"],
                    "llm_calls": p2["num_qa_calls"],
                },
                "qa_input":        p2["qa_input"],
                "total_input":     (p1["summ_input"] + p1["rel_input"]
                                    + p2["qa_input"]),
                "total_output":    (p1["summ_output"] + p1["rel_output"]
                                    + p2["refine_output"] + p2["qa_output"]),
                "total_llm_calls": (p1["summ_calls"] + p1["rel_calls"]
                                    + p2["num_refine_calls"] + p2["num_qa_calls"]),
            }
            results.append({
                "context_index":      context.context_index,
                "persona_id":         context.persona_id,
                "shared_context_id":  context.shared_context_id,
                "config_metadata":    self.config_metadata,
                "memory_at_qa_start": memory_at_qa_start_list[i],
                "qa_results":         qa_results_list[i],
                "token_statistics":   token_stats,
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
    ) -> Tuple[List[Dict], List[str]]:
        chunk_factor = cfg.CHUNK_FACTOR.get(self.benchmark_size, 1)
        block_sizes_list = []
        for context in contexts:
            bs: Dict[int, int] = {}
            for msg in context.messages:
                bs[msg.block_idx] = bs.get(msg.block_idx, 0) + 1
            block_sizes_list.append(bs)

        states = [
            {
                "prev_block_idx":          None,
                "prev_virtual_block_idx":  None,
                "current_day":             -1,
                "current_dialogue":        "",
                "current_block_messages":  [],   # for current_dialogue fallback
                "prev_block_messages":     [],   # one-back fallback (Q24=A)
                "finalize_messages":       [],
                "global_msg_counter":      0,    # Q23=C: assigned per message in order
            }
            for _ in contexts
        ]
        phase1_stats = [
            {
                "summ_input": 0, "summ_output": 0, "summ_calls": 0, "summ_fallback": 0,
                "rel_input":  0, "rel_output":  0, "rel_calls":  0,
            }
            for _ in contexts
        ]

        max_messages = max(len(c.messages) for c in contexts)

        for msg_idx in tqdm(range(max_messages), desc="Phase1 messages"):
            boundary_jobs = []
            active_indices = [
                i for i, c in enumerate(contexts) if msg_idx < len(c.messages)
            ]

            # Step 1: detect virtual-block transitions and emit finalize jobs
            for i in active_indices:
                msg = contexts[i].messages[msg_idx]
                state = states[i]
                bsize = block_sizes_list[i].get(msg.block_idx, 1)
                sub = min(msg.local_msg_idx * chunk_factor // bsize, chunk_factor - 1)
                virtual_block_idx = msg.block_idx * chunk_factor + sub
                if (state["prev_virtual_block_idx"] is not None
                        and virtual_block_idx != state["prev_virtual_block_idx"]
                        and state["finalize_messages"]):
                    boundary_jobs.append(
                        self._build_finalize_job(
                            i,
                            contexts[i].context_index,
                            modules[i],
                            state["prev_virtual_block_idx"],
                            state["finalize_messages"],
                        )
                    )
                    state["finalize_messages"] = []

            # Step 2: run finalize jobs (batched across contexts)
            self._run_finalize_jobs(boundary_jobs)

            # Step 3: per-message bookkeeping
            for i in active_indices:
                msg = contexts[i].messages[msg_idx]
                state = states[i]

                new_day = msg.block_idx // cfg.CONV_IDS_PER_DAY
                if new_day != state["current_day"]:
                    # Closing the previous block: snapshot it for fallback
                    if state["current_block_messages"]:
                        state["prev_block_messages"] = list(state["current_block_messages"])
                    state["current_dialogue"] = ""
                    state["current_block_messages"] = []
                    state["current_day"] = new_day

                state["current_dialogue"] += msg.content + "\n"
                state["current_block_messages"].append(msg)
                state["finalize_messages"].append(msg)
                state["prev_block_idx"] = msg.block_idx
                bsize = block_sizes_list[i].get(msg.block_idx, 1)
                sub = min(msg.local_msg_idx * chunk_factor // bsize, chunk_factor - 1)
                state["prev_virtual_block_idx"] = msg.block_idx * chunk_factor + sub

                # Q23=C: global message index assigned in iteration order
                msg._theanine_global_msg_idx = state["global_msg_counter"]
                state["global_msg_counter"] += 1

        # Phase 1 final flush
        final_jobs = []
        for i, context in enumerate(contexts):
            state = states[i]
            if state["finalize_messages"]:
                final_jobs.append(
                    self._build_finalize_job(
                        i,
                        context.context_index,
                        modules[i],
                        state["prev_virtual_block_idx"],
                        state["finalize_messages"],
                    )
                )
        self._run_finalize_jobs(final_jobs)

        # Q24=A: if last block dialogue is empty, fall back to the previous block
        final_dialogues = []
        for state in states:
            if state["current_dialogue"].strip():
                final_dialogues.append(state["current_dialogue"])
            elif state["prev_block_messages"]:
                final_dialogues.append(build_block_dialogue(state["prev_block_messages"]) + "\n")
            else:
                final_dialogues.append("")

        for i, module in enumerate(modules):
            mem_tokens = module.get_and_reset_memory_tokens()
            phase1_stats[i]["summ_input"]    += mem_tokens["call_3_summarization"]["input"]
            phase1_stats[i]["summ_output"]   += mem_tokens["call_3_summarization"]["output"]
            phase1_stats[i]["summ_calls"]    += mem_tokens["call_3_summarization"]["llm_calls"]
            phase1_stats[i]["summ_fallback"] += mem_tokens["call_3_summarization"]["parse_fallback_count"]
            phase1_stats[i]["rel_input"]     += mem_tokens["call_4_relation"]["input"]
            phase1_stats[i]["rel_output"]    += mem_tokens["call_4_relation"]["output"]
            phase1_stats[i]["rel_calls"]     += mem_tokens["call_4_relation"]["llm_calls"]

        return phase1_stats, final_dialogues

    def _build_finalize_job(
        self,
        batch_idx: int,
        context_index: int,
        module,
        block_idx: int,
        finalize_messages: List[PersonaMemMessage],
    ) -> Dict:
        msgs = list(finalize_messages)
        return {
            "batch_idx":            batch_idx,
            "context_index":        context_index,
            "module":               module,
            "block_idx":            block_idx,
            "full_block_dialogue":  build_block_dialogue(msgs),
            "local_msg_idx_start":  msgs[0].local_msg_idx,
            "local_msg_idx_end":    msgs[-1].local_msg_idx,
            "global_msg_idx_start": getattr(msgs[0],  "_theanine_global_msg_idx", -1),
            "global_msg_idx_end":   getattr(msgs[-1], "_theanine_global_msg_idx", -1),
        }

    def _run_finalize_jobs(self, finalize_jobs: List[Dict]):
        if not finalize_jobs:
            return

        from memory_graph import _SUMMARIZATION_GUIDED_JSON, _RELATION_GUIDED_JSON

        # Step A: batch summarization
        summarize_prompts = [
            job["module"].memory_graph.build_summarize_prompt(job["full_block_dialogue"])
            for job in finalize_jobs
        ]
        summarize_results, summarize_usages = self._batch_generate_with_retry(
            prompts=summarize_prompts,
            max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            guided_json=_SUMMARIZATION_GUIDED_JSON,
            batch_limit=cfg.SUMMARIZE_BATCH_SIZE,
        )

        for job, result, usage, prompt in zip(
            finalize_jobs, summarize_results, summarize_usages, summarize_prompts
        ):
            module = job["module"]
            graph = module.memory_graph
            graph.accumulate_summarize_usage(
                usage["prompt_tokens"], usage["completion_tokens"], 1
            )
            if graph._llm_logger is not None:
                graph._llm_logger.log("call_3_summarization", "", prompt, result)

            used_fallback = not (isinstance(result, dict) and "sentences" in result)
            if used_fallback:
                graph.accumulate_fallback_count(1)
                try:
                    sentences, token_info, fb_prompt, fb_result = (
                        graph.summarize_plaintext_fallback(job["full_block_dialogue"])
                    )
                    graph.accumulate_summarize_usage(
                        token_info["input"], token_info["output"], 1
                    )
                    if graph._llm_logger is not None:
                        graph._llm_logger.log("call_3_summarization", "", fb_prompt, fb_result)
                except Exception as e:
                    logger.warning(
                        f"summarize fallback failed for context {job['context_index']} "
                        f"block {job['block_idx']}: {e}"
                    )
                    sentences = []
            else:
                sentences = graph.parse_summarize_result(result)

            # Inject PersonaMem identifiers into the underlying memory_graph slots:
            #   conv_id    ← block_idx
            #   session_id ← context_index
            #   turn_id_*  ← local_msg_idx_*
            #   global_turn_id_* ← global_msg_idx_*
            job["new_nodes"] = graph.prepare_new_nodes(
                sentences=sentences,
                conv_id=job["block_idx"],
                session_id=job["context_index"],
                source_conv_dialogue=job["full_block_dialogue"],
                turn_id_start=job["local_msg_idx_start"],
                turn_id_end=job["local_msg_idx_end"],
                global_turn_id_start=job["global_msg_idx_start"],
                global_turn_id_end=job["global_msg_idx_end"],
            ) if sentences else []

        # Step B: batch relation extraction
        relation_items = []
        for job in finalize_jobs:
            relation_jobs = job["module"].memory_graph.build_relation_jobs(
                job.get("new_nodes", [])
            )
            job["relation_jobs"] = relation_jobs
            job["relation_results"] = []
            for relation_job in relation_jobs:
                relation_items.append((job, relation_job))

        if relation_items:
            relation_prompts = [rj["prompt"] for _, rj in relation_items]
            relation_results, relation_usages = self._batch_generate_with_retry(
                prompts=relation_prompts,
                max_tokens=cfg.MAX_TOKENS,
                guided_json=_RELATION_GUIDED_JSON,
                batch_limit=cfg.RELATION_BATCH_SIZE,
            )
            for (job, relation_job), result, usage in zip(
                relation_items, relation_results, relation_usages
            ):
                graph = job["module"].memory_graph
                graph.accumulate_relation_usage(
                    usage["prompt_tokens"], usage["completion_tokens"], 1
                )
                if graph._llm_logger is not None:
                    graph._llm_logger.log("call_4_relation", "", relation_job["prompt"], result)
                job["relation_results"].append(graph.parse_relation_result(result))

        # Step C: apply relations and commit nodes
        for job in finalize_jobs:
            graph = job["module"].memory_graph
            link_info = graph.apply_relation_results(
                job.get("new_nodes", []),
                job.get("relation_jobs", []),
                job.get("relation_results", []),
            )
            logger.info(
                f"finalize_block(batch): context_index={job['context_index']}, "
                f"block_idx={job['block_idx']}, "
                f"nodes_created={len(job.get('new_nodes', []))}, "
                f"relation_calls={link_info['relation_calls']}, "
                f"linked_edges={link_info['linked_edges']}"
            )

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        contexts: List[PersonaMemContext],
        modules,
        retrieval_log_paths: List[Path],
        final_dialogues: List[str],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        from generator import QA_SCHEMA_MULTICHOICE
        from timeline import _REFINEMENT_GUIDED_JSON

        qa_jobs = []
        refine_items = []
        phase2_stats = [
            {
                "qa_input": 0,
                "qa_prompt_input": 0,
                "qa_output": 0,
                "num_qa_calls": 0,
                "refine_input": 0,
                "refine_output": 0,
                "num_refine_calls": 0,
            }
            for _ in contexts
        ]

        for i, (context, module) in enumerate(zip(contexts, modules)):
            for qa in context.qa_pairs:
                qa_current_dialogue = final_dialogues[i].strip()
                timelines, log_items = module.retrieve(qa.question)

                retrieved_memories = _remap_retrieved_memories(log_items)

                qa_current_dialogue_messages = sum(
                    1 for line in qa_current_dialogue.split("\n") if line.strip()
                )
                write_retrieval_log(retrieval_log_paths[i], build_retrieval_log_entry(
                    phase="qa",
                    context_index=context.context_index,
                    block_idx=-1,
                    local_msg_idx=-1,
                    query=qa.question,
                    retrieved_items=log_items,
                    use_timelines=timelines.get("use_timeline", []),
                    timeline_info=timelines.get("timeline", []),
                    current_dialogue_messages=qa_current_dialogue_messages,
                ))

                qa_job = {
                    "batch_idx": i,
                    "qa": qa,
                    "module": module,
                    "qa_current_dialogue": qa_current_dialogue,
                    "retrieved_memories": retrieved_memories,
                    "refined_texts": [],
                }
                qa_jobs.append(qa_job)

                for path in timelines.get("use_timeline", []):
                    path_text = module.timeline.get_path_text(path, module.memory_graph)
                    refine_items.append({
                        "qa_job": qa_job,
                        "module": module,
                        "path_text": path_text,
                        "prompt": module.timeline.build_refine_prompt(
                            path_text,
                            qa_current_dialogue,
                            qa.question,
                        ),
                    })

        # Refinement (batched)
        if refine_items:
            refine_prompts = [item["prompt"] for item in refine_items]
            refine_results, refine_usages = self._batch_generate_with_retry(
                prompts=refine_prompts,
                max_tokens=cfg.MAX_TOKENS,
                guided_json=_REFINEMENT_GUIDED_JSON,
                batch_limit=cfg.REFINE_BATCH_SIZE,
            )
            for item, result, usage in zip(refine_items, refine_results, refine_usages):
                module = item["module"]
                module.timeline.accumulate_usage(
                    usage["prompt_tokens"], usage["completion_tokens"], 1
                )
                if module.timeline._llm_logger is not None:
                    module.timeline._llm_logger.log("call_2_refinement", "", item["prompt"], result)
                item["qa_job"]["refined_texts"].append(
                    module.timeline.parse_refine_result(result, item["path_text"])
                )

        # QA prompt building
        qa_prompts = []
        for qa_job in qa_jobs:
            prompt, _ = qa_job["module"].generator.build_qa_prompt(
                question=qa_job["qa"].question,
                retrieved_summaries=qa_job["refined_texts"],
                options=qa_job["qa"].all_options,
                current_dialogue=qa_job["qa_current_dialogue"],
            )
            qa_job["qa_prompt"] = prompt
            qa_prompts.append(prompt)

        # QA generation (batched)
        qa_results, qa_usages = self._batch_generate_with_retry(
            prompts=qa_prompts,
            max_tokens=cfg.MAX_TOKENS,
            guided_json=QA_SCHEMA_MULTICHOICE,
            batch_limit=cfg.QA_BATCH_SIZE,
        )

        qa_results_per_context: List[List[Dict]] = [[] for _ in contexts]
        for qa_job, result, usage in zip(qa_jobs, qa_results, qa_usages):
            i = qa_job["batch_idx"]
            module = qa_job["module"]
            if module.generator._llm_logger is not None:
                module.generator._llm_logger.log("call_5_qa", "", qa_job["qa_prompt"], result)

            answer = module.generator.parse_qa_result(result)
            phase2_stats[i]["qa_input"]        += usage["prompt_tokens"]
            phase2_stats[i]["qa_prompt_input"] += usage["prompt_tokens"]
            phase2_stats[i]["qa_output"]       += usage["completion_tokens"]
            phase2_stats[i]["num_qa_calls"]    += 1
            qa_results_per_context[i].append({
                "question":                    qa_job["qa"].question,
                "question_type":               qa_job["qa"].question_type,
                "topic":                       qa_job["qa"].topic,
                "all_options":                 qa_job["qa"].all_options,
                "generated_answer":            answer,
                "ground_truth_answer":         qa_job["qa"].correct_answer,
                "end_index_in_shared_context": qa_job["qa"].end_index_in_shared_context,
                "retrieved_memories":          qa_job["retrieved_memories"],
                "qa_tokens": {
                    "input":  usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "model":  self.model_path,
                },
            })

        # Drain refine token counters into per-context phase2_stats
        for i, module in enumerate(modules):
            refine_tokens = module.timeline.get_and_reset_token_counts()
            phase2_stats[i]["qa_input"]         += refine_tokens["input"]
            phase2_stats[i]["refine_input"]     += refine_tokens["input"]
            phase2_stats[i]["refine_output"]    += refine_tokens["output"]
            phase2_stats[i]["num_refine_calls"] += refine_tokens["llm_calls"]

        return qa_results_per_context, phase2_stats

    # ------------------------------------------------------------------
    # Batch helpers
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

    parser = argparse.ArgumentParser(
        description="Theanine Batch Experiment on PersonaMem (QA-Only)",
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
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

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
    logger.info("Theanine Batch Experiment — PersonaMem (QA-Only)")
    logger.info(f"  Config            : {config_name}")
    logger.info(f"  Benchmark size    : {args.benchmark_size}")
    logger.info(f"  Model             : {args.model}")
    logger.info(f"  Sessions          : [{args.start_session}, {args.end_session}]")
    logger.info(f"  Batch size        : {args.batch_size}")
    logger.info(f"  Summ batch size   : {cfg.SUMMARIZE_BATCH_SIZE}")
    logger.info(f"  Rel batch size    : {cfg.RELATION_BATCH_SIZE}")
    logger.info(f"  Refine batch size : {cfg.REFINE_BATCH_SIZE}")
    logger.info(f"  QA batch size     : {cfg.QA_BATCH_SIZE}")
    logger.info(f"  Embedding model   : {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Linking top_j     : {cfg.LINKING_TOP_J}")
    logger.info(f"  Retrieve top_k    : {cfg.RETRIEVE_TOP_K}")
    logger.info(f"  Save snapshots    : {cfg.SAVE_MEMORY_SNAPSHOTS}")
    logger.info(f"  Conv IDs/day      : {cfg.CONV_IDS_PER_DAY}")
    logger.info(f"  Minutes/turn      : {cfg.MINUTES_PER_TURN}")
    logger.info(f"  Finalize every    : {cfg.FINALIZE_EVERY_N_CONVS} block(s)")
    logger.info(f"  LLM call logging  : {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

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

    completed_ids: Set[int] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_ids = load_checkpoint(checkpoint_file)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} contexts already completed")

    pending_contexts = [
        c for c in target_contexts if c.context_index not in completed_ids
    ]
    if not pending_contexts:
        logger.info("All contexts already completed.")
        return 0

    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory,
        max_model_len=args.max_model_len,
    )
    logger.info("LLM client ready.")
    if args.max_model_len is not None:
        cfg.FINALIZE_INPUT_CONTEXT_LIMIT = args.max_model_len
        logger.info(f"FINALIZE_INPUT_CONTEXT_LIMIT set to {args.max_model_len} (from --max-model-len)")

    config_metadata = {
        "config_name":             config_name,
        "model":                   args.model,
        "benchmark_size":          args.benchmark_size,
        "embedding_model":         cfg.EMBEDDING_MODEL,
        "temperature":             cfg.TEMPERATURE,
        "max_tokens":              cfg.MAX_TOKENS,
        "session_range":           [args.start_session, args.end_session],
        "retrieve_top_k":          cfg.RETRIEVE_TOP_K,
        "linking_top_j":           cfg.LINKING_TOP_J,
        "timeline_sample_n":       cfg.TIMELINE_SAMPLE_N,
        "conv_ids_per_day":        cfg.CONV_IDS_PER_DAY,
        "finalize_every_n_convs":  cfg.FINALIZE_EVERY_N_CONVS,
    }

    runner = BatchedTheanineRunner(
        llm_client=llm_client,
        benchmark_size=args.benchmark_size,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
        config_metadata=config_metadata,
    )
    results = load_existing_results(results_file)

    print(f"\n{'#' * 60}")
    print(f"# Theanine Batch  |  benchmark_size={args.benchmark_size}")
    print(f"# Model : {cfg.extract_model_name(args.model)}")
    print(f"# Contexts [{args.start_session}, {args.end_session}]  ({len(pending_contexts)} to process)")
    print(f"# Batch size : {args.batch_size}")
    print(f"{'#' * 60}\n")

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
                    checkpoint_file,
                    completed_ids,
                    args.model,
                    args.benchmark_size,
                    args.start_session,
                    args.end_session,
                    config_name=config_name,
                )

            logger.info(
                f"Context {result['context_index']} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total LLM calls: {result['token_statistics']['total_llm_calls']}"
            )

    print(f"\n{'#' * 60}")
    print(f"# Batch experiment complete!  Results -> {results_file}")
    print(f"{'#' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
