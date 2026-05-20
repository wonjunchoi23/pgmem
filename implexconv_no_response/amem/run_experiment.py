"""
A-MEM Experiment Runner — ImplexConv (Batched Main Variant, QA-Only)

Runs multiple sessions in parallel within a single GPU by batching their LLM
calls together. Sessions are independent (separate memory states) so their
per-turn LLM prompts (analyze, evolve, QA) can be collected and submitted as
a single vLLM batch, dramatically improving GPU utilization for small models.

Batch execution flow:
  For each batch of N sessions (processed simultaneously):

  Phase 1 — Memory Construction (interleaved across sessions):
    For turn_idx = 0, 1, ..., max_turns:
      User turns across all N sessions:
        1. Retrieve memories (embedding, per-session)
        2. Write retrieval log
        3. Collect analyze prompts → [BATCH] vLLM generate
        4. apply_analyze_result → collect evolution prompts
        5. Store no-evolve notes immediately
        6. [BATCH] vLLM generate evolution prompts
        7. apply_evolve_result → store evolved notes
      Repeat for assistant turns.

  Phase 2 — QA Answering (all sessions batched together):
    1. Retrieve + build QA prompts for all sessions
    2. [BATCH] vLLM generate in QA_BATCH_SIZE chunks
    3. Distribute results

  Phase 3 — Cleanup:
    Save memory snapshots, clear memory, save results, update checkpoint.

Checkpoint format (new — set-based):
  {"completed_session_ids": [0, 1, 2, ...], ...}
  Old format {"last_completed_session_index": N} is auto-converted on load.

Usage:
    python run_experiment.py \\
        --start-session 0 --end-session 99 \\
        --subset opposed \\
        --model Qwen/Qwen3-1.7B \\
        --tensor-parallel 1 --gpu-memory 0.9 \\
        --batch-size 4 \\
        --config config_0
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from tqdm import tqdm

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

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
        log_file = log_dir / f"amem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path) -> Set[int]:
    """Load completed session IDs from checkpoint.

    Supports both new format (completed_session_ids list) and old format
    (last_completed_session_index int) for backward compatibility.
    """
    if not checkpoint_file.exists():
        return set()
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        if "completed_session_ids" in data:
            return set(data["completed_session_ids"])
        # Old format: convert to set
        last = data.get("last_completed_session_index")
        if last is not None:
            return set(range(last + 1))
        return set()
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
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# RETRIEVAL LOGGING
# =============================================================================

def write_retrieval_log(log_path: Path, entry: Dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    phase: str,
    session_id: int,
    conv_id: int,
    turn_id: int,
    query: str,
    retrieved_items: List[Dict],
    num_linked: int,
) -> Dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "session_id": session_id,
        "conv_id": conv_id,
        "turn_id": turn_id,
        "query": query,
        "memory_type":   ["direct", "linked"],
        "num_retrieved": [len(retrieved_items), num_linked],
        "retrieved_items": retrieved_items,
        "retrieval_scores": [item["score"] for item in retrieved_items],
        "module_specific": {
            "module": "amem",
            "num_direct_hits": len(retrieved_items),
            "num_linked_neighbors": num_linked,
        },
    }


# =============================================================================
# BATCHED AMEM RUNNER
# =============================================================================

class BatchedAMEMRunner:
    """
    Processes N sessions in parallel by batching their LLM calls.

    Key invariants:
    - Each session has its own BaseAgent (separate memory state).
    - All agents share one SentenceTransformer instance (shared_embedding_model).
    - Within Phase 1, sessions are interleaved turn-by-turn: all sessions'
      user turn prompts for turn_idx are batched together, then all assistant
      turn prompts, before moving to turn_idx+1.
    - Evolution notes are stored only AFTER the batch evolve call, so each
      session's neighbor search reflects the correct pre-turn memory state.
    """

    QA_SYSTEM_PROMPT = (
        "You are a helpful assistant answering a question about a user "
        "based on their conversation history stored in memory. "
        "Respond in JSON format with an 'answer' field."
    )

    def __init__(self, llm_client, subset: str, model_path: str, shared_embedding_model):
        self.llm_client = llm_client
        self.subset = subset
        self.model_path = model_path
        self.shared_embedding_model = shared_embedding_model

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_batch(
        self,
        sessions: List[Session],
        retrieval_log_paths: List[Path],
        snapshots_dir: Path,
        prompt_log_dirs: List[Path],
        config_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """Process a batch of sessions in parallel. Returns one result dict per session."""
        from agent import BaseAgent
        from memory_layer import LLMWrapper

        # Create per-session agents (shared embedding model)
        agents = [
            BaseAgent(self.llm_client, self.model_path,
                      embedding_model=self.shared_embedding_model)
            for _ in sessions
        ]

        # Set per-session LLM call loggers
        for agent, prompt_log_dir in zip(agents, prompt_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                agent.set_llm_logger(LLMCallLogger(prompt_log_dir))

        # Phase 1: memory construction (interleaved)
        phase1_stats = self._run_phase1_batched(sessions, agents)

        # Collect memory state + internal stats before Phase 2
        memory_stats_list = [agent.get_memory_stats() for agent in agents]
        internal_stats_list = [agent.get_and_reset_internal_stats() for agent in agents]

        # Phase 2: QA answering (all sessions batched)
        qa_results_list, phase2_stats = self._run_phase2_batched(sessions, agents, retrieval_log_paths)

        # Phase 3: cleanup + aggregate results
        results = []
        for i, (session, agent) in enumerate(zip(sessions, agents)):
            if cfg.SAVE_MEMORY_SNAPSHOTS:
                snapshot_dir = snapshots_dir / f"session_{session.session_id}"
                agent.save_memory_snapshot(snapshot_dir)
                logger.info(f"Memory snapshot saved: session {session.session_id}")

            agent.clear_memory()
            logger.info(f"Memory cleared: session {session.session_id}")

            p1 = phase1_stats[i]
            p2 = phase2_stats[i]
            note2 = p1["call_2_note_construction"]
            evo3  = p1["call_3_evolution"]
            qa4   = p2

            total_input  = note2["input"] + evo3["input"] + qa4["qa_input"]
            total_output = note2["output"] + evo3["output"] + qa4["qa_output"]
            total_llm_calls = note2["llm_calls"] + evo3["llm_calls"] + qa4["num_qa_llm_calls"]

            token_stats = {
                "call_2_note_construction": {
                    "input":               note2["input"],
                    "output":              note2["output"],
                    "llm_calls":           note2["llm_calls"],
                    "parse_fallback_count": internal_stats_list[i]["note_parse_fallback_count"],
                },
                "call_3_evolution": {
                    "input":     evo3["input"],
                    "output":    evo3["output"],
                    "llm_calls": evo3["llm_calls"],
                },
                "call_4_qa": {
                    "input":     qa4["qa_input"],
                    "output":    qa4["qa_output"],
                    "llm_calls": qa4["num_qa_llm_calls"],
                },
                "total_input":     total_input,
                "total_output":    total_output,
                "total_llm_calls": total_llm_calls,
            }

            evo_stats = {
                "evo_triggered_count": internal_stats_list[i]["evo_triggered_count"],
                "actions_taken":       internal_stats_list[i]["actions_taken"],
            }

            result = {
                "session_id":      session.session_id,
                "qa_results":      qa_results_list[i],
                "token_statistics":    token_stats,
                "evolution_statistics": evo_stats,
                "memory_at_qa_start":  memory_stats_list[i],
                "memory_snapshot_path": (
                    f"memory_snapshots/session_{session.session_id}/"
                    if cfg.SAVE_MEMORY_SNAPSHOTS else None
                ),
            }
            if config_metadata is not None:
                result["config_metadata"] = config_metadata
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_batched(
        self,
        sessions: List[Session],
        agents,
    ) -> List[Dict]:
        """Memory construction — interleaved across sessions, turn by turn."""
        phase1_stats = [
            {
                "call_2_note_construction": {"input": 0, "output": 0, "llm_calls": 0},
                "call_3_evolution":         {"input": 0, "output": 0, "llm_calls": 0},
            }
            for _ in sessions
        ]

        max_turns = max(len(s.get_turn_pairs()) for s in sessions)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active = [
                (i, sessions[i], agents[i])
                for i in range(len(sessions))
                if turn_idx < len(sessions[i].get_turn_pairs())
            ]

            self._process_turn_batch(
                active, turn_idx, is_user=True,
                phase1_stats=phase1_stats,
            )
            self._process_turn_batch(
                active, turn_idx, is_user=False,
                phase1_stats=phase1_stats,
            )

        # Drain each agent's LLMWrapper token counters (covers any sequential
        # add_note() calls that may have occurred via the sequential path)
        for i, agent in enumerate(agents):
            mem_tokens = agent.get_and_reset_memory_tokens()
            for ct in ("call_2_note_construction", "call_3_evolution"):
                phase1_stats[i][ct]["input"]     += mem_tokens[ct]["input"]
                phase1_stats[i][ct]["output"]    += mem_tokens[ct]["output"]
                phase1_stats[i][ct]["llm_calls"] += mem_tokens[ct]["llm_calls"]

        return phase1_stats

    def _process_turn_batch(
        self,
        active_sessions: List[Tuple],
        turn_idx: int,
        is_user: bool,
        phase1_stats: List[Dict],
    ):
        """Process one turn (user or assistant) across all active sessions as a batch."""
        from memory_layer import LLMWrapper, _EVOLUTION_GUIDED_JSON

        # Gather turn data
        turn_data = []
        for i, session, agent in active_sessions:
            user_turn, assistant_turn = session.get_turn_pairs()[turn_idx]
            turn = user_turn if is_user else assistant_turn
            if turn is None:
                continue
            turn_data.append((i, session, agent, turn))

        if not turn_data:
            return

        # Step 1: build analyze prompts
        analyze_jobs = []  # (i, session, agent, turn, prompt_content)
        for i, session, agent, turn in turn_data:
            memory_content = turn.to_memory_content()
            prompt_content = agent.memory_system.build_analyze_prompt(memory_content)
            analyze_jobs.append((i, session, agent, turn, prompt_content))

        # Step 2: batch analyze (plain text)
        analyze_prompt_contents = [j[4] for j in analyze_jobs]
        analyze_texts, analyze_usages = self._batch_generate_with_retry(
            prompts=analyze_prompt_contents,
            system_prompt=LLMWrapper.PLAIN_TEXT_SYSTEM,
            max_tokens=1000,
            temperature=cfg.TEMPERATURE,
            guided_json=None,
        )

        # Step 3: log + accumulate tokens per agent
        for (i, session, agent, turn, prompt_content), text, usage in zip(
            analyze_jobs, analyze_texts, analyze_usages
        ):
            if agent.memory_system.llm_logger is not None:
                agent.memory_system.llm_logger.log(
                    "call_2_note_construction", "", prompt_content, text
                )
            agent.accumulate_memory_tokens(
                usage["prompt_tokens"], usage["completion_tokens"], 1,
                call_type="call_2_note_construction",
            )

        # Step 4: apply analyze results → split into evolve / no-evolve
        evolve_jobs = []   # (i, session, agent, turn, prompt_content, note, evolve_ctx)
        no_evolve_items = []  # (i, agent, note)

        for (i, session, agent, turn, prompt_content), text in zip(analyze_jobs, analyze_texts):
            timestamp = f"{session.session_id:04d}_{turn.conv_id:04d}_{turn.turn_id:04d}"
            memory_content = turn.to_memory_content()
            note, evolve_prompt, evolve_ctx = agent.memory_system.apply_analyze_result(
                memory_content, text, time=timestamp
            )
            if evolve_prompt is None:
                no_evolve_items.append((i, agent, note))
            else:
                evolve_jobs.append((i, session, agent, turn, evolve_prompt, note, evolve_ctx))

        # Step 5: store no-evolve notes immediately
        for i, agent, note in no_evolve_items:
            agent.memory_system.store_note(note)

        # Step 6: batch evolve (JSON)
        if evolve_jobs:
            evolve_prompt_contents = [j[4] for j in evolve_jobs]
            evolve_results, evolve_usages = self._batch_generate_with_retry(
                prompts=evolve_prompt_contents,
                system_prompt=LLMWrapper.JSON_SYSTEM,
                max_tokens=1000,
                temperature=cfg.TEMPERATURE,
                guided_json=_EVOLUTION_GUIDED_JSON,
            )

            for (i, session, agent, turn, prompt_content, note, evolve_ctx), result, usage in zip(
                evolve_jobs, evolve_results, evolve_usages
            ):
                if agent.memory_system.llm_logger is not None:
                    agent.memory_system.llm_logger.log(
                        "call_3_evolution", "", prompt_content, result
                    )
                agent.accumulate_memory_tokens(
                    usage["prompt_tokens"], usage["completion_tokens"], 1,
                    call_type="call_3_evolution",
                )
                if isinstance(result, dict):
                    agent.memory_system.apply_evolve_result(note, result, evolve_ctx)
                else:
                    # Fallback: store without evolution
                    agent.memory_system.store_note(note)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2_batched(
        self,
        sessions: List[Session],
        agents,
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """QA answering — all sessions' QA prompts batched together."""
        from agent import QA_SCHEMA_OPPOSED, QA_SCHEMA_SUPPORTIVE
        from agent import QA_PROMPT_OPPOSED, QA_PROMPT_SUPPORTIVE

        schema = QA_SCHEMA_SUPPORTIVE if self.subset == "supportive" else QA_SCHEMA_OPPOSED
        prompt_template = QA_PROMPT_SUPPORTIVE if self.subset == "supportive" else QA_PROMPT_OPPOSED

        # Collect all QA jobs with retrieval
        qa_jobs = []  # (session_idx, qa, retrieved_metadata, prompt_content)

        for i, (session, agent) in enumerate(zip(sessions, agents)):
            for qa in session.qa:
                retrieved_str, retrieved_metadata = agent.retrieve_memory_with_metadata(qa.question)

                log_items, num_linked = agent.retrieve_for_log(qa.question)
                write_retrieval_log(retrieval_log_paths[i], build_retrieval_log_entry(
                    phase="qa",
                    session_id=session.session_id,
                    conv_id=-1,
                    turn_id=-1,
                    query=qa.question,
                    retrieved_items=log_items,
                    num_linked=num_linked,
                ))

                memory_str = retrieved_str if retrieved_str else "No memory available."
                prompt_content = prompt_template.format(
                    retrieved_memory=memory_str,
                    question=qa.question,
                )
                qa_jobs.append((i, qa, retrieved_metadata, prompt_content))

        # Batch generate in chunks
        all_pairs = []  # (qa_job, result, usage)

        for chunk_start in tqdm(range(0, len(qa_jobs), cfg.QA_BATCH_SIZE), desc="Phase2 QA chunks"):
            chunk = qa_jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            chunk_prompts = [j[3] for j in chunk]
            chunk_results, chunk_usages = self._batch_generate_with_retry(
                prompts=chunk_prompts,
                system_prompt=self.QA_SYSTEM_PROMPT,
                max_tokens=cfg.MAX_TOKENS,
                temperature=cfg.TEMPERATURE,
                guided_json=schema,
            )
            all_pairs.extend(zip(chunk, chunk_results, chunk_usages))

        # Distribute results
        qa_results_per_session: List[List[Dict]] = [[] for _ in sessions]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_llm_calls": 0}
            for _ in sessions
        ]

        for (i, qa, retrieved_metadata, prompt_content), result, usage in all_pairs:
            if agents[i]._llm_logger is not None:
                agents[i]._llm_logger.log("call_4_qa", "", prompt_content, result)

            phase2_stats[i]["qa_input"]          += usage["prompt_tokens"]
            phase2_stats[i]["qa_output"]         += usage["completion_tokens"]
            phase2_stats[i]["num_qa_llm_calls"]  += 1

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            if self.subset == "supportive":
                label = answer.strip().lower()
                answer = label if label in ("yes", "no") else "unknown"

            qa_results_per_session[i].append({
                "question":            qa.question,
                "generated_answer":    answer,
                "ground_truth_answer": qa.answer,
                "retrieved_memories":  retrieved_metadata,
                "qa_tokens": {
                    "input":  usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "model":  self.model_path,
                },
            })

        return qa_results_per_session, phase2_stats

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
        """Batch generate. Returns (results, usages).

        For plain text (guided_json=None): results is a list of strings.
        For JSON (guided_json set): results is a list of dicts.
        Items that fail JSON parsing are retried sequentially via generate().

        vLLM hard errors (OOM, internal crash) propagate immediately — the
        caller (main loop) handles these by stopping the process.
        """
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

        # JSON mode: parse each result
        from llm_client import _parse_json_response

        parsed = []
        retry_indices = []
        for idx, text in enumerate(texts):
            try:
                parsed.append(_parse_json_response(text) if isinstance(text, str) else text)
            except (json.JSONDecodeError, ValueError):
                parsed.append(None)
                retry_indices.append(idx)

        # Retry failed items sequentially
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

    from agent import BaseAgent  # noqa: E402

    # ---- Step 2: parse args ----
    parser = argparse.ArgumentParser(
        description="A-MEM Batch Experiment on ImplexConv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-session", type=int, required=True)
    parser.add_argument("--end-session",   type=int, required=True)
    parser.add_argument("--subset", type=str, required=True, choices=["opposed", "supportive"])
    parser.add_argument("--model", type=str, default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE,
                        help=f"Sessions to process in parallel (default: {cfg.BATCH_SIZE})")
    parser.add_argument("--config", type=str, default="config_0")
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    cfg.ensure_directories(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )

    session_dir = cfg.get_session_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file    = cfg.get_results_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    snapshots_dir   = cfg.get_memory_snapshots_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    prompt_log_dir  = cfg.get_prompt_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)

    logger.info("=" * 60)
    logger.info("A-MEM Batch Experiment")
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

    # Load dataset
    logger.info("Loading dataset...")
    dataset_path = cfg.DATASET_OPPOSED if args.subset == "opposed" else cfg.DATASET_SUPPORTIVE
    sessions = load_implexconv_dataset(dataset_path, args.subset)

    if args.end_session >= len(sessions):
        logger.error(f"end_session={args.end_session} out of range (dataset has {len(sessions)} sessions)")
        return 1

    target_sessions = sessions[args.start_session: args.end_session + 1]

    # Resume from checkpoint
    completed_ids: Set[int] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_ids = load_checkpoint(checkpoint_file)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} sessions already completed")

    pending_sessions = [s for s in target_sessions if s.session_id not in completed_ids]
    if not pending_sessions:
        logger.info("All sessions already completed.")
        return 0

    # Init shared embedding model
    logger.info(f"Loading shared embedding model: {cfg.EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    # Init LLM
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(args.model, args.tensor_parallel, args.gpu_memory,
                                   max_model_len=args.max_model_len)
    logger.info("LLM client ready.")

    runner = BatchedAMEMRunner(
        llm_client=llm_client,
        subset=args.subset,
        model_path=args.model,
        shared_embedding_model=shared_embedding_model,
    )

    config_metadata = {
        "config_name":         config_name,
        "model":               args.model,
        "subset":              args.subset,
        "embedding_model":     cfg.EMBEDDING_MODEL,
        "retrieve_k":          cfg.RETRIEVE_K,
        "evolution_threshold": cfg.EVOLUTION_THRESHOLD,
        "temperature":         cfg.TEMPERATURE,
        "max_tokens":          cfg.MAX_TOKENS,
        "session_range":       [args.start_session, args.end_session],
    }

    results = load_existing_results(results_file)

    print(f"\n{'#'*60}")
    print(f"# A-MEM Batch  |  subset={args.subset}")
    print(f"# Model  : {cfg.extract_model_name(args.model)}")
    print(f"# Sessions [{args.start_session}, {args.end_session}]  ({len(pending_sessions)} to process)")
    print(f"# Batch size : {args.batch_size}")
    print(f"{'#'*60}\n")

    # Process in batches
    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_sessions[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        session_ids = [s.session_id for s in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: sessions {session_ids}")

        # Prepare per-session paths
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
        except Exception as e:
            # Hard errors (vLLM crash, OOM) → stop immediately
            logger.error(f"Batch {batch_idx + 1} failed with hard error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Save each completed session immediately
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
