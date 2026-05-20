"""
LD-Agent Module (ImplexConv)

Single entry point wrapping EventMemory + Personas + Generator.

Public interface
----------------
  process_turn(user_utterance, gt_response, conv_id, turn_id, session_id)
      -> TurnResult
  get_qa_answer(question, subset)
      -> QAResult
  flush_to_ltm(session_id)
      -> Dict[str, int]   (flush token info)
  clear()
  save_snapshot(directory)

Changes from ldagent/ldagent_module.py:
  - drift_detected removed from TurnResult (no drift detection in new protocol)
  - set_ltm_write() removed from public API (no Phase 2)
  - process_turn() calls response_build_json() instead of response_build_with_drift_detection()
  - get_qa_answer() accepts subset parameter; calls generate_qa_answer(subset=...)
  - TurnResult and QAResult carry retrieval_log_data dict for retrieval logging
  - Internal token accumulation logic preserved unchanged

Reference: "Hello Again! LLM-powered Personalized Agent for Long-term Dialogue"
           (Li et al., NAACL 2025)
"""

import datetime
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from event_memory import EventMemory
from personas import Personas
from generator import Generator, format_memories_for_prompt
from load_dataset import compute_virtual_seconds


# =============================================================================
# LLM CALL LOGGER
# =============================================================================

class LLMCallLogger:
    """
    Logs all LLM calls (input + output) to call-type-specific folders as JSONL files.

    Folder structure:
        {base_dir}/call_1_response/
        {base_dir}/call_2_user_persona/
        {base_dir}/call_3_agent_persona/
        {base_dir}/call_4_summarization/
        {base_dir}/call_5_qa/

    Each folder contains a single 'calls.jsonl' file with one JSON object per line.
    """

    CALL_DIRS = [
        "call_1_response",
        "call_2_user_persona",
        "call_3_agent_persona",
        "call_4_summarization",
        "call_5_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        """Append one log entry to {base_dir}/{call_type}/calls.jsonl."""
        entry = {
            "timestamp":     datetime.datetime.now().isoformat(),
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
# RESULT DATACLASSES
# =============================================================================

@dataclass
class TurnResult:
    """Result of a single process_turn() call."""
    response:            str
    token_info:          Dict[str, int]   = field(default_factory=dict)
    internal_token_info: Dict[str, int]   = field(default_factory=dict)
    timing:              Dict[str, float] = field(default_factory=dict)
    retrieval_log_data:  Dict[str, Any]   = field(default_factory=dict)


@dataclass
class QAResult:
    """Result of a single get_qa_answer() call."""
    answer:             str
    token_info:         Dict[str, int]   = field(default_factory=dict)
    retrieved_memories: List[Dict]       = field(default_factory=list)
    timing:             Dict[str, float] = field(default_factory=dict)
    retrieval_log_data: Dict[str, Any]   = field(default_factory=dict)
    num_api_calls:      int              = 0


# =============================================================================
# HELPERS
# =============================================================================

def _read_token_attr(obj, attr: str) -> Dict[str, int]:
    """Safely read a token-info attribute from a sub-module (default zeros)."""
    info = getattr(obj, attr, None)
    return info if isinstance(info, dict) else {"input": 0, "output": 0}


def _accum_internal(dst: Dict, token_info: Dict, count: int = 1):
    """Add token_info into dst accumulator."""
    dst["input"]  += token_info.get("input",  0)
    dst["output"] += token_info.get("output", 0)
    dst["calls"]  += count


# =============================================================================
# LD-AGENT MODULE
# =============================================================================

class LDAgentModule:
    """
    LD-Agent module wrapping EventMemory + Personas + Generator.

    Contract:
      - GT agent responses are used wherever the module re-consumes agent text
        (STM storage, persona update).
      - Internal LLM tokens (persona extraction, boundary summarise) are
        collected and returned so the experiment runner can tally total usage.
      - QA exchanges are never stored in memory (caller responsibility).
    """

    def __init__(
        self,
        llm_client,
        config,
        logger: logging.Logger,
        sample_id: str,
    ):
        self._llm_client = llm_client
        self._config     = config
        self._logger     = logger
        self._sample_id  = sample_id

        self._memory_bank: Optional[EventMemory] = None
        self._personas:    Optional[Personas]    = None
        self._generator:   Optional[Generator]   = None

        self._initialize()

    # =========================================================================
    # INITIALISATION / TEARDOWN
    # =========================================================================

    def _initialize(self):
        cfg = self._config
        log = self._logger

        self._memory_bank = EventMemory(
            llm_client=self._llm_client,
            sample_id=self._sample_id,
            logger=log,
            usr_name=cfg.USR_NAME,
            agent_name=cfg.AGENT_NAME,
            relevance_memory_number=cfg.RELEVANCE_MEMORY_NUMBER,
            dist_threshold=cfg.DIST_THRESHOLD,
            decay_temp=cfg.DECAY_TEMP,
            ori_mem_query=cfg.ORI_MEM_QUERY,
            finalize_every_n_convs=cfg.FINALIZE_EVERY_N_CONVS,
        )

        self._personas = Personas(
            llm_client=self._llm_client,
            logger=log,
            usr_name=cfg.USR_NAME,
            agent_name=cfg.AGENT_NAME,
            max_user_personas=cfg.MAX_USER_PERSONAS,
            max_agent_personas=cfg.MAX_AGENT_PERSONAS,
        )

        self._generator = Generator(
            llm_client=self._llm_client,
            logger=log,
            usr_name=cfg.USR_NAME,
            agent_name=cfg.AGENT_NAME,
            max_tokens=cfg.MAX_TOKENS,
            temperature=cfg.TEMPERATURE,
            json_retry=cfg.JSON_RETRY,
        )

    def clear(self):
        if self._memory_bank:
            self._memory_bank.clear()
        if self._personas:
            self._personas.clear()

    def save_snapshot(self, directory: Path):
        if self._memory_bank:
            self._memory_bank.save_snapshot(directory)
        if self._personas:
            self._personas.save_snapshot(directory)

    def set_llm_logger(self, llm_logger):
        """Propagate LLMCallLogger to all sub-modules."""
        if self._memory_bank:
            self._memory_bank.set_llm_logger(llm_logger)
        if self._personas:
            self._personas.set_llm_logger(llm_logger)
        if self._generator:
            self._generator.set_llm_logger(llm_logger)

    # =========================================================================
    # CONTEXT FORMATTING
    # =========================================================================

    def _format_context(self, context_memories: List[Dict], current_inquiry: str) -> str:
        cfg   = self._config
        lines = (
            [f"[TURN {m.get('idx', i)}] : {m['dialog']}." for i, m in enumerate(context_memories)]
            if context_memories else []
        )
        lines.append(f"In this turn, {cfg.USR_NAME} said: {current_inquiry}.")
        return "\n".join(lines)

    # =========================================================================
    # STM → LTM FLUSH  (before QA)
    # =========================================================================

    def flush_to_ltm(self, session_id: int) -> Dict[str, int]:
        """
        Force-commit remaining STM content to LTM before running QA.
        Delegates to EventMemory.flush_stm(); STM is kept intact for QA context.

        Returns token info dict for the summarise call.
        """
        return self._memory_bank.flush_stm(current_session_id=session_id)

    # =========================================================================
    # MAIN TURN PROCESSING
    # =========================================================================

    def process_turn(
        self,
        user_utterance: str,
        gt_response:    str,
        conv_id:        int,
        turn_id:        int,
        session_id:     int,
    ) -> TurnResult:
        """
        Process one dialogue turn end-to-end (QA-only variant).

        Response prompt is constructed and logged but the LLM is NOT called.
        GT response is used for persona update and STM storage.

        Execution order:
          1. compute_virtual_seconds(conv_id, turn_id)
          2. context_retrieve(user_utterance)         → STM + context
          3. relevance_retrieve(user_utterance)       → LTM query
          4. _user_traits_update(user_utterance)      → user persona bank
          5. get_current_traits()                     → merged traits snapshot
          6. build_response_prompt_only(...)          → prompt logged, input tokens counted
          7. _agent_traits_update(gt_response)        → agent persona bank  [GT]
          8. add_agent_response(gt_response)          → STM updated          [GT]
        """
        mb  = self._memory_bank
        per = self._personas
        gen = self._generator
        cfg = self._config

        internal = {"input": 0, "output": 0, "calls": 0}

        # Compute virtual seconds for this turn
        vs = compute_virtual_seconds(
            conv_id, turn_id,
            cfg.CONV_IDS_PER_DAY,
            cfg.MINUTES_PER_TURN,
        )

        # --- 1 & 2: Memory retrieval ---
        context_memories = mb.context_retrieve(
            query=user_utterance,
            current_virtual_seconds=vs,
            current_conv_id=conv_id,
            current_session_id=session_id,
        )
        # Collect boundary summarise tokens if a conv_id boundary was crossed
        summ_tokens = _read_token_attr(mb, "last_summarize_token_info")
        if summ_tokens.get("input", 0) > 0:
            _accum_internal(internal, summ_tokens)

        context_str = self._format_context(context_memories, user_utterance)

        relevant_memories = mb.relevance_retrieve(
            ori_query=user_utterance,
            n_results=cfg.RELEVANCE_MEMORY_NUMBER,
            current_virtual_seconds=vs,
        )
        memories_str = format_memories_for_prompt(relevant_memories, vs)

        # --- 3 & 4: Persona update (user BEFORE generation) ---
        per._user_traits_update(user_utterance)
        _accum_internal(internal, _read_token_attr(per, "last_user_token_info"), count=1)

        user_traits, agent_traits = per.get_current_traits()

        # --- 5: Build response prompt only (NO LLM call) ---
        prompt_snapshot, estimated_input_tokens = gen.build_response_prompt_only(
            inquiry=user_utterance,
            context=context_str,
            memories=memories_str,
            user_traits=user_traits,
            agent_traits=agent_traits,
        )
        token_info = {"input": estimated_input_tokens, "output": 0}

        # --- 6: Persona update (agent AFTER, using GT) ---
        per._agent_traits_update(gt_response)
        _accum_internal(internal, _read_token_attr(per, "last_agent_token_info"), count=1)

        # --- 7: Store GT response in STM ---
        mb.add_agent_response(
            response=gt_response,
            current_virtual_seconds=vs,
            current_session_id=session_id,
        )

        # Build retrieval log data
        def _count_traits(traits_str: str) -> int:
            return len([t for t in traits_str.split("\n") if t.strip()]) if traits_str else 0

        retrieval_log_data = {
            "query":              user_utterance,
            "relevant_memories":  relevant_memories,
            "prompt_snapshot":    prompt_snapshot,
            "module_specific": {
                "ltm_entry_count":    mb.collection.count(),
                "stm_context_turns":  len(context_memories),
                "user_trait_count":   _count_traits(user_traits),
                "agent_trait_count":  _count_traits(agent_traits),
            },
        }

        return TurnResult(
            response="",
            token_info=token_info,
            internal_token_info=internal,
            timing={},
            retrieval_log_data=retrieval_log_data,
        )

    # =========================================================================
    # QA ANSWERING
    # =========================================================================

    def _format_stm_context_for_qa(self) -> str:
        """Format STM entries as context for QA (no current inquiry appended)."""
        mb = self._memory_bank
        if not mb.short_term_memory:
            return ""
        lines = [f"[TURN {i}] : {m['dialog']}." for i, m in enumerate(mb.short_term_memory)]
        return "\n".join(lines)

    def get_qa_answer(self, question: str, subset: str = "opposed") -> QAResult:
        """
        Answer a QA question using accumulated LTM + STM context.
        QA exchanges are never stored in memory (caller responsibility).

        Args:
            question: QA question string
            subset:   "opposed" (free-form) or "supportive" (yes/no/unknown)
        """
        mb  = self._memory_bank
        per = self._personas
        gen = self._generator
        cfg = self._config

        t0 = time.time()

        relevant_memories = mb.relevance_retrieve(
            ori_query=question,
            n_results=cfg.RETRIEVE_K,
            current_virtual_seconds=mb.current_virtual_seconds,
        )
        memories_str = format_memories_for_prompt(relevant_memories, mb.current_virtual_seconds)

        # Build STM context for QA (recent conversation turns)
        context_str = self._format_stm_context_for_qa()

        t1 = time.time()

        user_traits, agent_traits = per.get_current_traits()
        answer, token_info, prompt_snapshot, num_api_calls = gen.generate_qa_answer(
            question=question,
            memories=memories_str,
            user_traits=user_traits,
            agent_traits=agent_traits,
            subset=subset,
            context=context_str,
        )

        t2 = time.time()

        retrieved_metadata = [
            {
                "session_id":      mem.get("session_id",      0),
                "conv_id":         mem.get("conv_id",         0),
                "virtual_seconds": mem.get("virtual_seconds", 0.0),
                "score":           mem.get("score",           0.0),
            }
            for mem in relevant_memories
        ]

        def _count_traits(traits_str: str) -> int:
            return len([t for t in traits_str.split("\n") if t.strip()]) if traits_str else 0

        retrieval_log_data = {
            "query":             question,
            "relevant_memories": relevant_memories,
            "prompt_snapshot":   prompt_snapshot,
            "module_specific": {
                "ltm_entry_count":   mb.collection.count(),
                "stm_context_turns": len(mb.short_term_memory),
                "user_trait_count":  _count_traits(user_traits),
                "agent_trait_count": _count_traits(agent_traits),
            },
        }

        return QAResult(
            answer=answer,
            token_info=token_info,
            retrieved_memories=retrieved_metadata,
            timing={
                "retrieval_time": t1 - t0,
                "inference_time": t2 - t1,
                "total_time":     t2 - t0,
            },
            retrieval_log_data=retrieval_log_data,
            num_api_calls=num_api_calls,
        )
