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
  get_memory_stats()
      -> Dict[str, int]   (num_memories, total_content_tokens)
  clear()
  save_snapshot(directory)

Changes from v3:
  - LLMCallLogger.CALL_DIRS: call_1_response removed (response prompt no longer built)
  - TurnResult: token_info and timing removed (response generation tracking removed)
  - QAResult: timing removed; num_api_calls renamed to num_llm_calls
  - process_turn(): build_response_prompt_only() call removed
  - get_qa_answer(): timing code removed
  - get_memory_stats() added; delegates to EventMemory
  - LDAgentModule: shared_encoder parameter added; passed to EventMemory
  - collection.count() references replaced with get_memory_count()
"""

import json
import logging
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
        {base_dir}/call_2_user_persona/
        {base_dir}/call_3_agent_persona/
        {base_dir}/call_4_summarization/
        {base_dir}/call_5_qa/

    Each folder contains a single 'calls.jsonl' file with one JSON object per line.
    """

    CALL_DIRS = [
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
    response:           str
    internal_token_info: Dict[str, int]  = field(default_factory=dict)
    retrieval_log_data:  Dict[str, Any]  = field(default_factory=dict)


@dataclass
class QAResult:
    """Result of a single get_qa_answer() call."""
    answer:             str
    token_info:         Dict[str, int]   = field(default_factory=dict)
    retrieved_memories: List[Dict]       = field(default_factory=list)
    retrieval_log_data: Dict[str, Any]   = field(default_factory=dict)
    num_llm_calls:      int              = 0


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


def _count_traits(traits_str: str) -> int:
    """Count non-empty trait lines in a merged trait string."""
    return len([t for t in traits_str.split("\n") if t.strip()]) if traits_str else 0


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
        shared_lemma_tokenizer=None,
        shared_encoder=None,
    ):
        self._llm_client = llm_client
        self._config     = config
        self._logger     = logger
        self._sample_id  = sample_id
        self._shared_lemma_tokenizer = shared_lemma_tokenizer
        self._shared_encoder         = shared_encoder

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
            lemma_tokenizer=self._shared_lemma_tokenizer,
            encoder=self._shared_encoder,
            finalize_input_context_limit=cfg.FINALIZE_INPUT_CONTEXT_LIMIT,
            finalize_context_utilization=cfg.FINALIZE_CONTEXT_UTILIZATION,
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

    @property
    def memory_bank(self) -> EventMemory:
        return self._memory_bank

    @property
    def personas(self) -> Personas:
        return self._personas

    @property
    def generator(self) -> Generator:
        return self._generator

    @property
    def config(self):
        return self._config

    # =========================================================================
    # STM → LTM FLUSH  (before QA)
    # =========================================================================

    def flush_to_ltm(self, session_id: int) -> Dict[str, int]:
        """
        Force-commit remaining STM content to LTM before running QA.
        STM is kept intact for QA context.
        Returns token info dict for the summarise call.
        """
        return self._memory_bank.flush_stm(current_session_id=session_id)

    # =========================================================================
    # MEMORY STATS
    # =========================================================================

    def get_memory_stats(self) -> Dict[str, int]:
        """Return LTM entry count and estimated total content tokens."""
        return self._memory_bank.get_memory_stats()

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

        GT response is used for persona update and STM storage.
        No LLM call is made for response generation.

        Execution order:
          1. compute_virtual_seconds(conv_id, turn_id)
          2. context_retrieve(user_utterance)         → STM + context
          3. relevance_retrieve(user_utterance)       → LTM query
          4. _user_traits_update(user_utterance)      → user persona bank
          5. get_current_traits()                     → merged traits snapshot
          6. _agent_traits_update(gt_response)        → agent persona bank  [GT]
          7. add_agent_response(gt_response)          → STM updated          [GT]
        """
        mb  = self._memory_bank
        per = self._personas
        cfg = self._config

        internal = {"input": 0, "output": 0, "calls": 0}

        vs = compute_virtual_seconds(
            conv_id, turn_id,
            cfg.CONV_IDS_PER_DAY,
            cfg.MINUTES_PER_TURN,
        )

        # 1 & 2: Memory retrieval
        context_memories = mb.context_retrieve(
            query=user_utterance,
            current_virtual_seconds=vs,
            current_conv_id=conv_id,
            current_session_id=session_id,
        )
        summ_tokens = _read_token_attr(mb, "last_summarize_token_info")
        if summ_tokens.get("input", 0) > 0:
            _accum_internal(internal, summ_tokens)

        # 3 & 4: Persona update (user BEFORE generation)
        per._user_traits_update(user_utterance)
        _accum_internal(internal, _read_token_attr(per, "last_user_token_info"), count=1)

        user_traits, agent_traits = per.get_current_traits()

        # 5: Persona update (agent AFTER, using GT)
        per._agent_traits_update(gt_response)
        _accum_internal(internal, _read_token_attr(per, "last_agent_token_info"), count=1)

        # 6: Store GT response in STM
        mb.add_agent_response(
            response=gt_response,
            current_virtual_seconds=vs,
            current_session_id=session_id,
        )

        retrieval_log_data = {
            "query": user_utterance,
            "module_specific": {
                "ltm_entry_count":   mb.get_memory_count(),
                "stm_context_turns": len(context_memories),
                "user_trait_count":  _count_traits(user_traits),
                "agent_trait_count": _count_traits(agent_traits),
            },
        }

        return TurnResult(
            response="",
            internal_token_info=internal,
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
        QA exchanges are never stored in memory.

        Args:
            question: QA question string
            subset:   "opposed" (free-form) or "supportive" (yes/no/unknown)
        """
        mb  = self._memory_bank
        per = self._personas
        gen = self._generator
        cfg = self._config

        relevant_memories = mb.relevance_retrieve(
            ori_query=question,
            n_results=cfg.RETRIEVE_K,
            current_virtual_seconds=mb.current_virtual_seconds,
        )
        memories_str = format_memories_for_prompt(relevant_memories, mb.current_virtual_seconds)
        context_str  = self._format_stm_context_for_qa()

        user_traits, agent_traits = per.get_current_traits()
        answer, token_info, prompt_snapshot, num_llm_calls = gen.generate_qa_answer(
            question=question,
            memories=memories_str,
            user_traits=user_traits,
            agent_traits=agent_traits,
            subset=subset,
            context=context_str,
        )

        retrieved_metadata = [
            {
                "session_id":      mem.get("session_id",      0),
                "conv_id":         mem.get("conv_id",         0),
                "virtual_seconds": mem.get("virtual_seconds", 0.0),
                "score":           mem.get("score",           0.0),
            }
            for mem in relevant_memories
        ]

        retrieval_log_data = {
            "query":             question,
            "relevant_memories": relevant_memories,
            "prompt_snapshot":   prompt_snapshot,
            "module_specific": {
                "ltm_entry_count":   mb.get_memory_count(),
                "stm_context_turns": len(mb.short_term_memory),
                "user_trait_count":  _count_traits(user_traits),
                "agent_trait_count": _count_traits(agent_traits),
            },
        }

        return QAResult(
            answer=answer,
            token_info=token_info,
            retrieved_memories=retrieved_metadata,
            retrieval_log_data=retrieval_log_data,
            num_llm_calls=num_llm_calls,
        )
