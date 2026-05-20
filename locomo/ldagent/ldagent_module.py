"""
LD-Agent Module (LoComo) — Batch-enabled variant

Changes from previous version:
  - import time removed; all timing tracking removed
  - TurnResult: timing removed
  - QAResult: timing removed; num_api_calls renamed to num_llm_calls
  - LLMCallLogger.log(): timestamp field removed
  - get_qa_answer(): timing code removed
  - Per-call-type token accumulators added:
      _call1_input/output/calls  (call_1_speaker1_persona)
      _call2_input/output/calls  (call_2_speaker2_persona)
      _call3_input/output/calls  (call_3_summarization)
  - accumulate_internal_tokens() replaced by accumulate_tokens(call_type, ...)
  - get_and_reset_internal_tokens() replaced by get_and_reset_token_counts_by_type()
  - get_memory_stats() added; delegates to EventMemory
  - EventMemory now uses numpy+SentenceTransformer (no chromadb)

All original methods preserved for sequential-mode compatibility.
"""

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from event_memory import EventMemory
from personas import Personas
from generator import Generator, format_memories_for_prompt, format_context_for_prompt


# =============================================================================
# LLM CALL LOGGER
# =============================================================================

class LLMCallLogger:
    """
    Logs all LLM calls (input + output) to call-type-specific folders as JSONL.

    Folder structure:
        {base_dir}/call_1_speaker1_persona/
        {base_dir}/call_2_speaker2_persona/
        {base_dir}/call_3_summarization/
        {base_dir}/call_4_qa/
    """

    CALL_DIRS = [
        "call_1_speaker1_persona",
        "call_2_speaker2_persona",
        "call_3_summarization",
        "call_4_qa",
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
    internal_token_info: Dict[str, int] = field(default_factory=dict)
    retrieval_log_data:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    """Result of a single get_qa_answer() call."""
    answer:             str
    token_info:         Dict[str, int] = field(default_factory=dict)
    retrieved_memories: List[Dict]     = field(default_factory=list)
    retrieval_log_data: Dict[str, Any] = field(default_factory=dict)
    num_llm_calls:      int            = 0


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
    return len([t for t in traits_str.split("\n") if t.strip()]) if traits_str else 0


# =============================================================================
# LD-AGENT MODULE
# =============================================================================

class LDAgentModule:
    """
    LD-Agent module wrapping EventMemory + Personas + Generator (LoComo batch variant).

    Batch API extensions:
      - self._llm_logger  — direct access to LLMCallLogger
      - accumulate_call1/2/3_tokens() — per-call-type token accumulation
      - get_and_reset_token_counts_by_type() — returns per-call-type token dict
      - get_memory_stats() — delegates to EventMemory
    """

    def __init__(
        self,
        llm_client,
        config,
        logger:    logging.Logger,
        sample_id: str,
        speaker_a: str,
        speaker_b: str,
        encoder=None,
        lemma_tokenizer=None,
    ):
        self._llm_client = llm_client
        self._config     = config
        self._logger     = logger
        self._sample_id  = sample_id
        self._speaker_a  = speaker_a
        self._speaker_b  = speaker_b
        self._encoder    = encoder
        self._lemma_tokenizer = lemma_tokenizer

        self._memory_bank: Optional[EventMemory] = None
        self._personas:    Optional[Personas]    = None
        self._generator:   Optional[Generator]   = None

        self._llm_logger = None

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
            speaker_a=self._speaker_a,
            speaker_b=self._speaker_b,
            relevance_memory_number=cfg.RELEVANCE_MEMORY_NUMBER,
            dist_threshold=cfg.DIST_THRESHOLD,
            decay_temp=cfg.DECAY_TEMP,
            ori_mem_query=cfg.ORI_MEM_QUERY,
            flush_gap_seconds=cfg.STM_FLUSH_GAP_SECONDS,
            lemma_tokenizer=self._lemma_tokenizer,
            encoder=self._encoder,
        )

        self._personas = Personas(
            llm_client=self._llm_client,
            logger=log,
            speaker_a=self._speaker_a,
            speaker_b=self._speaker_b,
            max_speaker_a_personas=cfg.MAX_SPEAKER_A_PERSONAS,
            max_speaker_b_personas=cfg.MAX_SPEAKER_B_PERSONAS,
        )

        self._generator = Generator(
            llm_client=self._llm_client,
            logger=log,
            speaker_a=self._speaker_a,
            speaker_b=self._speaker_b,
            max_tokens=cfg.MAX_TOKENS,
            temperature=cfg.TEMPERATURE,
            temperature_c5=cfg.TEMPERATURE_C5,
            json_retry=cfg.JSON_RETRY,
        )

        # Per-call-type token accumulators
        self._call_tokens: Dict[str, Dict[str, int]] = {
            "call_1_speaker1_persona": {"input": 0, "output": 0, "calls": 0},
            "call_2_speaker2_persona": {"input": 0, "output": 0, "calls": 0},
            "call_3_summarization":    {"input": 0, "output": 0, "calls": 0},
        }

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
        """Propagate LLMCallLogger to all sub-modules and store locally."""
        self._llm_logger = llm_logger
        if self._memory_bank:
            self._memory_bank.set_llm_logger(llm_logger)
        if self._personas:
            self._personas.set_llm_logger(llm_logger)
        if self._generator:
            self._generator.set_llm_logger(llm_logger)

    @property
    def memory_bank(self) -> "EventMemory":
        return self._memory_bank

    @property
    def personas(self) -> "Personas":
        return self._personas

    @property
    def generator(self) -> "Generator":
        return self._generator

    # =========================================================================
    # PER-CALL-TYPE TOKEN ACCUMULATION
    # =========================================================================

    def accumulate_tokens(self, call_type: str, input_t: int, output_t: int, calls: int = 1) -> None:
        t = self._call_tokens[call_type]
        t["input"]  += input_t
        t["output"] += output_t
        t["calls"]  += calls

    def get_and_reset_token_counts_by_type(self) -> Dict[str, Dict[str, int]]:
        """Return Phase-1 token counts by call type and reset to zero."""
        result = {
            k: {"input": v["input"], "output": v["output"], "llm_calls": v["calls"]}
            for k, v in self._call_tokens.items()
        }
        for v in self._call_tokens.values():
            v["input"] = v["output"] = v["calls"] = 0
        return result

    # =========================================================================
    # MEMORY STATS
    # =========================================================================

    def get_memory_stats(self) -> Dict[str, int]:
        """Return LTM entry count and estimated content tokens."""
        return self._memory_bank.get_memory_stats()

    # =========================================================================
    # STM → LTM FLUSH  (before QA)
    # =========================================================================

    def flush_to_ltm(self, sample_id: str) -> Dict[str, int]:
        """
        Force-commit remaining STM content to LTM before running QA.
        STM is kept intact after flush for QA context.
        Returns token info dict for the summarise call.
        """
        return self._memory_bank.flush_stm(current_sample_id=sample_id)

    # =========================================================================
    # MAIN TURN PROCESSING
    # =========================================================================

    def process_turn(
        self,
        speaker_name: str,
        text:         str,
        dia_id:       str,
        timestamp:    float,
        session_num:  int,
        date_time:    str,
        sample_id:    str,
    ) -> TurnResult:
        """
        Process one dialogue turn (Phase 1, LoComo).

        Execution order:
          1. store_turn() → STM (may trigger session-boundary flush to LTM)
          2. Route persona update by speaker name
        """
        mb  = self._memory_bank
        per = self._personas

        internal = {"input": 0, "output": 0, "calls": 0}

        # ── 1. Store turn in STM (triggers flush at session boundaries) ──────
        mb.store_turn(
            speaker_name=speaker_name,
            text=text,
            dia_id=dia_id,
            timestamp=timestamp,
            session_num=session_num,
            date_time=date_time,
            sample_id=sample_id,
        )
        summ_tokens = _read_token_attr(mb, "last_summarize_token_info")
        if summ_tokens.get("input", 0) > 0:
            _accum_internal(internal, summ_tokens)

        # ── 2. Persona update (route by speaker) ─────────────────────────────
        if speaker_name == self._speaker_a:
            per._speaker_a_traits_update(text)
            _accum_internal(internal, _read_token_attr(per, "last_speaker_a_token_info"), count=1)
        else:
            per._speaker_b_traits_update(text)
            _accum_internal(internal, _read_token_attr(per, "last_speaker_b_token_info"), count=1)

        return TurnResult(
            internal_token_info=internal,
            retrieval_log_data={},
        )

    # =========================================================================
    # QA ANSWERING
    # =========================================================================

    def get_qa_answer(
        self,
        question:           str,
        category:           int,
        adversarial_answer: Optional[str] = None,
    ) -> QAResult:
        """
        Answer a QA question using accumulated LTM + STM context.
        QA exchanges are never stored in memory.
        """
        mb  = self._memory_bank
        per = self._personas
        gen = self._generator
        cfg = self._config

        # ── LTM retrieval ────────────────────────────────────────────────────
        relevant_memories = mb.relevance_retrieve(
            ori_query=question,
            n_results=cfg.RETRIEVE_K,
            current_timestamp=mb.current_timestamp,
        )
        memories_str = format_memories_for_prompt(relevant_memories)

        # ── STM context ──────────────────────────────────────────────────────
        stm_context = mb.get_stm_context()
        context_str = format_context_for_prompt(stm_context)

        # ── QA generation ────────────────────────────────────────────────────
        speaker1_traits, speaker2_traits = per.get_current_traits()
        answer, token_info, prompt_snapshot, num_llm_calls = gen.generate_qa_answer(
            question=question,
            category=category,
            context=context_str,
            memories=memories_str,
            speaker1_traits=speaker1_traits,
            speaker2_traits=speaker2_traits,
            adversarial_answer=adversarial_answer,
        )

        # ── Build retrieved_memories ──────────────────────────────────────────
        retrieved_memories = []
        for mem in relevant_memories:
            dia_ids_str = mem.get("dia_ids", "")
            dia_ids     = [d for d in dia_ids_str.split(",") if d] if dia_ids_str else []
            summary     = mem.get("summary", "")
            score       = mem.get("score", 0.0)
            if dia_ids:
                for did in dia_ids:
                    retrieved_memories.append({
                        "dia_id":          did,
                        "content_preview": summary[:100],
                        "score":           score,
                    })
            else:
                retrieved_memories.append({
                    "dia_id":          "",
                    "content_preview": summary[:100],
                    "score":           score,
                })

        retrieval_log_data = {
            "query":             question,
            "relevant_memories": relevant_memories,
            "prompt_snapshot":   prompt_snapshot,
            "module_specific": {
                "ltm_entry_count":       mb.get_memory_count(),
                "stm_context_turns":     len(stm_context),
                "speaker1_trait_count":  _count_traits(speaker1_traits),
                "speaker2_trait_count":  _count_traits(speaker2_traits),
            },
        }

        return QAResult(
            answer=answer,
            token_info=token_info,
            retrieved_memories=retrieved_memories,
            retrieval_log_data=retrieval_log_data,
            num_llm_calls=num_llm_calls,
        )
