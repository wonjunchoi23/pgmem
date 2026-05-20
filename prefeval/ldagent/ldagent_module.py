"""
LD-Agent Module — PrefEval port.

Trimmed from exp_implexconv_no_response/ldagent/ldagent_module.py:
- Removed `subset` parameter from get_qa_answer().
- Added load_snapshot() helper.

Otherwise unchanged: wraps EventMemory + Personas + Generator,
same token tracking and per-call logging.
"""

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from event_memory import EventMemory
from generator    import Generator, format_memories_for_prompt
from personas     import Personas
from load_dataset import compute_virtual_seconds

logger = logging.getLogger(__name__)


# =============================================================================
# LLM CALL LOGGER
# =============================================================================

class LLMCallLogger:
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
    response:           str
    internal_token_info: Dict[str, int] = field(default_factory=dict)
    retrieval_log_data:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    answer:             str
    token_info:         Dict[str, int] = field(default_factory=dict)
    retrieved_memories: List[Dict]     = field(default_factory=list)
    retrieval_log_data: Dict[str, Any] = field(default_factory=dict)
    num_llm_calls:      int            = 0


# =============================================================================
# HELPERS
# =============================================================================

def _read_token_attr(obj, attr: str) -> Dict[str, int]:
    info = getattr(obj, attr, None)
    return info if isinstance(info, dict) else {"input": 0, "output": 0}


def _accum_internal(dst: Dict, token_info: Dict, count: int = 1):
    dst["input"]  += token_info.get("input",  0)
    dst["output"] += token_info.get("output", 0)
    dst["calls"]  += count


def _count_traits(traits_str: str) -> int:
    return len([t for t in traits_str.split("\n") if t.strip()]) if traits_str else 0


# =============================================================================
# LD-AGENT MODULE
# =============================================================================

class LDAgentModule:
    """LD-Agent module wrapping EventMemory + Personas + Generator (single-chain variant)."""

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

    def load_snapshot(self, directory: Path):
        if self._memory_bank:
            self._memory_bank.load_snapshot(directory)
        if self._personas:
            self._personas.load_snapshot(directory)

    def set_llm_logger(self, llm_logger):
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
    # FLUSH (unused in chain mode — kept for parity)
    # =========================================================================

    def flush_to_ltm(self, session_id: int) -> Dict[str, int]:
        return self._memory_bank.flush_stm(current_session_id=session_id)

    # =========================================================================
    # MEMORY STATS
    # =========================================================================

    def get_memory_stats(self) -> Dict[str, int]:
        return self._memory_bank.get_memory_stats()

    # =========================================================================
    # PER-TURN PROCESSING (sequential — Q2 = a)
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
        Sequential per-turn pipeline (matches original LD-Agent).
        Per turn LLM calls: call_4_summarization (only at boundary), call_2_user_persona,
        call_3_agent_persona.
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

        # context_retrieve handles the boundary summarization (lazy)
        mb.context_retrieve(
            query=user_utterance,
            current_virtual_seconds=vs,
            current_conv_id=conv_id,
            current_session_id=session_id,
        )
        summ_tokens = _read_token_attr(mb, "last_summarize_token_info")
        if summ_tokens.get("input", 0) > 0:
            _accum_internal(internal, summ_tokens)

        per._user_traits_update(user_utterance)
        _accum_internal(internal, _read_token_attr(per, "last_user_token_info"), count=1)

        per._agent_traits_update(gt_response)
        _accum_internal(internal, _read_token_attr(per, "last_agent_token_info"), count=1)

        mb.add_agent_response(
            response=gt_response,
            current_virtual_seconds=vs,
            current_session_id=session_id,
        )

        return TurnResult(response="", internal_token_info=internal, retrieval_log_data={})

    # =========================================================================
    # QA ANSWERING
    # =========================================================================

    def _format_stm_context_for_qa(self) -> str:
        mb = self._memory_bank
        if not mb.short_term_memory:
            return ""
        lines = [f"[TURN {i}] : {m['dialog']}." for i, m in enumerate(mb.short_term_memory)]
        return "\n".join(lines)

    def get_qa_answer(self, question: str) -> QAResult:
        """
        Sequential single-question QA. Runner uses step-wise build/batch path
        for batched QA at a checkpoint, but this method is kept for parity.
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

        return QAResult(
            answer=answer,
            token_info=token_info,
            retrieved_memories=retrieved_metadata,
            retrieval_log_data={
                "query":             question,
                "relevant_memories": relevant_memories,
                "prompt_snapshot":   prompt_snapshot,
                "module_specific": {
                    "ltm_entry_count":   mb.get_memory_count(),
                    "stm_context_turns": len(mb.short_term_memory),
                    "user_trait_count":  _count_traits(user_traits),
                    "agent_trait_count": _count_traits(agent_traits),
                },
            },
            num_llm_calls=num_llm_calls,
        )
