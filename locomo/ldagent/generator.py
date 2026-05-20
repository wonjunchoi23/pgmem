"""
Response Generator Module for LD-Agent (LoComo) — Batch-enabled variant

Extends ldagent/generator.py with a batch-friendly prompt builder.

Changes from ldagent/generator.py:
  - import hashlib added
  - _deterministic_choice_order() added — hash-based choice ordering for cat5
    so that the answer order is reproducible across runs and in batch mode
  - build_qa_prompt_for_batch() added — returns (combined_prompt, temperature)
    where sys_prompt and user_prompt are merged for generate_batch_raw()
    (system_prompt=None), and cat5 ordering uses _deterministic_choice_order
  - All original methods preserved unchanged (generate_qa_answer still uses
    random.shuffle for sequential sequential-mode compatibility)
"""

import hashlib
import json
import logging
import random
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# JSON SCHEMAS
# =============================================================================

QA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


# =============================================================================
# GENERATOR CLASS
# =============================================================================

class Generator:
    """
    QA generation module for LD-Agent (LoComo batch variant).
    See ldagent/generator.py for full documentation.
    """

    def __init__(
        self,
        llm_client,
        logger:         logging.Logger,
        speaker_a:      str   = "Speaker A",
        speaker_b:      str   = "Speaker B",
        max_tokens:     int   = 750,
        temperature:    float = 0.7,
        temperature_c5: float = 0.5,
        json_retry:     int   = 3,
    ):
        self.llm_client     = llm_client
        self.logger         = logger
        self.speaker_a      = speaker_a
        self.speaker_b      = speaker_b
        self.max_tokens     = max_tokens
        self.temperature    = temperature
        self.temperature_c5 = temperature_c5
        self.json_retry     = json_retry
        self.llm_logger     = None

        logger.info(
            f"Generator initialized (speaker_a={speaker_a}, speaker_b={speaker_b})"
        )

    # =========================================================================
    # LLM LOGGER INJECTION
    # =========================================================================

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

    # =========================================================================
    # QA PROMPT BUILDERS  (unchanged from ldagent/)
    # =========================================================================

    def _build_context_block(
        self,
        context:         str,
        memories:        str,
        speaker1_traits: str,
        speaker2_traits: str,
    ) -> str:
        parts = []
        if context:
            parts.append(f"<CONTEXT>\nRecent conversation turns:\n{context}")
        parts.append(f"<MEMORIES>\nConversation memories:\n{memories}")
        if speaker1_traits:
            parts.append(
                f"<{self.speaker_a.upper()}_TRAITS>\n"
                f"{self.speaker_a}'s characteristics:\n{speaker1_traits}"
            )
        if speaker2_traits:
            parts.append(
                f"<{self.speaker_b.upper()}_TRAITS>\n"
                f"{self.speaker_b}'s characteristics:\n{speaker2_traits}"
            )
        return "\n\n".join(parts)

    def _build_qa_prompt_default(
        self,
        question:        str,
        context:         str,
        memories:        str,
        speaker1_traits: str,
        speaker2_traits: str,
    ) -> Tuple[str, str]:
        sys_prompt = (
            f"You are a helpful assistant that answers questions about a conversation "
            f"between {self.speaker_a} and {self.speaker_b} based on recorded memories."
        )
        ctx_block   = self._build_context_block(context, memories, speaker1_traits, speaker2_traits)
        user_prompt = (
            f"{ctx_block}\n\n"
            f"Based on the context above, write an answer in the form of a short phrase "
            f"for the following question. Answer with exact words from the context "
            f"whenever possible.\n\n"
            f"Question: {question} Short answer:"
        )
        return sys_prompt, user_prompt

    def _build_qa_prompt_temporal(
        self,
        question:        str,
        context:         str,
        memories:        str,
        speaker1_traits: str,
        speaker2_traits: str,
    ) -> Tuple[str, str]:
        sys_prompt = (
            f"You are a helpful assistant that answers questions about a conversation "
            f"between {self.speaker_a} and {self.speaker_b} based on recorded memories."
        )
        ctx_block   = self._build_context_block(context, memories, speaker1_traits, speaker2_traits)
        user_prompt = (
            f"{ctx_block}\n\n"
            f"Based on the context above, answer the following question. "
            f"Use DATE OF CONVERSATION to answer with an approximate date. "
            f"Please generate the shortest possible answer, using words from the "
            f"conversation where possible, and avoid using any subjects.\n\n"
            f"Question: {question} Short answer:"
        )
        return sys_prompt, user_prompt

    def _build_qa_prompt_adversarial(
        self,
        question:           str,
        adversarial_answer: str,
        context:            str,
        memories:           str,
        speaker1_traits:    str,
        speaker2_traits:    str,
    ) -> Tuple[str, str, str]:
        """Sequential adversarial prompt (uses random.shuffle — unchanged)."""
        not_mentioned = "Not mentioned in the conversation"
        choices = [adversarial_answer, not_mentioned]
        random.shuffle(choices)
        choice_a, choice_b = choices[0], choices[1]

        sys_prompt = (
            f"You are a helpful assistant that answers questions about a conversation "
            f"between {self.speaker_a} and {self.speaker_b} based on recorded memories."
        )
        ctx_block   = self._build_context_block(context, memories, speaker1_traits, speaker2_traits)
        user_prompt = (
            f"{ctx_block}\n\n"
            f"Based on the context above, answer the following question. {question}\n\n"
            f"Select the correct answer: {choice_a} or {choice_b}  Short answer:"
        )
        return sys_prompt, user_prompt, adversarial_answer

    # =========================================================================
    # SEQUENTIAL QA GENERATION  (unchanged from ldagent/)
    # =========================================================================

    def generate_qa_answer(
        self,
        question:           str,
        category:           int,
        context:            str           = "",
        memories:           str           = "",
        speaker1_traits:    str           = "",
        speaker2_traits:    str           = "",
        adversarial_answer: Optional[str] = None,
    ) -> Tuple[str, Dict[str, int], str, int]:
        temp = self.temperature_c5 if category == 5 else self.temperature

        if category == 3:
            sys_prompt, user_prompt = self._build_qa_prompt_temporal(
                question, context, memories, speaker1_traits, speaker2_traits
            )
        elif category == 5:
            adv = adversarial_answer or ""
            sys_prompt, user_prompt, _ = self._build_qa_prompt_adversarial(
                question, adv, context, memories, speaker1_traits, speaker2_traits
            )
        else:
            sys_prompt, user_prompt = self._build_qa_prompt_default(
                question, context, memories, speaker1_traits, speaker2_traits
            )

        prompt_snapshot = f"[SYSTEM]\n{sys_prompt}\n[USER]\n{user_prompt}"
        token_info = {"input": 0, "output": 0}

        try:
            result = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                guided_json=QA_RESPONSE_SCHEMA,
                max_tokens=self.max_tokens,
                temperature=temp,
                json_retry=self.json_retry,
                return_usage=True,
            )

            if isinstance(result, dict) and "_usage" in result:
                usage = result["_usage"]
                token_info = {
                    "input":  usage.get("prompt_tokens",     0),
                    "output": usage.get("completion_tokens", 0),
                }

            if self.llm_logger is not None:
                self.llm_logger.log("call_4_qa", sys_prompt, user_prompt, result)

            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            answer = str(answer).strip()

        except Exception as e:
            self.logger.error(f"Error in generate_qa_answer (cat={category}): {e}")
            return "", token_info, prompt_snapshot, 0

        return answer, token_info, prompt_snapshot, 1

    # =========================================================================
    # BATCH API  (new in ldagent_batch)
    # =========================================================================

    def _deterministic_choice_order(
        self, adversarial_answer: str, seed: str
    ) -> Tuple[str, str]:
        """
        Return (choice_a, choice_b) with deterministic ordering based on seed.
        Uses SHA-256 hash of seed to decide which option comes first.
        """
        not_mentioned = "Not mentioned in the conversation"
        digest        = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        if int(digest[-1], 16) % 2 == 0:
            return adversarial_answer, not_mentioned
        else:
            return not_mentioned, adversarial_answer

    def build_qa_prompt_for_batch(
        self,
        question:           str,
        category:           int,
        context:            str           = "",
        memories:           str           = "",
        speaker1_traits:    str           = "",
        speaker2_traits:    str           = "",
        adversarial_answer: Optional[str] = None,
        choice_order_seed:  Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Build QA prompt for batch inference.

        Returns:
            (combined_prompt, temperature)

        combined_prompt merges sys_prompt and user_prompt into a single string
        so it can be passed to generate_batch_raw(system_prompt=None).
        This allows batching across samples that have different speaker names
        (which would otherwise require different system prompts).

        For category 5, choice ordering is deterministic via choice_order_seed
        (SHA-256 hash). Use seed = f"{sample_id}::{qa_idx}::{question}" to
        ensure reproducibility across runs.
        """
        temp = self.temperature_c5 if category == 5 else self.temperature

        if category == 3:
            sys_p, usr_p = self._build_qa_prompt_temporal(
                question, context, memories, speaker1_traits, speaker2_traits
            )

        elif category == 5:
            adv  = adversarial_answer or ""
            seed = choice_order_seed or f"{question}::{adv}"
            choice_a, choice_b = self._deterministic_choice_order(adv, seed)
            sys_p     = (
                f"You are a helpful assistant that answers questions about a conversation "
                f"between {self.speaker_a} and {self.speaker_b} based on recorded memories."
            )
            ctx_block = self._build_context_block(
                context, memories, speaker1_traits, speaker2_traits
            )
            usr_p = (
                f"{ctx_block}\n\n"
                f"Based on the context above, answer the following question. {question}\n\n"
                f"Select the correct answer: {choice_a} or {choice_b}  Short answer:"
            )

        else:  # categories 1, 2, 4
            sys_p, usr_p = self._build_qa_prompt_default(
                question, context, memories, speaker1_traits, speaker2_traits
            )

        combined_prompt = f"{sys_p}\n\n{usr_p}"
        return combined_prompt, temp


# =============================================================================
# UTILITY FUNCTIONS  (unchanged from ldagent/)
# =============================================================================

def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return "No relevant Memories."

    formatted = []
    for mem in memories:
        date_time = mem.get("date_time", "")
        summary   = mem.get("summary", mem.get("dialog", "Unknown"))
        if date_time:
            formatted.append(f"[{date_time}] {summary}")
        else:
            formatted.append(f"{summary}")

    return "\n".join(formatted)


def format_context_for_prompt(context_memories: List[Dict[str, Any]]) -> str:
    if not context_memories:
        return "No previous context."
    return "\n".join(
        f"[TURN {m.get('idx', 0)}] : {m.get('dialog', '')}"
        for m in context_memories
    )
