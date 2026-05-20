"""
A-MEM Agent Module — LoComo batch-enabled variant
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config as cfg

from memory_layer import AgenticMemorySystem

logger = logging.getLogger(__name__)


# =============================================================================
# QA PROMPT TEMPLATES
# =============================================================================

QA_PROMPT_DEFAULT = (
    "Based on the context: {context}, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context whenever possible.\n\n"
    "Question: {question} Short answer:"
)

QA_PROMPT_TEMPORAL = (
    "Based on the context: {context}, answer the following question. "
    "Use DATE of CONVERSATION to answer with an approximate date.\n"
    "Please generate the shortest possible answer, using words from the conversation "
    "where possible, and avoid using any subjects.\n\n"
    "Question: {question} Short answer:"
)

QA_PROMPT_ADVERSARIAL = (
    "Based on the context: {context}, answer the following question. {question}\n\n"
    "Select the correct answer: {choice_a} or {choice_b}  Short answer:"
)


# =============================================================================
# JSON SCHEMA
# =============================================================================

QA_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


# =============================================================================
# PROMPT-TOO-LONG RETRY HELPER
# =============================================================================

_PROMPT_TOO_LONG_RE = re.compile(
    r"decoder prompt \(length (\d+)\).*?maximum model length of (\d+)",
    re.IGNORECASE | re.DOTALL,
)
_CHARS_PER_TOKEN = 4
_MEMORY_RETRY_MAX = 3


def _parse_prompt_too_long(error: Exception):
    m = _PROMPT_TOO_LONG_RE.search(str(error))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# =============================================================================
# TOKEN HELPER
# =============================================================================

def extract_token_info(response, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(response, dict) and "_usage" in response:
        usage = response["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def _deterministic_bool(seed_text: str) -> bool:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[-1], 16) % 2 == 0


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent:
    """
    A-MEM agent wrapping AgenticMemorySystem.
    """

    def __init__(self, llm_client, model_path: str = "", embedding_model=None):
        self.llm_client = llm_client
        self.model_path = model_path
        self.memory_system = AgenticMemorySystem(
            llm_client=llm_client,
            model_name=cfg.EMBEDDING_MODEL,
            embedding_model=embedding_model,
            evo_threshold=cfg.EVOLUTION_THRESHOLD,
            json_retry=cfg.JSON_RETRY,
        )
        self._llm_logger = None

    # ------------------------------------------------------------------
    # LLM call logger
    # ------------------------------------------------------------------

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger
        self.memory_system.llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def add_memory(self, content: str, time: Optional[str] = None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, query: str, k: int = None) -> str:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.find_related_memories_raw(query, k=k)

    def retrieve_memory_with_metadata(
        self, query: str, k: int = None
    ) -> Tuple[str, List[Dict]]:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.find_related_memories_with_metadata(query, k=k)

    def retrieve_for_log(self, query: str, k: int = None) -> Tuple[List[Dict], int]:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.retrieve_for_log(query, k=k)

    # ------------------------------------------------------------------
    # QA prompt building
    # ------------------------------------------------------------------

    def build_qa_prompt(
        self,
        question: str,
        retrieved_memory: str,
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: Optional[str] = None,
    ) -> Tuple[str, float]:
        memory_str = retrieved_memory if retrieved_memory else "No memory available."

        if category == 5:
            not_mentioned = "Not mentioned in the conversation"
            seed = choice_order_seed or f"{question}::{adversarial_answer}"
            if _deterministic_bool(seed):
                choice_a, choice_b = adversarial_answer, not_mentioned
            else:
                choice_a, choice_b = not_mentioned, adversarial_answer
            prompt = QA_PROMPT_ADVERSARIAL.format(
                context=memory_str,
                question=question,
                choice_a=choice_a,
                choice_b=choice_b,
            )
            return prompt, cfg.TEMPERATURE_C5

        if category == 3:
            prompt = QA_PROMPT_TEMPORAL.format(
                context=memory_str,
                question=question,
            )
            return prompt, cfg.TEMPERATURE

        prompt = QA_PROMPT_DEFAULT.format(
            context=memory_str,
            question=question,
        )
        return prompt, cfg.TEMPERATURE

    # ------------------------------------------------------------------
    # QA answering
    # ------------------------------------------------------------------

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str = None,
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: Optional[str] = None,
    ) -> Tuple[str, Dict, str, int]:
        if retrieved_memory is None:
            retrieved_memory = self.retrieve_memory(question)

        memory_str = retrieved_memory if retrieved_memory else "No memory available."

        for attempt in range(_MEMORY_RETRY_MAX):
            prompt, temperature = self.build_qa_prompt(
                question=question,
                retrieved_memory=memory_str,
                category=category,
                adversarial_answer=adversarial_answer,
                choice_order_seed=choice_order_seed,
            )
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=(
                        "You are a helpful assistant answering a question about a user "
                        "based on their conversation history stored in memory. "
                        "Respond in JSON format with an 'answer' field."
                    ),
                    guided_json=QA_SCHEMA,
                    temperature=temperature,
                    max_tokens=cfg.MAX_TOKENS,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                if self._llm_logger is not None:
                    self._llm_logger.log("call_3_qa", "", prompt, result)
                token_info = extract_token_info(result, self.model_path)
                answer = result.get("answer", "") if isinstance(result, dict) else ""
                return answer, token_info, prompt, 1

            except Exception as e:
                parsed = _parse_prompt_too_long(e)
                if parsed and attempt < _MEMORY_RETRY_MAX - 1:
                    prompt_len, max_len = parsed
                    cut_chars = (prompt_len - max_len + 200) * _CHARS_PER_TOKEN
                    new_len = max(len(memory_str) - cut_chars, 200)
                    logger.warning(
                        f"[answer_qa] Prompt too long "
                        f"({prompt_len} > {max_len} tokens). "
                        f"Truncating memory {len(memory_str)} -> {new_len} chars "
                        f"(attempt {attempt + 1}/{_MEMORY_RETRY_MAX})"
                    )
                    memory_str = memory_str[:new_len]
                else:
                    logger.error(f"Error answering QA: {e}")
                    return "", {"input": 0, "output": 0, "model": self.model_path}, prompt, 0

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def clear_memory(self):
        self.memory_system.clear()

    def save_memory_snapshot(self, directory: Path):
        self.memory_system.save_snapshot(directory)

    def get_memory_count(self) -> int:
        return self.memory_system.get_memory_count()

    def get_memory_stats(self) -> Dict:
        return self.memory_system.get_memory_stats()

    def get_and_reset_internal_stats(self) -> Dict:
        return self.memory_system.get_and_reset_internal_stats()

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_and_reset_memory_tokens(self) -> Dict:
        """Return per-call-type token counts and reset."""
        return self.memory_system.get_and_reset_token_counts_by_type()

    def accumulate_memory_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_calls: int = 1,
        call_type: str = "call_1_note_construction",
    ):
        """Accumulate token counts from an external batch LLM call."""
        self.memory_system.accumulate_token_counts(input_tokens, output_tokens, llm_calls, call_type)
