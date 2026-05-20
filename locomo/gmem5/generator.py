"""
Prompt construction and QA generation for GraphMem v5 on LoComo.
"""

import hashlib
import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


_NODE_TYPE_DESC = (
    "Node types:\n"
    "  State:  A speaker-specific condition that is currently or recently valid and may change over time. It captures the speaker's present stance, ongoing goal, constraint, situation, or preference shift. States are time-bounded and context-sensitive.\n"
    "  Trait:  A generalized speaker characteristic that persists across situations and time. It represents recurring dispositions, stable preferences, values, or habitual tendencies. Traits are cross-situational and relatively context-independent.\n"
    "  Memory: An episodic summary of what happened during a recent conversation. It captures concrete events, topics, and actions at a particular time, not generalized speaker attributes."
)

_QA_COMMON_RETRIEVAL_GUIDE = (
    "The information below describes what is known about the speakers from past conversations.\n"
    + _NODE_TYPE_DESC + "\n"
    "\n"
    "[Traits] are currently reliable; [Challenged Traits] may be outdated, replaced, or contradicted. "
    "For a challenged trait, prefer any \"shifted to\" entry and weigh listed conflicting evidence before using the trait.\n"
    "\n"
    "[Current Constraints] are high-impact speaker states that should be honored unless the current question explicitly overrides them.\n"
    "\n"
    "Each retrieved item is annotated with an absolute date (e.g. \"[8 May 2023]\") indicating when it was observed in the conversation."
)

_SYS_QA_BASE = (
    "You are a helpful assistant answering a question about a multi-session conversation between two named speakers, based on memory.\n"
    + _QA_COMMON_RETRIEVAL_GUIDE + "\n"
    "\n"
    "Use the retrieved memory to tailor your answer. If any item — even one off the question's topic — describes a constraint, situation, or trait that would change a standard answer, incorporate it. Do not give a generic answer when a relevant speaker circumstance is available.\n"
    "\n"
    "Answer naturally; do not say \"based on what I remember\" or similar."
)


# Category 1 (single-hop), 2 (multi-hop), 4 (open-domain): short phrase answer
SYS_QA_C1_C2_C4 = _SYS_QA_BASE

QA_PROMPT_C1_C2_C4 = """\
[Question]
{question}

[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

Answer in the form of a short phrase based on the retrieved memory (maximum 50 words).
Use exact words from the retrieved memory whenever possible.
Output JSON: {{"answer": "..."}}.\
"""

QA_SCHEMA_C1_C2_C4 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

# Category 3 (temporal): approximate date or time reasoning
SYS_QA_C3 = _SYS_QA_BASE

QA_PROMPT_C3 = """\
[Question]
{question}

[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

Use the date shown in brackets to answer with an approximate date.
Generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.
Output JSON: {{"answer": "..."}}.\
"""

QA_SCHEMA_C3 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

# Category 5 (adversarial): binary choice
SYS_QA_C5 = (
    "You are a helpful assistant answering a question about a multi-session conversation between two named speakers, based on memory.\n"
    + _QA_COMMON_RETRIEVAL_GUIDE + "\n"
    "\n"
    "Use the retrieved memory to choose the option that fits this speaker's situation.\n"
    "You must choose exactly one of the two provided options."
)

QA_PROMPT_C5 = """\
[Question]
{question}

[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

Choose exactly one of the following options:
Option A: {option_a}
Option B: {option_b}

If the retrieved memory does not contain evidence for either option, prefer the "Not mentioned in the conversation" option.
Output JSON with "choice" ("A" or "B") and "answer" (the full text of the chosen option).\
"""

QA_SCHEMA_C5 = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B"]},
        "answer": {"type": "string"},
    },
    "required": ["choice", "answer"],
    "additionalProperties": False,
}

_PROMPT_TOO_LONG_RE = re.compile(
    r"decoder prompt \(length \d+\).*maximum model length",
    re.IGNORECASE | re.DOTALL,
)
_RETRY_MAX = 3


def _c5_choice_order(adversarial_answer: str, seed: str) -> Tuple[str, str]:
    """Deterministically assign adversarial_answer to option A or B based on seed hash."""
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    not_mentioned = "Not mentioned in the conversation"
    if h % 2 == 0:
        return adversarial_answer, not_mentioned   # A=adversarial, B=not_mentioned
    return not_mentioned, adversarial_answer       # A=not_mentioned, B=adversarial


class GraphGenerator:
    """Builds response prompts and runs QA generation for LoComo 5-category QA."""

    def __init__(self, llm_client, model_path: str, config, llm_logger=None) -> None:
        self._llm = llm_client
        self._model_path = model_path
        self._cfg = config
        self._llm_logger = llm_logger

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    def build_qa_prompt(
        self,
        question: str,
        retrieved_memory: str,
        category: int,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
    ) -> Tuple[str, str, Dict, float]:
        """Returns (prompt, system_prompt, schema, temperature)."""
        mem = retrieved_memory or "No memory available."

        if category == 3:
            return (
                QA_PROMPT_C3.format(retrieved_memory=mem, question=question),
                SYS_QA_C3,
                QA_SCHEMA_C3,
                self._cfg.TEMPERATURE,
            )

        if category == 5:
            option_a, option_b = _c5_choice_order(adversarial_answer, choice_order_seed)
            return (
                QA_PROMPT_C5.format(
                    retrieved_memory=mem,
                    question=question,
                    option_a=option_a,
                    option_b=option_b,
                ),
                SYS_QA_C5,
                QA_SCHEMA_C5,
                self._cfg.TEMPERATURE_C5,
            )

        # categories 1, 2, 4
        return (
            QA_PROMPT_C1_C2_C4.format(retrieved_memory=mem, question=question),
            SYS_QA_C1_C2_C4,
            QA_SCHEMA_C1_C2_C4,
            self._cfg.TEMPERATURE,
        )

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str,
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
    ) -> Tuple[str, Dict, str]:
        prompt, system_prompt, schema, temperature = self.build_qa_prompt(
            question, retrieved_memory, category, adversarial_answer, choice_order_seed
        )
        memory_str = retrieved_memory or "No memory available."

        for _ in range(_RETRY_MAX):
            try:
                result = self._llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    guided_json=schema,
                    temperature=temperature,
                    max_tokens=self._cfg.MAX_TOKENS,
                    json_retry=self._cfg.JSON_RETRY,
                    return_usage=True,
                )
                if self._llm_logger is not None:
                    self._llm_logger.log("call_6_qa", system_prompt, prompt, result)
                answer = self._extract_answer(result, category, adversarial_answer, choice_order_seed)
                return answer, _extract_token_info(result, self._model_path), prompt
            except Exception as exc:
                if not _is_prompt_too_long(exc):
                    logger.error(f"QA generation failed: {exc}")
                    break
                old_len = len(memory_str)
                memory_str = memory_str[: max(old_len // 2, 1)]
                prompt, system_prompt, schema, temperature = self.build_qa_prompt(
                    question, memory_str, category, adversarial_answer, choice_order_seed
                )

        return "", {"input": 0, "output": 0, "model": self._model_path}, prompt

    def parse_qa_result(
        self,
        result,
        category: int,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
    ) -> Tuple[str, Dict]:
        answer = self._extract_answer(result, category, adversarial_answer, choice_order_seed)
        return answer, _extract_token_info(result, self._model_path)

    def _extract_answer(
        self,
        result,
        category: int,
        adversarial_answer: str,
        choice_order_seed: str,
    ) -> str:
        if not isinstance(result, dict):
            return ""
        if category == 5:
            choice = result.get("choice", "").strip().upper()
            option_a, option_b = _c5_choice_order(adversarial_answer, choice_order_seed)
            if choice == "A":
                return option_a
            if choice == "B":
                return option_b
            return result.get("answer", "")
        return result.get("answer", "")

    def _count_prompt_tokens(self, prompt: str) -> int:
        inner = getattr(self._llm, "client", None)
        tokenizer = getattr(inner, "tokenizer", None) if inner else None
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(prompt))
            except Exception:
                pass
        return len(prompt) // 4


def _extract_token_info(result, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(result, dict) and "_usage" in result:
        usage = result["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def _is_prompt_too_long(error: Exception) -> bool:
    return bool(_PROMPT_TOO_LONG_RE.search(str(error)))
