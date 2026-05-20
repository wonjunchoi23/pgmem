"""Prompt construction and QA generation for GraphMem v6 on LoComo."""

import hashlib
import logging
import re
from typing import Dict, Tuple

from updater import _NODE_TYPE_DESC

logger = logging.getLogger(__name__)

_QA_COMMON_RETRIEVAL_GUIDE = (
    _NODE_TYPE_DESC + "\n"
    "\n"
    "Each retrieved item is annotated with an absolute date (e.g. \"[8 May 2023]\") indicating when it was observed in the conversation."
)

_SYS_QA_BASE = (
    "You are a helpful assistant answering a question about a multi-session conversation between two named speakers, based on retrieved episodes.\n"
    "\n"
    + _QA_COMMON_RETRIEVAL_GUIDE + "\n"
    "\n"
    "Respond in JSON format with an 'answer' field."
)


# Category 1 (multi-hop), 3 (open-domain), 4 (single-hop): short phrase answer
SYS_QA_C1_C3_C4 = _SYS_QA_BASE

QA_PROMPT_C1_C3_C4 = """\
[Question]
{question}

[Retrieved Episode]
{retrieved_episode}

[Question]
{question}

Use exact words from the retrieved memory whenever possible. Answer concisely.\
"""

QA_SCHEMA_C1_C3_C4 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

# Category 2 (temporal): approximate date or time reasoning
SYS_QA_C2 = _SYS_QA_BASE

QA_PROMPT_C2 = """\
[Question]
{question}

[Retrieved Episode]
{retrieved_episode}

[Question]
{question}

Use the date shown in brackets to answer with an approximate date.
Generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects. Be concise.\
"""

QA_SCHEMA_C2 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

# Category 5 (adversarial): binary choice
SYS_QA_C5 = (
    _SYS_QA_BASE + "\n"
    "\n"
    "You must choose exactly one of the two provided options."
)

QA_PROMPT_C5 = """\
[Question]
{question}

[Retrieved Episode]
{retrieved_episode}

[Question]
{question}

Select the correct answer: {choice_a} or {choice_b}  Short answer:\
"""

QA_SCHEMA_C5 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
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
        retrieved_episode: str,
        category: int,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
    ) -> Tuple[str, str, Dict, float]:
        """Returns (prompt, system_prompt, schema, temperature)."""
        ep = retrieved_episode or "No episodes available."

        if category == 2:
            return (
                QA_PROMPT_C2.format(retrieved_episode=ep, question=question),
                SYS_QA_C2,
                QA_SCHEMA_C2,
                self._cfg.TEMPERATURE,
            )

        if category == 5:
            choice_a, choice_b = _c5_choice_order(adversarial_answer, choice_order_seed)
            return (
                QA_PROMPT_C5.format(
                    retrieved_episode=ep,
                    question=question,
                    choice_a=choice_a,
                    choice_b=choice_b,
                ),
                SYS_QA_C5,
                QA_SCHEMA_C5,
                self._cfg.TEMPERATURE_C5,
            )

        # categories 1, 3, 4
        return (
            QA_PROMPT_C1_C3_C4.format(retrieved_episode=ep, question=question),
            SYS_QA_C1_C3_C4,
            QA_SCHEMA_C1_C3_C4,
            self._cfg.TEMPERATURE,
        )

    def answer_qa(
        self,
        question: str,
        retrieved_episode: str,
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
    ) -> Tuple[str, Dict, str]:
        prompt, system_prompt, schema, temperature = self.build_qa_prompt(
            question, retrieved_episode, category, adversarial_answer, choice_order_seed
        )
        episode_str = retrieved_episode or "No episodes available."

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
                old_len = len(episode_str)
                episode_str = episode_str[: max(old_len // 2, 1)]
                prompt, system_prompt, schema, temperature = self.build_qa_prompt(
                    question, episode_str, category, adversarial_answer, choice_order_seed
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
