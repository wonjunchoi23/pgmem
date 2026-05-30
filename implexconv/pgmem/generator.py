"""Prompt construction and QA generation for PGMem."""

import logging
import re
from typing import Dict, Tuple

from updater import _NODE_TYPE_DESC

logger = logging.getLogger(__name__)

SYS_QA_OPPOSED = (
    "You are an assistant providing personalized help based on prior conversations with this user.\n"
    "\n"
    + _NODE_TYPE_DESC + "\n"
    "\n"
    "Treat retrieved episode as candidate evidence for personalization, not as something to force into every answer. "
    "Use a retrieved episode item when it directly answers the question or materially changes the user's ability, safety, cost, time, access, motivation, or appropriateness for the requested task. "
    "When such a factor applies, reflect it concretely: acknowledge the user's goal, name the relevant factor, and adjust the recommendation accordingly. "
    "Otherwise, answer the question normally without forcing personalization.\n"
    "\n"
    "If the current user message or recent conversation clearly conflicts with, updates, or overrides a retrieved episode item, the current message takes precedence.\n"
    "\n"
    "Answer naturally and personalize only when the retrieved episode meaningfully supports it. "
    "Do not say \"based on what I remember\" or similar."
)

SYS_QA_SUPPORTIVE = (
    "You are an assistant answering a yes/no question about a user based on past conversations.\n"
    + _NODE_TYPE_DESC + "\n"
    "\n"
    "Answer based only on the retrieved user information. Answer \"yes\" only if "
    "the retrieved episode clearly supports yes. If the retrieved episode is "
    "insufficient, irrelevant, or ambiguous, answer \"no\".\n"
    "\n"
    "Output JSON: {\"answer\": \"yes\"} or {\"answer\": \"no\"}."
)

QA_PROMPT_OPPOSED = """\
[Question]
{question}

[Retrieved Episode]
{retrieved_episode}

[Question]
{question}

Answer in 100 words or less. Make the answer appropriately personalized using the retrieved episode. Output JSON: {{"answer": "..."}}\
"""

QA_PROMPT_SUPPORTIVE = """\
[Question]
{question}

[Retrieved Episode]
{retrieved_episode}

[Question]
{question}

Answer with exactly one of: yes or no. Output JSON: {{"answer": "yes"}} or {{"answer": "no"}}.\
"""

QA_SCHEMA_OPPOSED = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

QA_SCHEMA_SUPPORTIVE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

_PROMPT_TOO_LONG_RE = re.compile(
    r"decoder prompt \(length \d+\).*maximum model length",
    re.IGNORECASE | re.DOTALL,
)
_RETRY_MAX = 3


class GraphGenerator:
    """Builds response prompts and runs QA generation."""

    def __init__(self, llm_client, model_path: str, config, llm_logger=None) -> None:
        self._llm = llm_client
        self._model_path = model_path
        self._cfg = config
        self._llm_logger = llm_logger

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    def build_qa_prompt(self, question: str, retrieved_episode: str, subset: str) -> Tuple[str, str, Dict]:
        if subset == "supportive":
            return (
                QA_PROMPT_SUPPORTIVE.format(
                    retrieved_episode=retrieved_episode or "No episode available.",
                    question=question,
                ),
                SYS_QA_SUPPORTIVE,
                QA_SCHEMA_SUPPORTIVE,
            )
        return (
            QA_PROMPT_OPPOSED.format(
                retrieved_episode=retrieved_episode or "No episode available.",
                question=question,
            ),
            SYS_QA_OPPOSED,
            QA_SCHEMA_OPPOSED,
        )

    def answer_qa(self, question: str, retrieved_episode: str, subset: str = "opposed") -> Tuple[str, Dict, str]:
        prompt, system_prompt, schema = self.build_qa_prompt(question, retrieved_episode, subset)
        episode_str = retrieved_episode or "No episode available."

        for _ in range(_RETRY_MAX):
            try:
                result = self._llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    guided_json=schema,
                    temperature=self._cfg.TEMPERATURE,
                    max_tokens=self._cfg.MAX_TOKENS,
                    json_retry=self._cfg.JSON_RETRY,
                    return_usage=True,
                )
                if self._llm_logger is not None:
                    self._llm_logger.log("call_6_qa", system_prompt, prompt, result)
                answer = result.get("answer", "") if isinstance(result, dict) else ""
                if subset == "supportive":
                    label = answer.strip().lower()
                    answer = label if label in {"yes", "no"} else "no"
                return answer, _extract_token_info(result, self._model_path), prompt
            except Exception as exc:
                if not _is_prompt_too_long(exc):
                    logger.error(f"QA generation failed: {exc}")
                    break
                old_len = len(episode_str)
                episode_str = episode_str[: max(old_len // 2, 1)]
                prompt, system_prompt, schema = self.build_qa_prompt(question, episode_str, subset)

        return "", {"input": 0, "output": 0, "model": self._model_path}, prompt


def _extract_token_info(result, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(result, dict) and "_usage" in result:
        usage = result["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def _is_prompt_too_long(error: Exception) -> bool:
    return bool(_PROMPT_TOO_LONG_RE.search(str(error)))
