"""
Prompt construction and QA generation for GraphMem v5.
"""

import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


_NODE_TYPE_DESC = (
    "Retrieved memory node types:\n"
    "- State: a time-bounded user condition (situation, goal, constraint).\n"
    "- Trait: a stable user characteristic across situations.\n"
    "- Memory: an episodic summary of a past conversation.\n"
    "Sections: Current Constraints (high-impact States to honor unless overridden), "
    "Traits, Challenged Traits (possibly outdated; prefer \"shifted to\" if listed), "
    "Relevant States, Relevant Memories, Recent Conversation."
)

SYS_QA_OPPOSED = (
    "You are an assistant who has talked with this user across multiple sessions.\n"
    + _NODE_TYPE_DESC + "\n"
    "\n"
    "Use the retrieved memory to tailor your answer. If any item — even one off "
    "the question's topic — describes a constraint, situation, or trait that "
    "would change a standard answer, incorporate it. Do not give a generic "
    "answer when a relevant user circumstance is available.\n"
    "\n"
    "Answer naturally; do not say \"based on what I remember\" or similar. "
    "Answer in 100 words or less. Output JSON: {\"answer\": \"...\"}."
)

SYS_QA_SUPPORTIVE = (
    "You are an assistant answering a yes/no question about a user based on past conversations.\n"
    + _NODE_TYPE_DESC + "\n"
    "\n"
    "Answer based only on the retrieved user information. Answer \"yes\" only if "
    "the retrieved memory clearly supports yes. If the retrieved memory is "
    "insufficient, irrelevant, or ambiguous, answer \"no\".\n"
    "\n"
    "Output JSON: {\"answer\": \"yes\"} or {\"answer\": \"no\"}."
)

QA_PROMPT_OPPOSED = """\
[Question]
{question}

[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

Answer in 100 words or less. Output JSON: {{"answer": "..."}}.\
"""

QA_PROMPT_SUPPORTIVE = """\
[Question]
{question}

[Retrieved Memory]
{retrieved_memory}

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

    def build_qa_prompt(self, question: str, retrieved_memory: str, subset: str) -> Tuple[str, str, Dict]:
        if subset == "supportive":
            return (
                QA_PROMPT_SUPPORTIVE.format(
                    retrieved_memory=retrieved_memory or "No memory available.",
                    question=question,
                ),
                SYS_QA_SUPPORTIVE,
                QA_SCHEMA_SUPPORTIVE,
            )
        return (
            QA_PROMPT_OPPOSED.format(
                retrieved_memory=retrieved_memory or "No memory available.",
                question=question,
            ),
            SYS_QA_OPPOSED,
            QA_SCHEMA_OPPOSED,
        )

    def answer_qa(self, question: str, retrieved_memory: str, subset: str = "opposed") -> Tuple[str, Dict, str]:
        prompt, system_prompt, schema = self.build_qa_prompt(question, retrieved_memory, subset)
        memory_str = retrieved_memory or "No memory available."

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
                old_len = len(memory_str)
                memory_str = memory_str[: max(old_len // 2, 1)]
                prompt, system_prompt, schema = self.build_qa_prompt(question, memory_str, subset)

        return "", {"input": 0, "output": 0, "model": self._model_path}, prompt

    def parse_qa_result(self, result, subset: str) -> Tuple[str, Dict]:
        answer = result.get("answer", "") if isinstance(result, dict) else ""
        if subset == "supportive":
            label = answer.strip().lower()
            answer = label if label in {"yes", "no"} else "no"
        return answer, _extract_token_info(result, self._model_path)

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
