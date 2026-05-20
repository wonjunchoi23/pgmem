"""Prompt construction and QA generation for GraphMem — PersonaMem variant."""

import logging
import re
from typing import Dict, List, Tuple

from updater import _NODE_TYPE_DESC

logger = logging.getLogger(__name__)

SYS_QA_MULTICHOICE = (
    "You are a helpful assistant answering a multiple-choice question about a user based on their memory.\n"
    "The information below describes what is known about the user from past conversations.\n"
    "\n"
    + _NODE_TYPE_DESC + "\n"
    "\n"
    "[Traits] are currently reliable; [Challenged Traits] may be outdated, replaced, or contradicted.\n "
    "\n"
    "Current Constraints are high-impact user states that should be honored unless the question explicitly overrides them.\n"
    "\n"
    "Task:\n"
    "- Use the user information when it is relevant to the question.\n"
    "- Choose the single best option (a, b, c, or d) that fits this specific user given the retrieved memory.\n"
    "- If the retrieved memory is insufficient or irrelevant, choose the option most consistent with the question on its own.\n"
    "\n"
    "Output JSON with the single field \"answer\": exactly one of \"a\", \"b\", \"c\", or \"d\"."
)

QA_PROMPT_MULTICHOICE = """\
[Question]
{question}

[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

[Options]
{options_text}

Choose the single best answer (a, b, c, or d) based only on the information in memory.
Output JSON: {{"answer": "..."}}\
"""

QA_SCHEMA_MULTICHOICE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["a", "b", "c", "d"]},
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

    def build_qa_prompt(
        self,
        question: str,
        retrieved_memory: str,
        options: List[str],
        subset: str = "multichoice",
    ) -> Tuple[str, str, Dict]:
        options_text = "\n".join(options) if options else ""
        return (
            QA_PROMPT_MULTICHOICE.format(
                retrieved_memory=retrieved_memory or "No memory available.",
                question=question,
                options_text=options_text,
            ),
            SYS_QA_MULTICHOICE,
            QA_SCHEMA_MULTICHOICE,
        )

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str,
        options: List[str],
        subset: str = "multichoice",
    ) -> Tuple[str, Dict, str]:
        prompt, system_prompt, schema = self.build_qa_prompt(question, retrieved_memory, options, subset)
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
                label = answer.strip().lower()
                answer = label if label in {"a", "b", "c", "d"} else "unknown"
                return answer, _extract_token_info(result, self._model_path), prompt
            except Exception as exc:
                if not _is_prompt_too_long(exc):
                    logger.error(f"QA generation failed: {exc}")
                    break
                old_len = len(memory_str)
                memory_str = memory_str[: max(old_len // 2, 1)]
                prompt, system_prompt, schema = self.build_qa_prompt(question, memory_str, options, subset)

        return "unknown", {"input": 0, "output": 0, "model": self._model_path}, prompt

    def parse_qa_result(self, result, subset: str = "multichoice") -> Tuple[str, Dict]:
        answer = result.get("answer", "") if isinstance(result, dict) else ""
        label = answer.strip().lower()
        answer = label if label in {"a", "b", "c", "d"} else "unknown"
        return answer, _extract_token_info(result, self._model_path)


def _extract_token_info(result, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(result, dict) and "_usage" in result:
        usage = result["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def _is_prompt_too_long(error: Exception) -> bool:
    return bool(_PROMPT_TOO_LONG_RE.search(str(error)))
