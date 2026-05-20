"""
Prompt construction and QA generation for GraphMem v5 — PersonaMem variant.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


_NODE_TYPE_DESC = (
    "Node types:\n"
    "  State:  A user-specific condition that is currently or recently valid and may change over time. It captures the user's present stance, ongoing goal, constraint, situation, or preference shift. States are time-bounded and context-sensitive, and they may affect upcoming decisions or responses. Exclude transient emotions unless they directly modify an active task constraint or decision.\n"
    "  Trait:  A generalized user characteristic that persists across situations and time. It represents recurring dispositions, stable preferences, values, or habitual tendencies. Traits are cross-situational and relatively context-independent. Evidential support is checked separately at later stages; focus here on whether the content itself is trait-like.\n"
    "  Memory: An episodic summary of what happened during a recent conversation. It captures concrete events, topics, and actions at a particular time, not generalized persona attributes."
)

SYS_QA_MULTICHOICE = (
    "You are a helpful assistant answering a multiple-choice question about a user based on their memory.\n"
    "The information below describes what is known about the user from past conversations.\n"
    + _NODE_TYPE_DESC + "\n"
    "\n"
    "[Traits] are currently reliable; [Challenged Traits] may be outdated, replaced, or contradicted. "
    "For a challenged trait, prefer any \"shifted to\" entry and weigh listed conflicting evidence before using the trait.\n"
    "\n"
    "Current Constraints are high-impact user states that should be honored unless the question explicitly overrides them.\n"
    "\n"
    "Task:\n"
    "- Use the user information when it is relevant to the question.\n"
    "- Choose the single best option (a, b, c, or d) that fits this specific user given the retrieved memory.\n"
    "- If the retrieved memory is insufficient or irrelevant, choose the option most consistent with the question on its own.\n"
    "\n"
    "Output requirements:\n"
    "Respond in JSON with two fields, in this order:\n"
    "- \"reasoning\": one short sentence (≤ 1 sentence) stating the key factor from the retrieved memory (or its absence) that drove the choice. Write reasoning BEFORE answer.\n"
    "- \"answer\": exactly one of \"a\", \"b\", \"c\", or \"d\"."
)

QA_PROMPT_MULTICHOICE = """\
[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

[Options]
{options_text}

Choose the single best answer (a, b, c, or d) based only on the information in memory.
Output JSON with "reasoning" first (≤ 1 sentence), then "answer".\
"""

# `reasoning` is listed first in `properties` and `required` to elicit
# chain-of-thought before the final answer. The post-processor reads only
# the `answer` field, so evaluation pipelines are unaffected.
QA_SCHEMA_MULTICHOICE = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string", "enum": ["a", "b", "c", "d"]},
    },
    "required": ["reasoning", "answer"],
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

    def build_qa_prompt(self, question: str, retrieved_memory: str, options: List[str], subset: str = "multichoice") -> Tuple[str, str, Dict]:
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

    def answer_qa(self, question: str, retrieved_memory: str, options: List[str], subset: str = "multichoice") -> Tuple[str, Dict, str]:
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
