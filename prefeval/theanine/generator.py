"""
Generator for Theanine — PrefEval port.

Trimmed from exp_implexconv_no_response/theanine/generator.py:
- Removed `subset` (no opposed/supportive split).
- Removed QA_PROMPT_SUPPORTIVE / QA_SCHEMA_SUPPORTIVE.
- QA word cap: 100 → 200 words.

Otherwise unchanged: same llm_client.generate() flow, same JSON schema, same
token tracking and per-call logging.
"""

import json
import logging
from typing import Dict, List, Tuple

import config as cfg

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

QA_PROMPT = """\
Generate the most plausible answer to the question based on the current conversation. You can refer to the memory, but you should ignore the memory if it misleads the answer.

Your answer should follow the style of the conversation.

Memory:
{memory_text}

Current conversation:
{current_dialogue}

Question:
{question}

Generate the answer for {speaker}. Output as a JSON object with an "answer" field. (maximum 200 words)"""


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
# HELPER
# =============================================================================

def extract_token_info(result, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(result, dict) and "_usage" in result:
        usage = result["_usage"]
        token_info["input"]  = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


# =============================================================================
# GENERATOR
# =============================================================================

class Generator:
    """QA answer generator for Theanine (Phase 2 only)."""

    def __init__(self, llm_client, model_path: str = ""):
        self.llm_client  = llm_client
        self.model_path  = model_path
        self._llm_logger = None

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    def build_qa_prompt(
        self,
        question: str,
        retrieved_summaries: List[str],
        current_dialogue: str = "",
    ) -> Tuple[str, Dict]:
        memory_text = ""
        for i, s in enumerate(retrieved_summaries):
            memory_text += f"{i+1}: {s}\n"
        memory_text = memory_text.strip() if memory_text.strip() else "(no memory retrieved)"

        prompt = QA_PROMPT.format(
            memory_text=memory_text,
            current_dialogue=current_dialogue,
            question=question,
            speaker="Assistant",
        )
        return prompt, QA_SCHEMA

    def parse_qa_result(self, result) -> str:
        if isinstance(result, dict):
            return result.get("answer", "")
        return str(result) if result else ""

    def generate_qa_answer(
        self,
        question: str,
        retrieved_summaries: List[str],
        current_dialogue: str = "",
    ) -> Tuple[str, Dict, str]:
        prompt, schema = self.build_qa_prompt(
            question=question,
            retrieved_summaries=retrieved_summaries,
            current_dialogue=current_dialogue,
        )

        try:
            result = self.llm_client.generate(
                prompt=prompt,
                guided_json=schema,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_TOKENS,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
            token_info = extract_token_info(result, self.model_path)
            answer = self.parse_qa_result(result)
            if self._llm_logger is not None:
                self._llm_logger.log("call_5_qa", "", prompt, result)
        except json.JSONDecodeError:
            logger.warning("generate_qa_answer: JSON parse failed after all retries, retrying without guided_json")
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_TOKENS,
                return_usage=True,
            )
            token_info = extract_token_info(result, self.model_path)
            answer = self.parse_qa_result(
                result.get("content", "") if isinstance(result, dict) else str(result),
            )
            if self._llm_logger is not None:
                self._llm_logger.log("call_5_qa", "", prompt, result)

        return answer, token_info, prompt
