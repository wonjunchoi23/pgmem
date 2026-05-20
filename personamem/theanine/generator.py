"""
Generator for Theanine — PersonaMem variant

Phase 2 multiple-choice QA answer generation only.

QA prompt structure (Q2 option (A) — Theanine original skeleton + multiple-choice
suffix): refined timeline memory + current_dialogue + question + 4 options +
"answer with a/b/c/d".
"""

import json
import logging
from typing import Dict, List, Tuple

import config as cfg

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT
# =============================================================================

QA_PROMPT_MULTICHOICE = """\
Answer the following multiple-choice question about the user based on the memory and the current conversation.

Memory:
{memory_text}

Current conversation:
{current_dialogue}

Question:
{question}

Options:
{options_text}

Choose the single best answer (a, b, c, or d) based only on the information above. \
Output as a JSON object with an "answer" field containing only the letter."""


# =============================================================================
# JSON SCHEMA
# =============================================================================

QA_SCHEMA_MULTICHOICE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["a", "b", "c", "d"]}
    },
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
        options: List[str],
        current_dialogue: str = "",
    ) -> Tuple[str, Dict]:
        """Build the QA prompt and return the matching guided-json schema."""
        memory_text = ""
        for i, s in enumerate(retrieved_summaries):
            memory_text += f"{i+1}: {s}\n"
        memory_text = memory_text.strip() if memory_text.strip() else "(no memory retrieved)"

        dialogue_text = current_dialogue.strip() if current_dialogue else "(no recent dialogue)"
        options_text = "\n".join(options)

        prompt = QA_PROMPT_MULTICHOICE.format(
            memory_text=memory_text,
            current_dialogue=dialogue_text,
            question=question,
            options_text=options_text,
        )
        return prompt, QA_SCHEMA_MULTICHOICE

    @staticmethod
    def parse_qa_result(result) -> str:
        """Normalize a raw QA result to a single letter or 'unknown'."""
        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = str(result) if result else ""
        label = (answer or "").strip().lower()
        return label if label in ("a", "b", "c", "d") else "unknown"

    def generate_qa_answer(
        self,
        question: str,
        retrieved_summaries: List[str],
        options: List[str],
        current_dialogue: str = "",
    ) -> Tuple[str, Dict, str]:
        """Generate a multiple-choice QA answer (sequential fallback path)."""
        prompt, schema = self.build_qa_prompt(
            question=question,
            retrieved_summaries=retrieved_summaries,
            options=options,
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
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    temperature=cfg.TEMPERATURE,
                    max_tokens=cfg.MAX_TOKENS,
                    return_usage=True,
                )
                token_info = extract_token_info(result, self.model_path)
                answer = self.parse_qa_result(
                    result.get("content", "") if isinstance(result, dict) else str(result)
                )
                if self._llm_logger is not None:
                    self._llm_logger.log("call_5_qa", "", prompt, result)
            except Exception as e:
                logger.error(f"generate_qa_answer: fallback also failed ({e})")
                token_info = {"input": 0, "output": 0, "model": self.model_path}
                answer = "unknown"

        return answer, token_info, prompt
