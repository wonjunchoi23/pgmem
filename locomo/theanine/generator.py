"""
Generator for Theanine — LoComo batch-enabled variant
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple

import config as cfg
from memory_graph import _extract_token_info

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

QA_PROMPT_CAT_124 = """\
Based on the context: {context}, write an answer in the form of a short phrase for the \
following question. Answer with exact words from the context whenever possible.

Current conversation:
{current_dialogue}

Question:
{question}

Short answer:"""

QA_PROMPT_CAT_3 = """\
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION \
to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where \
possible, and avoid using any subjects.

Current conversation:
{current_dialogue}

Question:
{question}

Short answer:"""

QA_PROMPT_CAT_5 = """\
Based on the context: {context}, answer the following question.

Current conversation:
{current_dialogue}

Question:
{question}

Select the correct answer: {choice_a} or {choice_b}"""


# =============================================================================
# JSON SCHEMAS
# =============================================================================

QA_SCHEMA_FREE = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


# =============================================================================
# HELPERS
# =============================================================================

def _deterministic_bool(seed_text: str) -> bool:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[-1], 16) % 2 == 0


# =============================================================================
# GENERATOR
# =============================================================================

class Generator:
    def __init__(self, llm_client, model_path: str = ""):
        self.llm_client = llm_client
        self.model_path = model_path
        self._llm_logger = None

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    def build_qa_prompt(
        self,
        question: str,
        refined_texts: List[str],
        category: int,
        current_dialogue: str = "",
        adversarial_answer: str = "",
        choice_order_seed: Optional[str] = None,
    ) -> Tuple[str, float, dict]:
        context = ""
        for i, text in enumerate(refined_texts):
            context += f"{i + 1}: {text}\n"
        context = context.strip() if context.strip() else "(no memory retrieved)"

        if category == 5:
            temperature = cfg.TEMPERATURE_C5
            other_choice = "Not mentioned in the conversation"
            seed = choice_order_seed or f"{question}::{adversarial_answer}"
            if _deterministic_bool(seed):
                choice_a, choice_b = adversarial_answer, other_choice
            else:
                choice_a, choice_b = other_choice, adversarial_answer
            prompt = QA_PROMPT_CAT_5.format(
                context=context,
                current_dialogue=current_dialogue,
                question=question,
                choice_a=choice_a,
                choice_b=choice_b,
            )
            schema = QA_SCHEMA_FREE
        elif category == 3:
            temperature = cfg.TEMPERATURE
            prompt = QA_PROMPT_CAT_3.format(
                context=context,
                current_dialogue=current_dialogue,
                question=question,
            )
            schema = QA_SCHEMA_FREE
        else:
            temperature = cfg.TEMPERATURE
            prompt = QA_PROMPT_CAT_124.format(
                context=context,
                current_dialogue=current_dialogue,
                question=question,
            )
            schema = QA_SCHEMA_FREE

        return prompt, temperature, schema

    def generate_qa_answer(
        self,
        question: str,
        refined_texts: List[str],
        category: int,
        current_dialogue: str = "",
        adversarial_answer: str = "",
        choice_order_seed: Optional[str] = None,
    ) -> Tuple[str, Dict, str]:
        prompt, temperature, schema = self.build_qa_prompt(
            question=question,
            refined_texts=refined_texts,
            category=category,
            current_dialogue=current_dialogue,
            adversarial_answer=adversarial_answer,
            choice_order_seed=choice_order_seed,
        )

        try:
            result = self.llm_client.generate(
                prompt=prompt,
                guided_json=schema,
                temperature=temperature,
                max_tokens=cfg.MAX_TOKENS,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
            token_info = _extract_token_info(result, self.model_path)
            answer = result.get("answer", "") if isinstance(result, dict) else ""
            if self._llm_logger is not None:
                self._llm_logger.log("call_4_qa", "", prompt, result)
        except json.JSONDecodeError:
            logger.warning("generate_qa_answer: JSON parse failed, retrying without guided_json")
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=cfg.MAX_TOKENS,
                return_usage=True,
            )
            token_info = _extract_token_info(result, self.model_path)
            answer = result.get("content", "") if isinstance(result, dict) else str(result)
            if self._llm_logger is not None:
                self._llm_logger.log("call_4_qa", "", prompt, result)

        return answer, token_info, prompt
