"""
Generator for Theanine-New — ImplexConv

Based on original Theanine (NAACL 2025) src/theanine.py:
  (+ generate_qa_answer() for Phase 2, following amem_new pattern)

Adaptations for exp_implexconv:
  - llm_client.generate() instead of LangChain LLMChain
  - JSON structured output via guided_json (QA schemas)
  - QA prompts follow amem_new/agent.py pattern (opposed / supportive subsets)
  - Returns (answer, token_info, prompt_snapshot) 3-tuple for full traceability

Mapping:
  original "cost"          →  token_info dict ({"input": ..., "output": ...})
  original result string   →  structured JSON field extraction
"""

import json
import logging
from typing import Dict, List, Tuple

import config as cfg

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

QA_PROMPT_OPPOSED = """\
Generate the most plausible next response based on the current conversation. You can refer to the memory, but you should ignore the memory if it misleads the next response. Do not put too much information in the next response.

Your response should follow the style of the conversation.

Memory:
{memory_text}

Current conversation:
{current_dialogue}

Generate the next response for {speaker}. Output as a JSON object with an "answer" field. (maximum 100 words)"""

QA_PROMPT_SUPPORTIVE = """\
Generate the most plausible next response based on the current conversation. You can refer to the memory, but you should ignore the memory if it misleads the next response.

Memory:
{memory_text}

Current conversation:
{current_dialogue}

Answer the yes/no question for {speaker}. You MUST answer with exactly one of: yes or no. Output as a JSON object with an "answer" field."""


# =============================================================================
# JSON SCHEMAS
# =============================================================================

QA_SCHEMA_OPPOSED = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

QA_SCHEMA_SUPPORTIVE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]}
    },
    "required": ["answer"],
    "additionalProperties": False,
}


# =============================================================================
# HELPER
# =============================================================================

def extract_token_info(result, model_path: str = "") -> Dict:
    """Extract token usage from llm_client.generate() result."""
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
    """
    QA answer generator for Theanine.

    generate_qa_answer(): Phase 2 QA answering using refined timeline memories.
      New method following amem_new/agent.py answer_qa() pattern.

    Returns (text, token_info, prompt_snapshot) for full traceability.
    """

    def __init__(self, llm_client, model_path: str = ""):
        self.llm_client  = llm_client
        self.model_path  = model_path
        self._llm_logger = None  # set via set_llm_logger()

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    def generate_qa_answer(
        self,
        question: str,
        retrieved_summaries: List[str],
        subset: str,
        current_dialogue: str = "",
    ) -> Tuple[str, Dict, str]:
        """
        Generate QA answer using retrieved memory node summaries.
        Phase 2 only — no timeline retrieval or refinement.

        Uses the same prompt structure as generate_response():
          Memory (numbered summaries) + Current conversation (full session
          dialogue up to and including the question) + speaker.
        The question is appended to current_dialogue as the last User turn
        before calling the LLM, treating QA as a continuation of the session.

        Args:
            question:            QA question string.
            retrieved_summaries: List of memory node summary strings
                                 retrieved by cosine similarity to the question.
            subset:              "opposed" (free-form) or "supportive" (yes/no).
            current_dialogue:    Full session dialogue (Phase 1 GT turns) with
                                 the question already appended as the last User turn.

        Returns:
            (answer_text, token_info, prompt_snapshot)
            answer_text:    Generated answer string.
                            For supportive: one of {"yes", "no"}.
            token_info:     {"input": ..., "output": ..., "model": ...}
            prompt_snapshot: Full prompt string sent to LLM.
        """
        # Build numbered memory text — same format as generate_response()
        memory_text = ""
        for i, s in enumerate(retrieved_summaries):
            memory_text += f"{i+1}: {s}\n"
        memory_text = memory_text.strip() if memory_text.strip() else "(no memory retrieved)"

        if subset == "opposed":
            prompt = QA_PROMPT_OPPOSED.format(
                memory_text=memory_text,
                current_dialogue=current_dialogue,
                speaker="Assistant",
            )
            schema = QA_SCHEMA_OPPOSED
        else:
            prompt = QA_PROMPT_SUPPORTIVE.format(
                memory_text=memory_text,
                current_dialogue=current_dialogue,
                speaker="Assistant",
            )
            schema = QA_SCHEMA_SUPPORTIVE

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
            answer = result.get("answer", "") if isinstance(result, dict) else ""
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
            answer = result.get("content", "") if isinstance(result, dict) else str(result)
            if self._llm_logger is not None:
                self._llm_logger.log("call_5_qa", "", prompt, result)

        if subset == "supportive":
            # Normalise to the allowed set in case guided_json is not enforced
            label = answer.strip().lower()
            if label not in ("yes", "no"):
                label = "unknown"
            answer = label

        return answer, token_info, prompt
