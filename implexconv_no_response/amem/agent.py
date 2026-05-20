"""
A-MEM Agent Module — Batch-enabled variant

Changes from amem/agent.py:
- BaseAgent.__init__() accepts an optional embedding_model parameter
  (pre-built SentenceTransformer instance) to share weights across agents.
- accumulate_memory_tokens() updated with call_type parameter for per-type
  token attribution from batch LLM calls.
- get_and_reset_memory_tokens() returns per-call-type breakdown.
- get_memory_stats() and get_and_reset_internal_stats() added (delegating to
  AgenticMemorySystem).
- Response generation (RESPONSE_PROMPT, build_response_prompt,
  generate_response) removed — Phase 1 stores memory only, no LLM response.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config as cfg

from memory_layer import AgenticMemorySystem

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

QA_PROMPT_OPPOSED = """\
Based on the conversation memory:
{retrieved_memory}

Question from user: {question}

Answer the question based only on the information provided in the memory above.
Be concise (maximum 100 words)."""

QA_PROMPT_SUPPORTIVE = """\
Based on the conversation memory:
{retrieved_memory}

Question from user: {question}

Answer the yes/no question based only on the information in the memory above.
You MUST answer with exactly one of: yes or no."""

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
# PROMPT-TOO-LONG RETRY HELPER
# =============================================================================

_PROMPT_TOO_LONG_RE = re.compile(
    r"decoder prompt \(length (\d+)\).*?maximum model length of (\d+)",
    re.IGNORECASE | re.DOTALL,
)
_CHARS_PER_TOKEN = 4
_MEMORY_RETRY_MAX = 3


def _parse_prompt_too_long(error: Exception):
    m = _PROMPT_TOO_LONG_RE.search(str(error))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# =============================================================================
# TOKEN HELPER
# =============================================================================

def extract_token_info(response, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(response, dict) and "_usage" in response:
        usage = response["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent:
    """
    A-MEM agent wrapping AgenticMemorySystem.

    Batch-enabled changes:
    - Accepts optional embedding_model (shared SentenceTransformer instance).
    - accumulate_memory_tokens() lets the batch runner push per-item token
      counts into this agent's internal LLMWrapper counters.
    """

    def __init__(self, llm_client, model_path: str = "", embedding_model=None):
        self.llm_client = llm_client
        self.model_path = model_path
        self.memory_system = AgenticMemorySystem(
            llm_client=llm_client,
            model_name=cfg.EMBEDDING_MODEL,
            embedding_model=embedding_model,
            evo_threshold=cfg.EVOLUTION_THRESHOLD,
            json_retry=cfg.JSON_RETRY,
            conv_ids_per_day=cfg.CONV_IDS_PER_DAY,
            minutes_per_turn=cfg.MINUTES_PER_TURN,
        )
        self._llm_logger = None

    # ------------------------------------------------------------------
    # LLM call logger
    # ------------------------------------------------------------------

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger
        self.memory_system.llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def add_memory(self, content: str, time: Optional[str] = None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, query: str, k: int = None) -> str:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.find_related_memories_raw(query, k=k)

    def retrieve_memory_with_metadata(self, query: str, k: int = None) -> Tuple[str, List[Dict]]:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.find_related_memories_with_metadata(query, k=k)

    def retrieve_for_log(self, query: str, k: int = None) -> Tuple[List[Dict], int]:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.retrieve_for_log(query, k=k)

    # ------------------------------------------------------------------
    # QA answering
    # ------------------------------------------------------------------

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str = None,
        subset: str = "opposed",
    ) -> Tuple[str, Dict, str, int]:
        if retrieved_memory is None:
            retrieved_memory = self.retrieve_memory(question)

        memory_str = retrieved_memory if retrieved_memory else "No memory available."
        schema = QA_SCHEMA_SUPPORTIVE if subset == "supportive" else QA_SCHEMA_OPPOSED
        prompt_template = QA_PROMPT_SUPPORTIVE if subset == "supportive" else QA_PROMPT_OPPOSED

        for attempt in range(_MEMORY_RETRY_MAX):
            prompt = prompt_template.format(
                retrieved_memory=memory_str,
                question=question,
            )
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=(
                        "You are a helpful assistant answering a question about a user "
                        "based on their conversation history stored in memory. "
                        "Respond in JSON format with an 'answer' field."
                    ),
                    guided_json=schema,
                    temperature=cfg.TEMPERATURE,
                    max_tokens=cfg.MAX_TOKENS,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                if self._llm_logger is not None:
                    self._llm_logger.log("call_4_qa", "", prompt, result)
                token_info = extract_token_info(result, self.model_path)
                answer = result.get("answer", "") if isinstance(result, dict) else ""

                if subset == "supportive":
                    label = answer.strip().lower()
                    if label not in ("yes", "no"):
                        label = "unknown"
                    answer = label

                return answer, token_info, prompt, 1

            except Exception as e:
                parsed = _parse_prompt_too_long(e)
                if parsed and attempt < _MEMORY_RETRY_MAX - 1:
                    prompt_len, max_len = parsed
                    cut_chars = (prompt_len - max_len + 200) * _CHARS_PER_TOKEN
                    new_len = max(len(memory_str) - cut_chars, 200)
                    logger.warning(
                        f"[answer_qa] Prompt too long "
                        f"({prompt_len} > {max_len} tokens). "
                        f"Truncating memory {len(memory_str)} -> {new_len} chars "
                        f"(attempt {attempt + 1}/{_MEMORY_RETRY_MAX})"
                    )
                    memory_str = memory_str[:new_len]
                else:
                    logger.error(f"Error answering QA: {e}")
                    return "", {"input": 0, "output": 0, "model": self.model_path}, prompt, 0

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_and_reset_memory_tokens(self) -> Dict:
        """Return per-call-type token counts and reset."""
        return self.memory_system.get_and_reset_token_counts_by_type()

    def accumulate_memory_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_calls: int = 1,
        call_type: str = "call_2_note_construction",
    ):
        """Accumulate token counts from an external batch LLM call."""
        self.memory_system.accumulate_token_counts(input_tokens, output_tokens, llm_calls, call_type)

    # ------------------------------------------------------------------
    # Memory stats
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict:
        """Return current memory count and estimated total content tokens."""
        return self.memory_system.get_memory_stats()

    def get_and_reset_internal_stats(self) -> Dict:
        """Return evolution and parser-fallback stats, then reset counters."""
        return self.memory_system.get_and_reset_internal_stats()

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def clear_memory(self):
        self.memory_system.clear()

    def save_memory_snapshot(self, directory: Path):
        self.memory_system.save_snapshot(directory)

    def get_memory_count(self) -> int:
        return self.memory_system.get_memory_count()
