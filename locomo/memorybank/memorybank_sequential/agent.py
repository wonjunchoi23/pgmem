"""
MemoryBank Agent Module — LoComo variant

MemoryBankAgent
---------------
Wraps MemoryBankSystem with the experiment interface:
  - add_memory               (store one turn; embedding only, no LLM)
  - retrieve_memory          (cosine similarity top-k)
  - on_session_end           (session event + personality summarization; 2 LLM calls)
  - on_phase1_end            (global summary synthesis; 2 LLM calls)
  - apply_forgetting         (probabilistic deletion before Phase 2)
  - answer_qa                (category-aware QA; 1 LLM call per question)
  - clear_memory / save_memory_snapshot / get_memory_count
  - get_and_reset_summary_tokens  (summarization LLM token counts)
  - set_llm_logger           (inject LLMCallLogger for per-sample prompt logging)

LLM call types:
  call_1_session_event        — session event summary (Phase 1, per session)
  call_2_session_personality  — two-speaker personality analysis (Phase 1, per session)
  call_3_global_event         — global event summary (Phase 1 → Phase 2 transition)
  call_4_global_personality   — global personality portrait (Phase 1 → Phase 2 transition)
  call_5_qa                   — QA answering (Phase 2, per question)

QA prompts follow AMEM original (locomo_experiment.md):
  - Category 1, 3, 4: short phrase, exact words from context
  - Category 2 (Temporal): use DATE of CONVERSATION for approximate date
  - Category 5 (Adversarial): binary choice; order randomised; TEMPERATURE_C5 used
"""

import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config as cfg

from memory_bank import MemoryBankSystem, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# LLM CALL LOGGER
# =============================================================================

class LLMCallLogger:
    """
    Logs all LLM calls (input + output) to call-type-specific folders as JSONL.

    Folder structure (per sample):
        {base_dir}/call_1_session_event/calls.jsonl       # Session event summary (Phase 1)
        {base_dir}/call_2_session_personality/calls.jsonl # Session personality (Phase 1)
        {base_dir}/call_3_global_event/calls.jsonl        # Global event summary (Phase 1 end)
        {base_dir}/call_4_global_personality/calls.jsonl  # Global personality (Phase 1 end)
        {base_dir}/call_5_qa/calls.jsonl                  # QA answering (Phase 2) ← required
    """

    CALL_DIRS = [
        "call_1_session_event",
        "call_2_session_personality",
        "call_3_global_event",
        "call_4_global_personality",
        "call_5_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "timestamp":     datetime.now().isoformat(),
            "call_type":     call_type,
            "system_prompt": system_prompt,
            "user_prompt":   user_prompt,
            "output":        output if not isinstance(output, dict) else {
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# QA PROMPT TEMPLATES  (AMEM original, per category)
# =============================================================================

# Category 1, 3, 4 — short phrase, exact words from context
QA_PROMPT_DEFAULT = (
    "Based on the context: {context}, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context whenever possible.\n\n"
    "Question: {question} Short answer:"
)

# Category 2 — temporal; use the conversation date
QA_PROMPT_TEMPORAL = (
    "Based on the context: {context}, answer the following question. "
    "Use DATE of CONVERSATION to answer with an approximate date.\n"
    "Please generate the shortest possible answer, using words from the conversation "
    "where possible, and avoid using any subjects.\n\n"
    "Question: {question} Short answer:"
)

# Category 5 — adversarial binary choice
QA_PROMPT_ADVERSARIAL = (
    "Based on the context: {context}, answer the following question. {question}\n\n"
    "Select the correct answer: {choice_a} or {choice_b}  Short answer:"
)


# =============================================================================
# JSON SCHEMAS
# =============================================================================

QA_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
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
# CONTEXT BUILDER
# =============================================================================

def _build_context(
    retrieved_memory: str,
    event_summary: str,
    user_portrait: str,
) -> str:
    """
    Combine retrieved memories, global event summary, and personality portrait
    into a single context string for the QA prompt.

    Matches original MemoryBank approach of including both retrieved memories
    and hierarchical summaries in the prompt context.
    """
    parts = []
    if retrieved_memory and retrieved_memory.strip():
        parts.append(retrieved_memory.strip())
    if event_summary:
        parts.append(f"[Summary of past conversations]: {event_summary}")
    if user_portrait:
        parts.append(f"[Speaker profiles]: {user_portrait}")
    return "\n\n".join(parts) if parts else ""


# =============================================================================
# MEMORYBANK AGENT
# =============================================================================

class MemoryBankAgent:
    """
    MemoryBank agent for the LoComo experiment (QA-only variant).

    Wraps MemoryBankSystem with the per-sample experiment interface.
    """

    def __init__(self, llm_client, model_path: str = ""):
        self.llm_client = llm_client
        self.model_path = model_path
        self.memory_system = MemoryBankSystem(
            llm_client=llm_client,
            embedding_model=cfg.EMBEDDING_MODEL,
            forgetting_divisor=cfg.FORGETTING_DIVISOR,
            retrieve_k=cfg.RETRIEVE_K,
            summarize_temperature=cfg.SUMMARIZE_TEMPERATURE,
            summarize_max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            json_retry=cfg.JSON_RETRY,
        )
        self._llm_logger = None

    def set_llm_logger(self, llm_logger):
        """Inject a per-sample LLMCallLogger. Also propagated to memory_system."""
        self._llm_logger = llm_logger
        self.memory_system.set_llm_logger(llm_logger)

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        dia_id: str,
        session_id: int,
        date_str: str,
    ):
        """Store a single dialogue turn (embedding only, no LLM call)."""
        self.memory_system.add_memory(
            content=content,
            dia_id=dia_id,
            session_id=session_id,
            date_str=date_str,
            is_summary=False,
        )

    def retrieve_memory(
        self,
        query: str,
        k: int = None,
        update_strength: bool = False,
    ) -> RetrievalResult:
        """Retrieve memories by cosine similarity."""
        k = k or cfg.RETRIEVE_K
        return self.memory_system.retrieve(query, k, update_strength)

    # ------------------------------------------------------------------
    # Summarization triggers
    # ------------------------------------------------------------------

    def on_session_end(
        self,
        session_id: int,
        date_str: str,
        dialogue_text: str,
        speaker_a: str,
        speaker_b: str,
    ):
        """
        Trigger session-level event + personality summarization.

        Call after all turns in a session have been stored. Makes 2 LLM calls.
        The event summary is also added to the embedding store as a searchable
        memory document (is_summary=True, dia_id=None).
        """
        self.memory_system.summarize_session(
            session_id=session_id,
            date_str=date_str,
            dialogue_text=dialogue_text,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )

    def on_phase1_end(self):
        """
        Synthesize global summaries from all session-level summaries.

        Call after all sessions in the sample have been processed.
        Makes 2 LLM calls. Global summaries are then used as additional
        context in Phase 2 QA prompts.
        """
        self.memory_system.synthesize_global()

    def apply_forgetting(self, now_date_str: str):
        """
        Apply Ebbinghaus forgetting curve using the last session's date as "now".

        Call once before Phase 2. Permanently deletes forgotten entries from
        the embedding store so they never appear in Phase 2 retrieval results.

        Args:
            now_date_str: Date string of the last session
                          (e.g. "1:56 pm on 8 May, 2023").
        """
        self.memory_system.apply_forgetting(now_date_str)

    # ------------------------------------------------------------------
    # QA answering
    # ------------------------------------------------------------------

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str,
        category: int,
        adversarial_answer: str = "",
    ) -> Tuple[str, Dict, str, int]:
        """
        Answer a QA question using category-specific prompts (AMEM original).

        Context = retrieved memories + global event summary + global personality.
        No retrieval happens inside this method; caller passes retrieved_memory.

        Args:
            question:           QA question text.
            retrieved_memory:   Formatted string from retrieve_memory().
            category:           QA category (1–5).
            adversarial_answer: Ground-truth answer for category 5 (used to
                                build the binary choice with random ordering).

        Returns:
            Tuple of (answer_str, token_info_dict, prompt_snapshot_str, api_calls_int).
        """
        event_summary = self.memory_system.get_event_summary()
        user_portrait = self.memory_system.get_user_portrait()

        context = _build_context(retrieved_memory, event_summary, user_portrait)
        memory_str = context  # may be shortened on retry

        # Build category-5 choice order once (randomised, fixed for all retries)
        if category == 5:
            not_mentioned = "Not mentioned in the conversation"
            choices = [adversarial_answer, not_mentioned]
            random.shuffle(choices)
            choice_a, choice_b = choices[0], choices[1]

        for attempt in range(_MEMORY_RETRY_MAX):
            # Select prompt template by category
            if category == 2:
                prompt = QA_PROMPT_TEMPORAL.format(
                    context=memory_str,
                    question=question,
                )
                temperature = cfg.TEMPERATURE
            elif category == 5:
                prompt = QA_PROMPT_ADVERSARIAL.format(
                    context=memory_str,
                    question=question,
                    choice_a=choice_a,
                    choice_b=choice_b,
                )
                temperature = cfg.TEMPERATURE_C5
            else:
                prompt = QA_PROMPT_DEFAULT.format(
                    context=memory_str,
                    question=question,
                )
                temperature = cfg.TEMPERATURE

            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt="",
                    guided_json=QA_SCHEMA,
                    temperature=temperature,
                    max_tokens=cfg.MAX_TOKENS,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                token_info = extract_token_info(result, self.model_path)
                answer = (
                    result.get("answer", "")
                    if isinstance(result, dict)
                    else ""
                )

                if self._llm_logger is not None:
                    self._llm_logger.log(
                        call_type="call_5_qa",
                        system_prompt="",
                        user_prompt=prompt,
                        output=result,
                    )

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
                        f"Truncating context {len(memory_str)} → {new_len} "
                        f"chars (attempt {attempt + 1}/{_MEMORY_RETRY_MAX})"
                    )
                    memory_str = memory_str[:new_len]
                else:
                    logger.error(f"Error answering QA (category {category}): {e}")
                    return (
                        "",
                        {"input": 0, "output": 0, "model": self.model_path},
                        prompt,
                        0,
                    )

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def clear_memory(self):
        self.memory_system.clear()
        # Note: _llm_logger is intentionally preserved; set_llm_logger() manages it.

    def save_memory_snapshot(self, directory: Path):
        self.memory_system.save_snapshot(directory)

    def get_memory_count(self) -> int:
        return self.memory_system.get_memory_count()

    def get_event_summary(self) -> str:
        return self.memory_system.get_event_summary()

    def get_user_portrait(self) -> str:
        return self.memory_system.get_user_portrait()

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_and_reset_summary_tokens(self) -> Dict:
        """
        Return and reset accumulated summarization LLM token counts.

        Call after on_session_end() and on_phase1_end() to capture overhead.
        Returns Dict with "input", "output", "api_calls".
        """
        return self.memory_system.get_and_reset_token_counts()
