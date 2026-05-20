"""
MemoryBank Agent Module — LoComo batch variant

MemoryBankAgent
---------------
Wraps MemoryBankSystem with the experiment interface.

LLM call types (LoComo numbering):
  call_1_daily_event        — session event summary (Phase 1, per session)
  call_2_daily_personality  — two-speaker personality analysis (Phase 1, per session)
  call_3_global_event       — global event summary (Phase 1 end)
  call_4_global_personality — global personality portrait (Phase 1 end)
  call_5_qa                 — QA answering (Phase 2, per question)
"""

import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config as cfg

from memory_bank import MemoryBankSystem, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# LLM CALL LOGGER
# =============================================================================

class LLMCallLogger:
    """Logs all LLM calls (input + output) to call-type-specific folders as JSONL."""

    CALL_DIRS = [
        "call_1_daily_event",
        "call_2_daily_personality",
        "call_3_global_event",
        "call_4_global_personality",
        "call_5_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: Optional[str], user_prompt: str, output) -> None:
        entry = {
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
# QA PROMPT TEMPLATES  (SiliconFriend style, adapted for LoComo)
# =============================================================================

QA_SYSTEM_PROMPT = """\
Now you will play the role of an AI Companion named SiliconFriend. \
You should be able to: (1) provide warm companionship to chat users; \
(2) understand past [memory], and if they are relevant to the current question, \
you must extract information from the [memory] to answer the question; \
(3) you are also an excellent psychological counselor, and when users confide \
in you about their difficulties and seek help, you can provide them with warm \
and helpful responses."""

QA_PROMPT_PREFIX = """\
{user_portrait_section}\
{memory_section}\
Below is a multi-round conversation between two speakers. \
You should refer to the context of the conversation, past [memory], and provide \
detailed answers to the question. Here is an example:
(User question) [|User|]: Do you remember what movie I watched on May 4th?
2. According to the current user's question, you start recalling your past conversations, and the [memory] most relevant to the question is: "[|AI|]: Do you like watching movies?
[|User|]: I like watching movies, I went to see "Rise of the Planet of the Apes" today, it's really good."
The date of this [memory] in the memory is May 4th
"3. (Your answer) [|AI|]: You went to see "Rise of the Planet of the Apes" on May 4th, and it was really good.
Please understand and use [memory] according to the example. The human's questions start with [|User|]:, and your answers start with [|AI|]:.
"""

# Category 1, 2, 4 — short phrase answer
QA_PROMPT_DEFAULT_SUFFIX = """\
Write an answer in the form of a short phrase. Answer with exact words from the context whenever possible.
Please start the conversation in the following format:
{history_text}"""

# Category 3 — temporal; use the date of the conversation
QA_PROMPT_TEMPORAL_SUFFIX = """\
Use DATE of CONVERSATION to answer with an approximate date. \
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.
Please start the conversation in the following format:
{history_text}"""

# Category 5 — adversarial binary choice
QA_PROMPT_CAT5_SUFFIX = """\
Select the correct answer: {choice_a} or {choice_b}
Please start the conversation in the following format:
{history_text}"""


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
# PROMPT SECTION BUILDERS
# =============================================================================

def _build_user_portrait_section(portrait: str) -> str:
    content = portrait.strip() if portrait else ""
    return (
        f"The personality of the user and the AI Companion's response strategy "
        f"are: {content}\n\n"
    )


def _build_memory_section(memory: str, memo_dates: str) -> str:
    content = memory.strip() if memory else ""
    dates = memo_dates.strip() if memo_dates else ""
    return (
        f'Based on the current user\'s question, you start recalling past '
        f'conversations, and the [memory] most relevant to the question is: '
        f'"{content}\n'
        f'The date of this [memory] in the memory is {dates}."\n\n'
    )


def _build_history_text(history: list, question: str) -> str:
    """Format recent turns plus the question.

    history: list of Turn objects (actual speaker names, per LoComo choice B).
    The question is appended as [|User|]: since it is posed to the AI system.
    """
    lines = []
    for turn in history:
        lines.append(f"[{turn.speaker}]: {turn.text}")
    lines.append(f"[|User|]: {question}")
    lines.append("[|AI|]: ")
    return "\n".join(lines)


def _deterministic_shuffle(choices: list, seed_str: str) -> list:
    """Shuffle choices deterministically based on a string seed."""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    result = list(choices)
    rng.shuffle(result)
    return result


# =============================================================================
# MEMORYBANK AGENT
# =============================================================================

class MemoryBankAgent:
    """MemoryBank agent for the LoComo experiment."""

    def __init__(self, llm_client, model_path: str = "", embedding_model=None):
        self.llm_client = llm_client
        self.model_path = model_path
        self.memory_system = MemoryBankSystem(
            llm_client=llm_client,
            embedding_model=embedding_model if embedding_model is not None else cfg.EMBEDDING_MODEL,
            forgetting_divisor=cfg.FORGETTING_DIVISOR,
            retrieve_k=cfg.RETRIEVE_K,
            summarize_temperature=cfg.SUMMARIZE_TEMPERATURE,
            summarize_max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            json_retry=cfg.JSON_RETRY,
        )
        self._llm_logger = None

    def set_llm_logger(self, llm_logger):
        self._llm_logger = llm_logger
        self.memory_system.set_llm_logger(llm_logger)

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        session_id: int,
        date_str: str,
        timestamp: str,
    ):
        """Store a single dialogue turn (embedding only, no LLM call)."""
        self.memory_system.add_memory(
            content=content,
            session_id=session_id,
            date_str=date_str,
            timestamp=timestamp,
        )

    def retrieve_memory(
        self,
        query: str,
        k: int = None,
        update_strength: bool = False,
    ) -> RetrievalResult:
        """Retrieve memories from both dialogue and summary stores."""
        k = k or cfg.RETRIEVE_K
        return self.memory_system.retrieve(query, k, update_strength)

    # ------------------------------------------------------------------
    # Summarization triggers (sequential fallback)
    # ------------------------------------------------------------------

    def on_session_end(
        self,
        session_id: int,
        date_str: str,
        dialogue_text: str,
        speaker_a: str,
        speaker_b: str,
    ):
        """Trigger daily event + personality summarization (sequential fallback)."""
        self.memory_system.summarize_daily(
            session_id=session_id,
            date_str=date_str,
            dialogue_text=dialogue_text,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )

    def on_phase1_end(self):
        """Synthesize global summaries from all session-level summaries (sequential fallback)."""
        self.memory_system.synthesize_global()

    def apply_forgetting(self, now_date_str: str):
        """Apply Ebbinghaus forgetting curve. Call once before Phase 2."""
        self.memory_system.apply_forgetting(now_date_str)

    # ------------------------------------------------------------------
    # QA prompt builder — batch path (no LLM call)
    # ------------------------------------------------------------------

    def build_qa_prompt(
        self,
        question: str,
        retrieved_memory: str,
        memo_dates: str = "",
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
        history: list = None,
    ) -> Tuple[str, float]:
        """Build a QA prompt without calling the LLM.

        Returns:
            (prompt_str, temperature_float)
        """
        portrait = self.memory_system.get_user_portrait()
        history_text = _build_history_text(history or [], question)
        portrait_section = _build_user_portrait_section(portrait)
        memory_section = _build_memory_section(retrieved_memory or "", memo_dates)

        if category == 3:
            suffix = QA_PROMPT_TEMPORAL_SUFFIX.format(history_text=history_text)
            temperature = cfg.TEMPERATURE
        elif category == 5:
            not_mentioned = "Not mentioned in the conversation"
            choices = [adversarial_answer, not_mentioned]
            if choice_order_seed:
                choices = _deterministic_shuffle(choices, choice_order_seed)
            else:
                random.shuffle(choices)
            choice_a, choice_b = choices[0], choices[1]
            suffix = QA_PROMPT_CAT5_SUFFIX.format(
                choice_a=choice_a,
                choice_b=choice_b,
                history_text=history_text,
            )
            temperature = cfg.TEMPERATURE_C5
        else:
            suffix = QA_PROMPT_DEFAULT_SUFFIX.format(history_text=history_text)
            temperature = cfg.TEMPERATURE

        prompt = (
            QA_PROMPT_PREFIX.format(
                user_portrait_section=portrait_section,
                memory_section=memory_section,
            )
            + suffix
        )
        return prompt, temperature

    # ------------------------------------------------------------------
    # QA answering — sequential fallback
    # ------------------------------------------------------------------

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str,
        memo_dates: str = "",
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
        history: list = None,
    ) -> Tuple[str, Dict, str]:
        """Answer a QA question using category-specific prompts.

        Returns:
            Tuple of (answer_str, token_info_dict, prompt_snapshot_str).
        """
        memory_str = retrieved_memory or ""

        # Build category-5 choice order once (fixed across retries)
        cat5_choice_a = cat5_choice_b = ""
        if category == 5:
            not_mentioned = "Not mentioned in the conversation"
            choices = [adversarial_answer, not_mentioned]
            if choice_order_seed:
                choices = _deterministic_shuffle(choices, choice_order_seed)
            else:
                random.shuffle(choices)
            cat5_choice_a, cat5_choice_b = choices[0], choices[1]

        for attempt in range(_MEMORY_RETRY_MAX):
            portrait = self.memory_system.get_user_portrait()
            history_text = _build_history_text(history or [], question)
            portrait_section = _build_user_portrait_section(portrait)
            memory_section = _build_memory_section(memory_str, memo_dates)

            if category == 3:
                suffix = QA_PROMPT_TEMPORAL_SUFFIX.format(history_text=history_text)
                temperature = cfg.TEMPERATURE
            elif category == 5:
                suffix = QA_PROMPT_CAT5_SUFFIX.format(
                    choice_a=cat5_choice_a,
                    choice_b=cat5_choice_b,
                    history_text=history_text,
                )
                temperature = cfg.TEMPERATURE_C5
            else:
                suffix = QA_PROMPT_DEFAULT_SUFFIX.format(history_text=history_text)
                temperature = cfg.TEMPERATURE

            prompt = (
                QA_PROMPT_PREFIX.format(
                    user_portrait_section=portrait_section,
                    memory_section=memory_section,
                )
                + suffix
            )

            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=QA_SYSTEM_PROMPT,
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
                        system_prompt=QA_SYSTEM_PROMPT,
                        user_prompt=prompt,
                        output=result,
                    )

                return answer, token_info, prompt

            except Exception as e:
                parsed = _parse_prompt_too_long(e)
                if parsed and attempt < _MEMORY_RETRY_MAX - 1:
                    prompt_len, max_len = parsed
                    cut_chars = (prompt_len - max_len + 200) * _CHARS_PER_TOKEN
                    new_len = max(len(memory_str) - cut_chars, 200)
                    logger.warning(
                        f"[answer_qa] Prompt too long "
                        f"({prompt_len} > {max_len} tokens). "
                        f"Truncating memory {len(memory_str)} -> {new_len} "
                        f"chars (attempt {attempt + 1}/{_MEMORY_RETRY_MAX})"
                    )
                    memory_str = memory_str[:new_len]
                else:
                    logger.error(f"Error answering QA (category {category}): {e}")
                    return (
                        "",
                        {"input": 0, "output": 0, "model": self.model_path},
                        prompt,
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
        return self.memory_system.get_and_reset_token_counts_by_type()

    def accumulate_summary_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_calls: int,
        call_type: str,
    ):
        self.memory_system.accumulate_token_counts(input_tokens, output_tokens, llm_calls, call_type)

    def get_memory_stats(self) -> Dict:
        return self.memory_system.get_memory_stats()

    def get_and_reset_internal_stats(self) -> Dict:
        return self.memory_system.get_and_reset_internal_stats()
