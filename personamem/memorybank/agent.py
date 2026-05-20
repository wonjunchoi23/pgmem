"""
MemoryBank Agent Module — PersonaMem variant

Wraps MemoryBankSystem with the experiment interface. QA prompts use a
multiple-choice format (a/b/c/d) and follow option (C) from the migration spec:
user_portrait + retrieved memory + memo_dates + question + 4 options.
The SiliconFriend persona and the [|User|]/[|AI|] dialogue framing are removed.
History (most recent N blocks of user/assistant messages) is included with
[|User|]/[|AI|] labels (option B from the migration spec).

LLM call types:
  call_2_daily_event        — daily event summary (memory build)
  call_3_daily_personality  — daily personality summary (memory build)
  call_4_global_event       — global event summary (Phase 1 → 2 transition)
  call_5_global_personality — global personality summary (Phase 1 → 2 transition)
  call_6_qa                 — multiple-choice QA answering (Phase 2)
"""

import json
import logging
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
    CALL_DIRS = [
        "call_2_daily_event",
        "call_3_daily_personality",
        "call_4_global_event",
        "call_5_global_personality",
        "call_6_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: Optional[str], user_prompt: str, output) -> None:
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
# PROMPT TEMPLATES
# =============================================================================

QA_SYSTEM_PROMPT = (
    "You are a helpful assistant answering a multiple-choice question about a "
    "user based on their conversation history stored in memory. Respond in JSON "
    "format with an 'answer' field containing only the letter a, b, c, or d."
)

QA_PROMPT_TEMPLATE = """\
{user_portrait_section}\
{memory_section}\
{history_section}\
Question: {question}

Options:
{options_text}

Choose the single best answer (a, b, c, or d) based only on the information above."""


# =============================================================================
# JSON SCHEMAS
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
# PROMPT SECTION BUILDERS
# =============================================================================

def _build_user_portrait_section(portrait: str) -> str:
    content = portrait.strip() if portrait else ""
    if not content:
        return ""
    return f"[User Portrait]\n{content}\n\n"


def _build_memory_section(memory: str, memo_dates: str) -> str:
    content = memory.strip() if memory else ""
    dates = memo_dates.strip() if memo_dates else ""
    if not content:
        return "[Memory]\n(no relevant memory retrieved)\n\n"
    return (
        f"[Memory]\n{content}\n"
        f"[Memory Dates] {dates}\n\n"
    )


def _build_history_section(history_lines: List[str]) -> str:
    if not history_lines:
        return ""
    return "[Recent History]\n" + "\n".join(history_lines) + "\n\n"


def _strip_speaker_prefix(content: str) -> str:
    """Remove a leading 'User: ' / 'Assistant: ' prefix if present."""
    for prefix in ("User: ", "Assistant: "):
        if content.startswith(prefix):
            return content[len(prefix):]
    return content


def format_history_messages(messages: list) -> List[str]:
    """Format PersonaMem messages into [|User|]/[|AI|] labelled lines (option B).

    Strips the existing 'User: '/'Assistant: ' prefix from content and
    applies MemoryBank's original labels.
    """
    lines = []
    for msg in messages:
        text = _strip_speaker_prefix(msg.content)
        if msg.role == "user":
            lines.append(f"[|User|]: {text}")
        elif msg.role == "assistant":
            lines.append(f"[|AI|]: {text}")
    return lines


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
# MEMORYBANK AGENT
# =============================================================================

class MemoryBankAgent:
    def __init__(
        self,
        llm_client,
        model_path: str = "",
        embedding_model=None,
    ):
        self.llm_client = llm_client
        self.model_path = model_path
        self.memory_system = MemoryBankSystem(
            llm_client=llm_client,
            embedding_model=(
                embedding_model
                if embedding_model is not None
                else cfg.EMBEDDING_MODEL
            ),
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

    def add_memory(self, content: str, conv_id: int, timestamp: str):
        """Store one PersonaMem message (block_idx is passed via the conv_id slot)."""
        self.memory_system.add_memory(content, conv_id, timestamp)

    def retrieve_memory(
        self,
        query: str,
        current_conv_id: int,
        k: int = None,
        update_strength: bool = True,
    ) -> RetrievalResult:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.retrieve(
            query, k, current_conv_id, update_strength
        )

    def build_qa_prompt(
        self,
        question: str,
        retrieved_memory: str,
        options: List[str],
        memo_dates: str = "",
        history_lines: Optional[List[str]] = None,
    ) -> str:
        user_portrait = self.memory_system.get_user_portrait()
        memory_str = retrieved_memory if retrieved_memory else ""
        options_text = "\n".join(options)

        return QA_PROMPT_TEMPLATE.format(
            user_portrait_section=_build_user_portrait_section(user_portrait),
            memory_section=_build_memory_section(memory_str, memo_dates),
            history_section=_build_history_section(history_lines or []),
            question=question,
            options_text=options_text,
        )

    @staticmethod
    def normalize_qa_answer(answer: str) -> str:
        label = (answer or "").strip().lower()
        return label if label in ("a", "b", "c", "d") else "unknown"

    # ------------------------------------------------------------------
    # QA answering (sequential fallback path)
    # ------------------------------------------------------------------

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str,
        options: List[str],
        memo_dates: str = "",
        history_lines: Optional[List[str]] = None,
    ) -> Tuple[str, Dict, str]:
        memory_str = retrieved_memory if retrieved_memory else ""

        for attempt in range(_MEMORY_RETRY_MAX):
            prompt = self.build_qa_prompt(
                question=question,
                retrieved_memory=memory_str,
                options=options,
                memo_dates=memo_dates,
                history_lines=history_lines,
            )
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=QA_SYSTEM_PROMPT,
                    guided_json=QA_SCHEMA_MULTICHOICE,
                    temperature=cfg.TEMPERATURE,
                    max_tokens=cfg.MAX_TOKENS,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                token_info = extract_token_info(result, self.model_path)
                answer = (
                    result.get("answer", "") if isinstance(result, dict) else ""
                )
                answer = self.normalize_qa_answer(answer)

                if self._llm_logger is not None:
                    self._llm_logger.log(
                        call_type="call_6_qa",
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
                    logger.error(f"Error answering QA: {e}")
                    return (
                        "unknown",
                        {"input": 0, "output": 0, "model": self.model_path},
                        prompt,
                    )

    # ------------------------------------------------------------------
    # Summarization triggers
    # ------------------------------------------------------------------

    def apply_forgetting(self, current_conv_id: int):
        self.memory_system.apply_forgetting(current_conv_id)

    def on_conv_boundary(self, conv_id: int, dialogue_text: str):
        self.memory_system.summarize_daily(conv_id, dialogue_text)

    def on_session_end(self):
        self.memory_system.synthesize_global()

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def clear_memory(self):
        self.memory_system.clear()
        self._llm_logger = None

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

    def get_and_reset_token_counts_by_type(self) -> Dict:
        return self.memory_system.get_and_reset_token_counts_by_type()

    def accumulate_call_tokens(
        self,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
    ):
        self.memory_system.accumulate_call_tokens(
            call_type, input_tokens, output_tokens
        )

    # ------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict:
        return self.memory_system.get_memory_stats()

    def get_and_reset_internal_stats(self) -> Dict:
        return self.memory_system.get_and_reset_internal_stats()
