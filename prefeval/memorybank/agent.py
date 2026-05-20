"""
MemoryBank Agent Module — PrefEval port.

Trimmed from exp_implexconv_no_response/memorybank/agent.py:
- Removed `subset` (no opposed/supportive split).
- Removed QA_PROMPT_SUPPORTIVE_SUFFIX / QA_SCHEMA_SUPPORTIVE / yes-no normalization.
- QA word cap: 100 → 200 words (in QA_PROMPT_SUFFIX).
- Added load_memory_snapshot() helper.

Otherwise unchanged: same memory_bank interface, hierarchical summarization,
forgetting curve, token tracking.
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
Below is a multi-round conversation between you (SiliconFriend) and the user. \
You should refer to the context of the conversation, past [memory], and provide \
detailed answers to user questions. Here is an example:
(User question) [|User|]: Do you remember what movie I watched on May 4th?
2. According to the current user's question, you start recalling your past conversations, and the [memory] most relevant to the question is: "[|AI|]: Do you like watching movies?
[|User|]: I like watching movies, I went to see "Rise of the Planet of the Apes" today, it's really good."
The date of this [memory] in the memory is May 4th
"3. (Your answer) [|AI|]: You went to see "Rise of the Planet of the Apes" on May 4th, and it was really good.
Please understand and use [memory] according to the example. The human's questions start with [|User|]:, and your answers start with [|AI|]:.
"""

QA_PROMPT_SUFFIX = """\
Please answer the user's final question concisely in English (maximum 200 words).
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
    lines = []
    for user_turn, assistant_turn in history:
        if user_turn:
            lines.append(f"[|User|]: {user_turn.utterance}")
        if assistant_turn:
            lines.append(f"[|AI|]: {assistant_turn.utterance}")
    lines.append(f"[|User|]: {question}")
    lines.append("[|AI|]: ")
    return "\n".join(lines)


# =============================================================================
# PROMPT-TOO-LONG RETRY HELPER
# =============================================================================

_PROMPT_TOO_LONG_RE = re.compile(
    r"decoder prompt \(length (\d+)\).*?maximum model length of (\d+)",
    re.IGNORECASE | re.DOTALL,
)
_CHARS_PER_TOKEN  = 4
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
        token_info["input"]  = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


# =============================================================================
# MEMORYBANK AGENT
# =============================================================================

class MemoryBankAgent:
    """MemoryBank agent (single-chain variant)."""

    def __init__(self, llm_client, model_path: str = "", embedding_model=None):
        self.llm_client = llm_client
        self.model_path = model_path
        self.memory_system = MemoryBankSystem(
            llm_client=llm_client,
            embedding_model=(
                embedding_model if embedding_model is not None else cfg.EMBEDDING_MODEL
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
        self.memory_system.add_memory(content, conv_id, timestamp)

    def retrieve_memory(
        self,
        query: str,
        current_conv_id: int,
        k: int = None,
        update_strength: bool = True,
    ) -> RetrievalResult:
        k = k or cfg.RETRIEVE_K
        return self.memory_system.retrieve(query, k, current_conv_id, update_strength)

    def build_qa_prompt(
        self,
        question: str,
        retrieved_memory: str,
        memo_dates: str = "",
        history: list = None,
    ) -> str:
        user_portrait = self.memory_system.get_user_portrait()
        memory_str = retrieved_memory if retrieved_memory else ""
        history_text = _build_history_text(history or [], question)

        return (
            QA_PROMPT_PREFIX.format(
                user_portrait_section=_build_user_portrait_section(user_portrait),
                memory_section=_build_memory_section(memory_str, memo_dates=memo_dates),
            )
            + QA_PROMPT_SUFFIX.format(history_text=history_text)
        )

    # ------------------------------------------------------------------
    # QA answering (sequential path; runner uses batched path directly)
    # ------------------------------------------------------------------

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str,
        memo_dates: str = "",
        history: list = None,
    ) -> Tuple[str, Dict, str]:
        memory_str = retrieved_memory if retrieved_memory else ""

        for attempt in range(_MEMORY_RETRY_MAX):
            prompt = self.build_qa_prompt(
                question=question,
                retrieved_memory=memory_str,
                memo_dates=memo_dates,
                history=history,
            )
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=QA_SYSTEM_PROMPT,
                    guided_json=QA_SCHEMA,
                    temperature=cfg.TEMPERATURE,
                    max_tokens=cfg.MAX_TOKENS,
                    json_retry=cfg.JSON_RETRY,
                    return_usage=True,
                )
                token_info = extract_token_info(result, self.model_path)
                answer = result.get("answer", "") if isinstance(result, dict) else ""

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
                        f"[answer_qa] Prompt too long ({prompt_len} > {max_len} tokens). "
                        f"Truncating memory {len(memory_str)} -> {new_len} chars "
                        f"(attempt {attempt + 1}/{_MEMORY_RETRY_MAX})"
                    )
                    memory_str = memory_str[:new_len]
                else:
                    logger.error(f"Error answering QA: {e}")
                    return "", {"input": 0, "output": 0, "model": self.model_path}, prompt

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

    def load_memory_snapshot(self, directory: Path):
        self.memory_system.load_snapshot(directory)

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

    def accumulate_call_tokens(self, call_type: str, input_tokens: int, output_tokens: int):
        self.memory_system.accumulate_call_tokens(call_type, input_tokens, output_tokens)

    # ------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict:
        return self.memory_system.get_memory_stats()

    def get_and_reset_internal_stats(self) -> Dict:
        return self.memory_system.get_and_reset_internal_stats()
