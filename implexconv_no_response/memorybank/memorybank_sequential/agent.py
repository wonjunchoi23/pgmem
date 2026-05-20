"""
MemoryBank Agent Module

MemoryBankAgent
---------------
Wraps MemoryBankSystem with the experiment interface:
  - add_memory / retrieve_memory
  - build_response_prompt  (construct prompt only — NO LLM call; logged as call_1_response)
  - answer_qa              (event summary + user portrait + retrieved memory + question)
  - on_conv_boundary       (daily event + personality summarization)
  - on_session_end         (global summary synthesis)
  - clear_memory / save_memory_snapshot / get_memory_count
  - get_and_reset_summary_tokens  (summarization LLM token counts)
  - set_llm_logger         (inject LLMCallLogger for per-session prompt logging)

LLM call types:
  call_1_response         — response prompt (output=null, Phase 1)
  call_2_daily_event      — daily event summary (memory build)
  call_3_daily_personality — daily personality summary (memory build)
  call_4_global_event     — global event summary (Phase 1→2 transition)
  call_5_global_personality — global personality summary (Phase 1→2 transition)
  call_6_qa               — QA answering (Phase 2)
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# config is loaded dynamically in run_experiment.py and injected into
# sys.modules['config'] before this module is imported.
import config as cfg

from memory_bank import MemoryBankSystem, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# LLM CALL LOGGER
# =============================================================================

class LLMCallLogger:
    """
    Logs all LLM calls (input + output) to call-type-specific folders as JSONL.

    Folder structure (per session):
        {base_dir}/call_1_response/calls.jsonl          # Response prompt, output=null (Phase 1)
        {base_dir}/call_2_daily_event/calls.jsonl       # Daily event summary (memory build)
        {base_dir}/call_3_daily_personality/calls.jsonl # Daily personality summary (memory build)
        {base_dir}/call_4_global_event/calls.jsonl      # Global event summary (Phase 1→2)
        {base_dir}/call_5_global_personality/calls.jsonl # Global personality summary (Phase 1→2)
        {base_dir}/call_6_qa/calls.jsonl                # QA answering (Phase 2)
    """

    CALL_DIRS = [
        "call_1_response",
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
# PROMPT TEMPLATES
# =============================================================================

RESPONSE_SYSTEM_PROMPT = """\
Now you will play the role of an AI Companion named SiliconFriend. \
You should be able to: (1) provide warm companionship to chat users; \
(2) understand past [memory], and if they are relevant to the current question, \
you must extract information from the [memory] to answer the question; \
(3) you are also an excellent psychological counselor, and when users confide \
in you about their difficulties and seek help, you can provide them with warm \
and helpful responses."""

RESPONSE_PROMPT = """\
{event_summary_section}\
{user_portrait_section}\
{memory_section}\
{history_section}\
You should refer to the context of the conversation, past [memory], and provide \
detailed answers to user questions. Please respond in English (maximum 100 words).
[|User|]: {user_utterance}
[|AI|]: """

QA_SYSTEM_PROMPT = """\
Now you will play the role of an AI Companion named SiliconFriend. \
You should be able to: (1) provide warm companionship to chat users; \
(2) understand past [memory], and if they are relevant to the current question, \
you must extract information from the [memory] to answer the question; \
(3) you are also an excellent psychological counselor, and when users confide \
in you about their difficulties and seek help, you can provide them with warm \
and helpful responses."""

QA_PROMPT_OPPOSED = """\
{event_summary_section}\
{user_portrait_section}\
{memory_section}\
{history_section}\
You should refer to the past [memory] and answer the following question. \
Provide a concise answer in English (maximum 100 words).
[|User|]: {question}
[|AI|]: """

QA_PROMPT_SUPPORTIVE = """\
{event_summary_section}\
{user_portrait_section}\
{memory_section}\
{history_section}\
Based on the above information, answer the following yes/no question about the user. \
You MUST answer with exactly one of: yes, no.
[|User|]: {question}
Answer: """


# =============================================================================
# JSON SCHEMAS
# =============================================================================

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"response": {"type": "string"}},
    "required": ["response"],
    "additionalProperties": False,
}

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
# PROMPT SECTION BUILDERS
# =============================================================================

def _build_event_summary_section(summary: str) -> str:
    if not summary:
        return ""
    return f"The summary of your past memories with the user is: {summary}\n\n"


def _build_user_portrait_section(portrait: str) -> str:
    if not portrait:
        return ""
    return (
        f"The personality of the user and the AI Companion's response strategy "
        f"are: {portrait}\n\n"
    )


def _build_memory_section(memory: str) -> str:
    # Always include the memory slot (empty string if no memory), matching original
    # which always passes related_memory_content to the prompt.
    content = memory.strip() if memory else ""
    return (
        f'Based on the current user\'s question, you start recalling past '
        f'conversations, and the [memory] most relevant to the question is: '
        f'"{content}"\n\n'
    )


def _build_history_section(history: list) -> str:
    """Format recent (user_turn, assistant_turn) pairs using original [|User|]/[|AI|] labels."""
    if not history:
        return ""
    lines = []
    for user_turn, assistant_turn in history:
        if user_turn:
            lines.append(f"[|User|]: {user_turn.utterance}")
        if assistant_turn:
            lines.append(f"[|AI|]: {assistant_turn.utterance}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


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
    """If the error is a prompt-too-long error, return (prompt_tokens, max_tokens)."""
    m = _PROMPT_TOO_LONG_RE.search(str(error))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# =============================================================================
# TOKEN HELPER
# =============================================================================

def extract_token_info(response, model_path: str = "") -> Dict:
    """Extract token usage from an LLM response dict."""
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
    """
    MemoryBank agent wrapping MemoryBankSystem.

    Provides the experiment interface for memory operations, LLM generation,
    and hierarchical summarization.
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
        """Inject a per-session LLMCallLogger. Also propagated to memory_system."""
        self._llm_logger = llm_logger
        self.memory_system.set_llm_logger(llm_logger)

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def add_memory(self, content: str, conv_id: int, timestamp: str):
        """Store a dialogue turn (embedding only, no LLM call)."""
        self.memory_system.add_memory(content, conv_id, timestamp)

    def retrieve_memory(
        self,
        query: str,
        current_conv_id: int,
        k: int = None,
        update_strength: bool = True,
    ) -> RetrievalResult:
        """Retrieve memories with forgetting curve re-ranking."""
        k = k or cfg.RETRIEVE_K
        return self.memory_system.retrieve(
            query, k, current_conv_id, update_strength
        )

    # ------------------------------------------------------------------
    # Response prompt construction (NO LLM call)
    # ------------------------------------------------------------------

    def build_response_prompt(
        self,
        user_utterance: str,
        retrieved_memory: str,
        history: list = None,
    ) -> str:
        """
        Construct the response generation prompt WITHOUT calling the LLM.

        Logs the constructed prompt to call_1_response (output=null) if a
        logger is set. Returns the prompt string for token estimation.

        Args:
            user_utterance: Current user message.
            retrieved_memory: Formatted retrieval string from retrieve_memory().
            history: Recent (user_turn, assistant_turn) pairs (max MAX_HISTORY_PAIRS).

        Returns:
            prompt_str: The full constructed prompt.
        """
        event_summary = self.memory_system.get_event_summary()
        user_portrait = self.memory_system.get_user_portrait()
        memory_str = retrieved_memory if retrieved_memory else ""

        prompt = RESPONSE_PROMPT.format(
            event_summary_section=_build_event_summary_section(event_summary),
            user_portrait_section=_build_user_portrait_section(user_portrait),
            memory_section=_build_memory_section(memory_str),
            history_section=_build_history_section(history or []),
            user_utterance=user_utterance,
        )

        if self._llm_logger is not None:
            self._llm_logger.log(
                call_type="call_1_response",
                system_prompt=RESPONSE_SYSTEM_PROMPT,
                user_prompt=prompt,
                output=None,  # No LLM call — always null
            )

        return prompt

    # ------------------------------------------------------------------
    # QA answering
    # ------------------------------------------------------------------

    def answer_qa(
        self,
        question: str,
        retrieved_memory: str,
        subset: str = "opposed",
        history: list = None,
    ) -> Tuple[str, Dict, str]:
        """
        Answer a QA question from memory.

        Uses question text directly as retrieval query (no keyword generation),
        following the original MemoryBank approach.

        For the supportive subset, the answer is normalised to one of {yes, no}.
        Fallback on error is "unknown" (excluded from the guided schema but used
        as a safety net to preserve label validity).
        For the opposed subset, the answer is free-form.

        Args:
            question: QA question text.
            retrieved_memory: Formatted retrieval string from retrieve_memory().
            subset: "opposed" (free-form) or "supportive" (yes/no).
            history: Recent (user_turn, assistant_turn) pairs for context.

        Returns:
            Tuple of (answer_str, token_info, prompt_snapshot).
        """
        event_summary = self.memory_system.get_event_summary()
        user_portrait = self.memory_system.get_user_portrait()
        memory_str = retrieved_memory if retrieved_memory else ""

        schema = (
            QA_SCHEMA_SUPPORTIVE
            if subset == "supportive"
            else QA_SCHEMA_OPPOSED
        )
        prompt_template = (
            QA_PROMPT_SUPPORTIVE
            if subset == "supportive"
            else QA_PROMPT_OPPOSED
        )

        for attempt in range(_MEMORY_RETRY_MAX):
            prompt = prompt_template.format(
                event_summary_section=_build_event_summary_section(
                    event_summary
                ),
                user_portrait_section=_build_user_portrait_section(
                    user_portrait
                ),
                memory_section=_build_memory_section(memory_str),
                history_section=_build_history_section(history or []),
                question=question,
            )
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=QA_SYSTEM_PROMPT,
                    guided_json=schema,
                    temperature=cfg.TEMPERATURE,
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

                if subset == "supportive":
                    label = answer.strip().lower()
                    if label not in ("yes", "no"):
                        label = "unknown"
                    answer = label

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
                    # Fallback: "unknown" for supportive (preserves label validity),
                    # empty string for opposed (free-form).
                    fallback = "unknown" if subset == "supportive" else ""
                    return (
                        fallback,
                        {"input": 0, "output": 0, "model": self.model_path},
                        prompt,
                    )

    # ------------------------------------------------------------------
    # Summarization triggers
    # ------------------------------------------------------------------

    def apply_forgetting(self, current_conv_id: int):
        """
        Probabilistically delete memories based on the Ebbinghaus forgetting curve.
        Call at each conv_id boundary BEFORE on_conv_boundary().
        Matches original MemoryBank forget_memory.py behaviour.
        """
        self.memory_system.apply_forgetting(current_conv_id)

    def on_conv_boundary(self, conv_id: int, dialogue_text: str):
        """
        Trigger daily event + personality summarization for a completed conv_id.
        Call when a new conv_id is detected (= "new day" boundary).
        Makes 2 LLM calls.
        """
        self.memory_system.summarize_daily(conv_id, dialogue_text)

    def on_session_end(self):
        """
        Synthesize global summaries from all daily summaries.
        Call between Phase 1 and Phase 2 so QA can use global summaries.
        Makes 2 LLM calls.
        """
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

    def get_and_reset_summary_tokens(self) -> Dict:
        """
        Return and reset accumulated summarization LLM token counts.

        Call after on_conv_boundary() or on_session_end() to capture
        summarization overhead.
        Returns Dict with "input", "output", "api_calls".
        """
        return self.memory_system.get_and_reset_token_counts()
