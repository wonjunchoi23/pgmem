"""
MemoryBank Memory System — LoComo variant

Core module implementing the MemoryBank memory mechanism adapted for LoComo:
- Embedding store for dialogue turns (numpy cosine similarity)
- Ebbinghaus forgetting curve: probabilistic deletion at Phase 2 start
  using actual calendar dates (last session's date_time as "now")
- Session-level summarization (event + two-speaker personality) after each session
- Global summarization (overall event + personality) after Phase 1
- Session summaries added to FAISS alongside raw turns
- Retrieval: raw cosine similarity top-k; forgotten memories already removed

Paper: MemoryBank: Enhancing Large Language Models with Long-Term Memory (AAAI 2024)
"""

import math
import random
import re
import uuid
import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, date as _date
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import config as cfg
from retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)


# =============================================================================
# DATE PARSING HELPER
# =============================================================================

_DATE_PATTERN = re.compile(r'(\d{1,2})\s+(\w+),?\s+(\d{4})')
_DATE_FORMATS = ["%d %B %Y", "%d %b %Y"]


def _parse_date(date_time_str: str) -> _date:
    """
    Parse a calendar date from a LoComo date_time string.

    Handles formats like '1:56 pm on 8 May, 2023' or '8 May, 2023'.
    Falls back to today's date on parse failure (logs a warning).
    """
    m = _DATE_PATTERN.search(date_time_str)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(f"{day} {month} {year}", fmt).date()
            except ValueError:
                pass
    logger.warning(f"Could not parse date from '{date_time_str}'; using today.")
    return _date.today()


# =============================================================================
# SUMMARIZATION PROMPTS
# =============================================================================

# Per-session event summary (adapted from original summarize_content_prompt, English)
SESSION_EVENT_PROMPT = """\
Please summarize the following dialogue as concisely as possible, extracting \
the main themes and key information. If there are multiple key events, you may \
summarize them separately. Dialogue content:
{dialogue}

Summarization:"""

# Per-session personality analysis — two-speaker variant
# (adapted from original summarize_person_prompt; covers both speakers together)
SESSION_PERSONALITY_PROMPT = """\
Based on the following dialogue between {speaker_a} and {speaker_b}, please \
summarize the personality traits, emotions, and communication patterns of both \
speakers. Dialogue content:
{dialogue}

Personality analysis of {speaker_a} and {speaker_b}:"""

# Global event summary (adapted from original summarize_overall_prompt, English)
GLOBAL_EVENT_PROMPT = """\
Please provide a highly concise summary of the following events, capturing the \
essential key information as succinctly as possible. Summarize the events:
{daily_summaries}

Summarization:"""

# Global personality summary — two-speaker variant
# (adapted from original summarize_overall_personality)
GLOBAL_PERSONALITY_PROMPT = """\
The following are the personality traits and communication patterns exhibited \
by both speakers throughout multiple conversations:
{daily_personalities}

Please provide a highly concise and general summary of both speakers' \
personalities and communication styles, summarized as:"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryEntry:
    """Single memory unit in the MemoryBank."""
    id: str
    content: str
    dia_id: Optional[str]       # None for session-summary entries
    session_id: int
    date_str: str               # parsed date string, e.g. "8 May, 2023"
    strength: int = 1           # starts at 1; +1 on recall (no Phase 1 retrieval → always 1)
    last_recall_date_str: str = ""  # initialised to date_str; updated on recall
    is_summary: bool = False    # True for session summary documents


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    formatted: str                                    # Formatted string for prompt context
    items: List[Dict] = field(default_factory=list)  # Detailed items for logging / metadata
    total_memories: int = 0


# =============================================================================
# MEMORYBANK SYSTEM
# =============================================================================

class MemoryBankSystem:
    """
    Core MemoryBank memory system for LoComo.

    Stores dialogue turns and session summaries as embeddings,
    retrieves via cosine similarity, and generates hierarchical summaries.

    Forgetting curve is applied once before Phase 2 using the last
    session's date as "now_date" (actual calendar days).
    """

    def __init__(
        self,
        llm_client,
        embedding_model: str = "all-MiniLM-L6-v2",
        forgetting_divisor: int = 5,
        retrieve_k: int = 6,
        summarize_temperature: float = 0.7,
        summarize_max_tokens: int = 400,
        json_retry: int = 3,
    ):
        self.llm_client = llm_client
        self.forgetting_divisor = forgetting_divisor
        self.retrieve_k = retrieve_k
        self.summarize_temperature = summarize_temperature
        self.summarize_max_tokens = summarize_max_tokens
        self.json_retry = json_retry

        # Memory store (entries and retriever stay in sync)
        self.entries: List[MemoryEntry] = []
        self.retriever = EmbeddingRetriever(embedding_model)

        # Hierarchical summaries
        self.session_event_summaries: Dict[int, str] = {}      # session_id → event summary
        self.session_personality_summaries: Dict[int, str] = {} # session_id → personality
        self.session_dates: Dict[int, str] = {}                 # session_id → date_str
        self.global_event_summary: str = ""
        self.global_user_portrait: str = ""

        # Token tracking for summarization LLM calls
        self._summary_input_tokens = 0
        self._summary_output_tokens = 0
        self._summary_api_calls = 0

        # LLM call logger (injected from outside, optional)
        self._llm_logger = None

    def set_llm_logger(self, llm_logger):
        """Inject an LLMCallLogger instance for prompt/output logging."""
        self._llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Forgetting curve
    # ------------------------------------------------------------------

    def _compute_retention(self, entry: MemoryEntry, now_date: _date) -> float:
        """
        Ebbinghaus retention: R = exp(-day_gap / (DIVISOR * S))

        day_gap = actual calendar days between last recall date and now_date.
        Since there is no Phase 1 retrieval, last_recall_date_str == date_str
        for all entries (strength = 1, S unchanged).
        """
        ref_date_str = entry.last_recall_date_str or entry.date_str
        ref_date = _parse_date(ref_date_str)
        day_gap = max((now_date - ref_date).days, 0)

        if day_gap == 0:
            return 1.0

        return math.exp(-day_gap / (self.forgetting_divisor * entry.strength))

    def apply_forgetting(self, now_date_str: str):
        """
        Probabilistically delete memories whose retention falls below random().

        Matches original MemoryBank forget_memory.py behaviour, adapted for
        actual calendar dates. Called once before Phase 2 using the last
        session's date_time string as now_date.

        Both self.entries and self.retriever are updated together so their
        indices always stay in sync.
        """
        if not self.entries:
            return

        now_date = _parse_date(now_date_str)

        forget_indices = []
        for i, entry in enumerate(self.entries):
            retention = self._compute_retention(entry, now_date)
            if random.random() > retention:
                forget_indices.append(i)

        if not forget_indices:
            logger.info(
                f"apply_forgetting(now={now_date}): "
                f"no memories forgotten, {len(self.entries)} remaining"
            )
            return

        # Remove from retriever first (same index space)
        self.retriever.remove_by_indices(forget_indices)

        # Remove from entries list in reverse order
        for idx in sorted(forget_indices, reverse=True):
            self.entries.pop(idx)

        logger.info(
            f"apply_forgetting(now={now_date}): "
            f"removed {len(forget_indices)} memories, "
            f"{len(self.entries)} remaining"
        )

    # ------------------------------------------------------------------
    # Memory storage
    # ------------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        dia_id: Optional[str],
        session_id: int,
        date_str: str,
        is_summary: bool = False,
    ):
        """
        Store a single memory entry (turn or session summary).

        Args:
            content:    Text to embed and store.
            dia_id:     Dialogue turn ID (e.g. "D1:3"); None for summary entries.
            session_id: LoComo session number.
            date_str:   Calendar date string for this session (for forgetting curve).
            is_summary: True if this is a session-level summary document.
        """
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            dia_id=dia_id,
            session_id=session_id,
            date_str=date_str,
            last_recall_date_str=date_str,
            is_summary=is_summary,
        )
        self.entries.append(entry)
        self.retriever.add_document(content)

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int,
        update_strength: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve top-k memories by cosine similarity.

        Forgetting is handled via apply_forgetting() before Phase 2 —
        forgotten entries are permanently deleted from both self.entries
        and self.retriever, so they never appear in retrieval results.

        Args:
            query:           Query string for embedding similarity search.
            k:               Maximum number of results to return.
            update_strength: If True, increment strength for recalled entries.
                             False during Phase 2 (memory frozen).

        Returns:
            RetrievalResult with formatted context string and item metadata.
        """
        if not self.entries:
            return RetrievalResult(formatted="", items=[], total_memories=0)

        k_actual = min(k, len(self.entries))
        raw_results = self.retriever.search_with_scores(query, k_actual)

        if not raw_results:
            return RetrievalResult(
                formatted="", items=[], total_memories=len(self.entries)
            )

        if update_strength:
            for idx, _ in raw_results:
                self.entries[idx].strength += 1
                # Note: last_recall_date_str stays as-is (no "current date" concept here)

        items = []
        for idx, score in raw_results:
            entry = self.entries[idx]
            items.append({
                "dia_id": entry.dia_id,           # None for summary entries
                "session_id": entry.session_id,
                "content_preview": entry.content[:200],
                "score": round(score, 6),
                "memory_type": (
                    "session_summary" if entry.is_summary else "dialogue_memory"
                ),
                "is_summary": entry.is_summary,
                "strength": entry.strength,
            })

        formatted_parts = [self.entries[idx].content for idx, _ in raw_results]
        formatted = (
            "\n".join(f"- {part}" for part in formatted_parts)
            if formatted_parts
            else ""
        )

        return RetrievalResult(
            formatted=formatted,
            items=items,
            total_memories=len(self.entries),
        )

    # ------------------------------------------------------------------
    # Hierarchical summarization
    # ------------------------------------------------------------------

    def summarize_session(
        self,
        session_id: int,
        date_str: str,
        dialogue_text: str,
        speaker_a: str,
        speaker_b: str,
    ):
        """
        Generate session-level event and personality summaries.

        Makes 2 LLM calls:
          1. Session event summary (call_1_session_event)
          2. Two-speaker personality analysis (call_2_session_personality)

        The event summary is also added to the embedding store as a
        searchable memory document (is_summary=True, dia_id=None).
        This matches original MemoryBank behaviour where summaries are
        indexed in FAISS alongside raw turns.
        """
        # ---- Event summary ----
        event_prompt = SESSION_EVENT_PROMPT.format(dialogue=dialogue_text)
        event_summary = self._call_llm_for_summary(
            event_prompt, call_type="call_1_session_event"
        )
        if event_summary:
            self.session_event_summaries[session_id] = event_summary
            self.session_dates[session_id] = date_str

            # Add summary to embedding store (searchable)
            summary_content = (
                f"Summary of conversation on {date_str}: {event_summary}"
            )
            self.add_memory(
                content=summary_content,
                dia_id=None,
                session_id=session_id,
                date_str=date_str,
                is_summary=True,
            )
        else:
            logger.warning(
                f"Session event summary for session {session_id} was empty; "
                f"skipping FAISS insertion."
            )

        # ---- Personality analysis (two speakers) ----
        personality_prompt = SESSION_PERSONALITY_PROMPT.format(
            dialogue=dialogue_text,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        personality_summary = self._call_llm_for_summary(
            personality_prompt, call_type="call_2_session_personality"
        )
        if personality_summary:
            self.session_personality_summaries[session_id] = personality_summary
        else:
            logger.warning(
                f"Session personality summary for session {session_id} was empty; "
                f"keeping previous value."
            )

        logger.info(
            f"Session summaries for session {session_id} ({date_str}): "
            f"event={len(event_summary)} chars, "
            f"personality={len(personality_summary)} chars"
        )

    def synthesize_global(self):
        """
        Synthesize global summaries from all per-session summaries.

        Makes 2 LLM calls:
          1. Global event summary (call_3_global_event)
          2. Global personality portrait (call_4_global_personality)

        Call after all sessions are processed (Phase 1 end) so Phase 2
        QA can use the global summaries as additional context.
        """
        if not self.session_event_summaries:
            logger.info("No session summaries to synthesize into global summary.")
            return

        # ---- Global event summary ----
        daily_event_text = "\n".join(
            f"On {self.session_dates.get(sid, 'unknown date')}: {summary}"
            for sid, summary in sorted(self.session_event_summaries.items())
        )
        event_prompt = GLOBAL_EVENT_PROMPT.format(daily_summaries=daily_event_text)
        new_event = self._call_llm_for_summary(
            event_prompt, call_type="call_3_global_event"
        )
        if new_event:
            self.global_event_summary = new_event
        else:
            logger.warning(
                "Global event synthesis returned empty; keeping previous value."
            )

        # ---- Global personality portrait ----
        daily_personality_text = "\n".join(
            f"On {self.session_dates.get(sid, 'unknown date')}: {summary}"
            for sid, summary in sorted(self.session_personality_summaries.items())
        )
        personality_prompt = GLOBAL_PERSONALITY_PROMPT.format(
            daily_personalities=daily_personality_text
        )
        new_portrait = self._call_llm_for_summary(
            personality_prompt, call_type="call_4_global_personality"
        )
        if new_portrait:
            self.global_user_portrait = new_portrait
        else:
            logger.warning(
                "Global personality synthesis returned empty; keeping previous value."
            )

        logger.info(
            f"Global summaries synthesized: "
            f"event={len(self.global_event_summary)} chars, "
            f"portrait={len(self.global_user_portrait)} chars"
        )

    def _call_llm_for_summary(self, prompt: str, call_type: str) -> str:
        """
        Call LLM for summarization with token tracking and call logging.

        Passes '/no_think' as system prompt to suppress Qwen3 chain-of-thought.
        Returns empty string on failure; callers handle fallback.
        """
        system_prompt = "/no_think"
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.summarize_temperature,
                max_tokens=self.summarize_max_tokens,
                return_usage=True,
            )

            self._summary_api_calls += 1

            if isinstance(result, dict):
                if "_usage" in result:
                    usage = result["_usage"]
                    self._summary_input_tokens += usage.get("prompt_tokens", 0)
                    self._summary_output_tokens += usage.get("completion_tokens", 0)
                text = result.get("content", "")
            else:
                text = str(result)

            # Strip residual <think> blocks (complete or truncated)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
            text = text.strip()

            if self._llm_logger is not None:
                self._llm_logger.log(
                    call_type=call_type,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    output=text,
                )

            return text

        except Exception as e:
            logger.error(f"Summarization LLM error ({call_type}): {e}")
            self._summary_api_calls += 1

            if self._llm_logger is not None:
                self._llm_logger.log(
                    call_type=call_type,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    output=None,
                )

            return ""

    # ------------------------------------------------------------------
    # Summary getters
    # ------------------------------------------------------------------

    def get_event_summary(self) -> str:
        """Return best available event summary (global > session concatenation)."""
        if self.global_event_summary:
            return self.global_event_summary
        if self.session_event_summaries:
            return "\n".join(
                f"On {self.session_dates.get(sid, '?')}: {s}"
                for sid, s in sorted(self.session_event_summaries.items())
            )
        return ""

    def get_user_portrait(self) -> str:
        """Return best available personality portrait (global > session concatenation)."""
        if self.global_user_portrait:
            return self.global_user_portrait
        if self.session_personality_summaries:
            return "\n".join(
                f"On {self.session_dates.get(sid, '?')}: {s}"
                for sid, s in sorted(self.session_personality_summaries.items())
            )
        return ""

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_and_reset_token_counts(self) -> Dict:
        """Return and reset summarization token counters."""
        counts = {
            "input": self._summary_input_tokens,
            "output": self._summary_output_tokens,
            "api_calls": self._summary_api_calls,
        }
        self._summary_input_tokens = 0
        self._summary_output_tokens = 0
        self._summary_api_calls = 0
        return counts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def get_memory_count(self) -> int:
        return len(self.entries)

    def clear(self):
        """Reset all memory state for a new sample."""
        self.entries.clear()
        self.retriever.corpus.clear()
        self.retriever.embeddings = None
        self.session_event_summaries.clear()
        self.session_personality_summaries.clear()
        self.session_dates.clear()
        self.global_event_summary = ""
        self.global_user_portrait = ""
        self._summary_input_tokens = 0
        self._summary_output_tokens = 0
        self._summary_api_calls = 0
        # Note: _llm_logger is intentionally preserved; set_llm_logger() manages it.

    def save_snapshot(self, directory: Path):
        """Save memory state to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Entries
        entries_data = [asdict(e) for e in self.entries]
        with open(directory / "entries.json", "w") as f:
            json.dump(entries_data, f, indent=2)

        # Embeddings + corpus
        self.retriever.save(directory)

        # Summaries
        summaries = {
            "session_event_summaries": {
                str(k): v for k, v in self.session_event_summaries.items()
            },
            "session_personality_summaries": {
                str(k): v for k, v in self.session_personality_summaries.items()
            },
            "session_dates": {
                str(k): v for k, v in self.session_dates.items()
            },
            "global_event_summary": self.global_event_summary,
            "global_user_portrait": self.global_user_portrait,
        }
        with open(directory / "summaries.json", "w") as f:
            json.dump(summaries, f, indent=2)

        # Metadata
        metadata = {
            "total_memories": len(self.entries),
            "session_summaries_count": len(self.session_event_summaries),
            "has_global_summary": bool(self.global_event_summary),
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved memory snapshot to {directory} "
            f"({len(self.entries)} entries, "
            f"{len(self.session_event_summaries)} session summaries)"
        )
