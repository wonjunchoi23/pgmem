"""
MemoryBank Memory System — LoComo batch variant

Core module implementing the MemoryBank memory mechanism adapted for LoComo.

Key design decisions:
- Forgetting applied once before Phase 2 (matches original MemoryBank behaviour)
- Separate daily_summary_retriever for session summaries (not forgotten)
- Calendar-date based forgetting (actual dates from dataset, not virtual days)
- Two-speaker personality prompts (LoComo has two human participants)
- Memory content prefixed with date label for retrieval context

Paper: MemoryBank: Enhancing Large Language Models with Long-Term Memory (AAAI 2024)
"""

import math
import random
import re
import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date as _date
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)


# =============================================================================
# DATE HELPERS
# =============================================================================

_DATE_PATTERN = re.compile(r'(\d{1,2})\s+(\w+),?\s+(\d{4})')
_DATE_FORMATS = ["%d %B %Y", "%d %b %Y"]


def _parse_date(date_time_str: str) -> _date:
    """Parse a calendar date from a LoComo date_time string.

    Handles formats like '1:56 pm on 8 May, 2023' or '8 May, 2023'.
    Falls back to today on failure.
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


def _format_source_label(date_time_str: str) -> str:
    """Extract clean date label from a raw date_time string.

    E.g. '1:56 pm on 8 May, 2023' -> '8 May, 2023'
    Falls back to the raw string if no match.
    """
    m = _DATE_PATTERN.search(date_time_str)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return f"{day} {month}, {year}"
    return date_time_str


def _strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> blocks (complete or truncated) from text."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


# =============================================================================
# SUMMARIZATION PROMPTS
# =============================================================================

DAILY_EVENT_PROMPT = """\
Please summarize the following dialogue as concisely as possible, extracting \
the main themes and key information. If there are multiple key events, you may \
summarize them separately. Dialogue content:
{dialogue}

Summarization:"""

# Two-speaker variant: LoComo has two human participants.
DAILY_PERSONALITY_PROMPT = """\
Based on the following dialogue between {speaker_a} and {speaker_b}, please \
summarize the personality traits, emotions, and communication patterns of both \
speakers. Dialogue content:
{dialogue}

Personality analysis of {speaker_a} and {speaker_b}:"""

GLOBAL_EVENT_PROMPT = """\
Please provide a highly concise summary of the following events, capturing the \
essential key information as succinctly as possible. Summarize the events:
{daily_summaries}

Summarization:"""

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
    """Single dialogue memory unit — subject to forgetting."""
    id: str
    content: str
    timestamp: str          # turn identifier (dia_id)
    session_id: int
    source_label: str       # parsed date string, e.g. "8 May, 2023"
    strength: int = 1
    last_recall_date_str: str = ""   # used by forgetting curve; initialized to source_label


@dataclass
class DailySummaryEntry:
    """Session event summary — separate retrieval store, not subject to forgetting."""
    id: str
    content: str
    session_id: int
    source_label: str


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    formatted: str
    memo_dates: str = ""
    items: List[Dict] = field(default_factory=list)
    total_memories: int = 0
    total_daily_summaries: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)
    prompt_block_count: int = 0


# =============================================================================
# MEMORYBANK SYSTEM
# =============================================================================

class MemoryBankSystem:
    """
    Core MemoryBank memory system for LoComo.

    Two-store retrieval:
      - entries / retriever: dialogue turns, subject to Ebbinghaus forgetting
      - daily_summary_entries / daily_summary_retriever: session event summaries, never forgotten

    Forgetting is applied once before Phase 2 (matching original MemoryBank behaviour).
    """

    def __init__(
        self,
        llm_client,
        embedding_model="all-MiniLM-L6-v2",
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

        # Dialogue memory store (subject to forgetting)
        self.entries: List[MemoryEntry] = []
        self.retriever = EmbeddingRetriever(embedding_model)

        # Daily summary store (NOT forgotten)
        self.daily_summary_entries: List[DailySummaryEntry] = []
        self.daily_summary_retriever = EmbeddingRetriever(embedding_model)

        # Hierarchical summaries
        self.daily_event_summaries: Dict[int, str] = {}
        self.daily_personality_summaries: Dict[int, str] = {}
        self.session_dates: Dict[int, str] = {}   # session_id -> source_label (for global prompts)
        self.global_event_summary: str = ""
        self.global_user_portrait: str = ""

        # Per-call-type token tracking (call_1~call_4 = LoComo numbering)
        self._token_counts = {
            "call_1_daily_event":        {"input": 0, "output": 0, "llm_calls": 0},
            "call_2_daily_personality":  {"input": 0, "output": 0, "llm_calls": 0},
            "call_3_global_event":       {"input": 0, "output": 0, "llm_calls": 0},
            "call_4_global_personality": {"input": 0, "output": 0, "llm_calls": 0},
        }

        self._forgetting_events = 0
        self._memories_forgotten = 0
        self._llm_logger = None

    def set_llm_logger(self, llm_logger):
        self._llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Forgetting curve — applied once before Phase 2
    # ------------------------------------------------------------------

    def _compute_retention(self, entry: MemoryEntry, now_date: _date) -> float:
        """Ebbinghaus: R = exp(-day_gap / (DIVISOR * S)), calendar-date based."""
        ref_date_str = entry.last_recall_date_str or entry.source_label
        ref_date = _parse_date(ref_date_str)
        day_gap = max((now_date - ref_date).days, 0)
        if day_gap == 0:
            return 1.0
        return math.exp(-day_gap / (self.forgetting_divisor * entry.strength))

    def apply_forgetting(self, now_date_str: str):
        """Probabilistically delete memories based on retention probability.

        Called once after all sessions are processed (Phase 1 end),
        matching original MemoryBank forget_memory.py behaviour.
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

        self._forgetting_events += 1
        self._memories_forgotten += len(forget_indices)
        self.retriever.remove_by_indices(forget_indices)
        for idx in sorted(forget_indices, reverse=True):
            self.entries.pop(idx)

        logger.info(
            f"apply_forgetting(now={now_date}): "
            f"removed {len(forget_indices)} memories, {len(self.entries)} remaining"
        )

    # ------------------------------------------------------------------
    # Memory storage
    # ------------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        session_id: int,
        date_str: str,
        timestamp: str,
    ):
        """Store a dialogue turn as a retrievable memory entry.

        Content is prefixed with the date label for retrieval context.
        """
        source_label = _format_source_label(date_str)
        full_content = f"Conversation content on {source_label}: {content}"
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=full_content,
            timestamp=timestamp,
            session_id=session_id,
            source_label=source_label,
            last_recall_date_str=source_label,
        )
        self.entries.append(entry)
        self.retriever.add_document(full_content)

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int,
        update_strength: bool = False,
    ) -> RetrievalResult:
        """Retrieve top-k memories from both dialogue and summary stores.

        Steps:
          1. Search each retriever for top-k candidates
          2. Merge and select top-k by score
          3. Re-sort chronologically (by source_label date) for prompt
          4. Group into per-date blocks
        """
        total_memories = len(self.entries)
        total_daily_summaries = len(self.daily_summary_entries)

        if not self.entries and not self.daily_summary_entries:
            return RetrievalResult(
                formatted="", memo_dates="", items=[],
                total_memories=0, total_daily_summaries=0,
                type_counts={"dialogue_snippet": 0, "daily_summary": 0},
                prompt_block_count=0,
            )

        dialogue_results = []
        if self.entries:
            dialogue_results = self.retriever.search_with_scores(
                query, min(k, len(self.entries))
            )

        summary_results = []
        if self.daily_summary_entries:
            summary_results = self.daily_summary_retriever.search_with_scores(
                query, min(k, len(self.daily_summary_entries))
            )

        combined = []
        for idx, score in dialogue_results:
            combined.append({"subtype": "dialogue_snippet", "idx": idx, "score": score})
        for idx, score in summary_results:
            combined.append({"subtype": "daily_summary", "idx": idx, "score": score})

        if not combined:
            return RetrievalResult(
                formatted="", memo_dates="", items=[],
                total_memories=total_memories,
                total_daily_summaries=total_daily_summaries,
                type_counts={"dialogue_snippet": 0, "daily_summary": 0},
                prompt_block_count=0,
            )

        combined.sort(key=lambda x: x["score"], reverse=True)
        selected = combined[: min(k, len(combined))]

        if update_strength:
            for item in selected:
                if item["subtype"] == "dialogue_snippet":
                    self.entries[item["idx"]].strength += 1

        # Re-sort chronologically for prompt; dialogue before summary within same date
        selected_for_prompt = sorted(
            selected,
            key=lambda x: (
                _parse_date(self._get_source_label(x)),
                0 if x["subtype"] == "dialogue_snippet" else 1,
            ),
        )

        type_counts = {"dialogue_snippet": 0, "daily_summary": 0}
        items = []
        prompt_blocks = []
        memo_labels = []
        current_label = None
        current_lines = []

        for item in selected_for_prompt:
            doc = self._get_doc(item)
            subtype = item["subtype"]
            type_counts[subtype] += 1

            if doc["source_label"] != current_label:
                if current_lines:
                    prompt_blocks.append("\n".join(current_lines))
                current_label = doc["source_label"]
                memo_labels.append(current_label)
                current_lines = [doc["content"]]
            else:
                current_lines.append(doc["content"])

            log_item = {
                "memory_subtype": subtype,
                "source_label": doc["source_label"],
                "content_preview": doc["content"][:200],
                "score": round(item["score"], 6),
            }
            if subtype == "dialogue_snippet":
                log_item["timestamp"] = doc["timestamp"]
                log_item["strength"] = doc["strength"]
            else:
                log_item["source_summary_session_id"] = doc["session_id"]
            items.append(log_item)

        if current_lines:
            prompt_blocks.append("\n".join(current_lines))

        return RetrievalResult(
            formatted="\n".join(prompt_blocks),
            memo_dates=", ".join(memo_labels),
            items=items,
            total_memories=total_memories,
            total_daily_summaries=total_daily_summaries,
            type_counts=type_counts,
            prompt_block_count=len(prompt_blocks),
        )

    def _get_source_label(self, item: Dict) -> str:
        if item["subtype"] == "dialogue_snippet":
            return self.entries[item["idx"]].source_label
        return self.daily_summary_entries[item["idx"]].source_label

    def _get_doc(self, item: Dict) -> Dict:
        if item["subtype"] == "dialogue_snippet":
            e = self.entries[item["idx"]]
            return {
                "content": e.content,
                "source_label": e.source_label,
                "timestamp": e.timestamp,
                "session_id": e.session_id,
                "strength": e.strength,
            }
        e = self.daily_summary_entries[item["idx"]]
        return {
            "content": e.content,
            "source_label": e.source_label,
            "session_id": e.session_id,
        }

    # ------------------------------------------------------------------
    # Daily summary index helpers
    # ------------------------------------------------------------------

    def _rebuild_daily_summary_index(self):
        self.daily_summary_retriever.corpus.clear()
        self.daily_summary_retriever.embeddings = None
        for entry in self.daily_summary_entries:
            self.daily_summary_retriever.add_document(entry.content)

    def _upsert_daily_summary_memory(
        self, session_id: int, source_label: str, event_summary: str
    ):
        summary_content = (
            f"The summary of the conversation on {source_label} is: {event_summary}"
        )
        for idx, entry in enumerate(self.daily_summary_entries):
            if entry.session_id == session_id:
                self.daily_summary_entries[idx] = DailySummaryEntry(
                    id=entry.id,
                    content=summary_content,
                    session_id=session_id,
                    source_label=source_label,
                )
                self._rebuild_daily_summary_index()
                return
        self.daily_summary_entries.append(
            DailySummaryEntry(
                id=str(uuid.uuid4()),
                content=summary_content,
                session_id=session_id,
                source_label=source_label,
            )
        )
        self.daily_summary_retriever.add_document(summary_content)

    # ------------------------------------------------------------------
    # Hierarchical summarization — batch APIs (prompt builders + appliers)
    # ------------------------------------------------------------------

    def build_daily_summary_prompts(
        self, dialogue_text: str, speaker_a: str, speaker_b: str
    ) -> Tuple[str, str]:
        """Build daily event and personality prompts without calling LLM."""
        return (
            DAILY_EVENT_PROMPT.format(dialogue=dialogue_text),
            DAILY_PERSONALITY_PROMPT.format(
                dialogue=dialogue_text,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
            ),
        )

    def apply_daily_summary_results(
        self,
        session_id: int,
        date_str: str,
        event_summary: str,
        personality_summary: str,
    ):
        """Apply batch LLM results for daily summarization.

        Strips think blocks, stores summaries, and upserts the event summary
        into the daily_summary_retriever for retrieval in Phase 2.
        """
        source_label = _format_source_label(date_str)
        event_summary = _strip_think_blocks(event_summary or "")
        personality_summary = _strip_think_blocks(personality_summary or "")

        if event_summary:
            self.daily_event_summaries[session_id] = event_summary
            self.session_dates[session_id] = source_label
            self._upsert_daily_summary_memory(session_id, source_label, event_summary)
        else:
            logger.warning(
                f"Daily event summary for session {session_id} was empty; "
                f"skipping summary index insertion."
            )

        if personality_summary:
            self.daily_personality_summaries[session_id] = personality_summary
        else:
            logger.warning(
                f"Daily personality summary for session {session_id} was empty."
            )

        logger.info(
            f"Daily summaries for session {session_id} ({source_label}): "
            f"event={len(event_summary)} chars, personality={len(personality_summary)} chars"
        )

    def build_global_summary_prompts(self) -> Optional[Tuple[str, str]]:
        """Build global event and personality prompts without calling LLM.

        Returns None if no daily summaries exist.
        """
        if not self.daily_event_summaries:
            return None

        daily_event_text = "\n".join(
            f"On {self.session_dates.get(sid, 'unknown date')}: {summary}"
            for sid, summary in sorted(self.daily_event_summaries.items())
        )
        daily_personality_text = "\n".join(
            f"On {self.session_dates.get(sid, 'unknown date')}: {summary}"
            for sid, summary in sorted(self.daily_personality_summaries.items())
        )

        return (
            GLOBAL_EVENT_PROMPT.format(daily_summaries=daily_event_text),
            GLOBAL_PERSONALITY_PROMPT.format(daily_personalities=daily_personality_text),
        )

    def apply_global_summary_results(self, event_summary: str, user_portrait: str):
        """Apply batch LLM results for global synthesis."""
        event_summary = _strip_think_blocks(event_summary or "")
        user_portrait = _strip_think_blocks(user_portrait or "")

        if event_summary:
            self.global_event_summary = event_summary
        else:
            logger.warning("Global event synthesis returned empty; keeping previous value.")

        if user_portrait:
            self.global_user_portrait = user_portrait
        else:
            logger.warning("Global personality synthesis returned empty; keeping previous value.")

        logger.info(
            f"Global summaries synthesized: "
            f"event={len(self.global_event_summary)} chars, "
            f"portrait={len(self.global_user_portrait)} chars"
        )

    # ------------------------------------------------------------------
    # Sequential fallback APIs (preserved for fallback)
    # ------------------------------------------------------------------

    def summarize_daily(
        self,
        session_id: int,
        date_str: str,
        dialogue_text: str,
        speaker_a: str,
        speaker_b: str,
    ):
        """Generate daily event and personality summaries (sequential path)."""
        ep, pp = self.build_daily_summary_prompts(dialogue_text, speaker_a, speaker_b)
        event_summary = self._call_llm_for_summary(ep, "call_1_daily_event")
        personality_summary = self._call_llm_for_summary(pp, "call_2_daily_personality")
        self.apply_daily_summary_results(session_id, date_str, event_summary, personality_summary)

    def synthesize_global(self):
        """Synthesize global summaries from all daily summaries (sequential path)."""
        prompts = self.build_global_summary_prompts()
        if prompts is None:
            logger.info("No daily summaries to synthesize.")
            return
        event_prompt, personality_prompt = prompts
        event_summary = self._call_llm_for_summary(event_prompt, "call_3_global_event")
        user_portrait = self._call_llm_for_summary(personality_prompt, "call_4_global_personality")
        self.apply_global_summary_results(event_summary, user_portrait)

    def _call_llm_for_summary(self, prompt: str, call_type: str) -> str:
        """Call LLM for summarization with token tracking and call logging."""
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                system_prompt=None,
                temperature=self.summarize_temperature,
                max_tokens=self.summarize_max_tokens,
                return_usage=True,
            )

            if call_type in self._token_counts:
                self._token_counts[call_type]["llm_calls"] += 1

            if isinstance(result, dict):
                if "_usage" in result:
                    usage = result["_usage"]
                    if call_type in self._token_counts:
                        self._token_counts[call_type]["input"] += usage.get("prompt_tokens", 0)
                        self._token_counts[call_type]["output"] += usage.get("completion_tokens", 0)
                text = result.get("content", "")
            else:
                text = str(result)

            text = _strip_think_blocks(text)

            if self._llm_logger is not None:
                self._llm_logger.log(
                    call_type=call_type,
                    system_prompt=None,
                    user_prompt=prompt,
                    output=text,
                )

            return text

        except Exception as e:
            logger.error(f"Summarization LLM error ({call_type}): {e}")
            if call_type in self._token_counts:
                self._token_counts[call_type]["llm_calls"] += 1
            if self._llm_logger is not None:
                self._llm_logger.log(
                    call_type=call_type,
                    system_prompt=None,
                    user_prompt=prompt,
                    output=None,
                )
            return ""

    # ------------------------------------------------------------------
    # Batch token tracking
    # ------------------------------------------------------------------

    def accumulate_token_counts(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_calls: int,
        call_type: str,
    ):
        if call_type in self._token_counts:
            self._token_counts[call_type]["input"] += input_tokens
            self._token_counts[call_type]["output"] += output_tokens
            self._token_counts[call_type]["llm_calls"] += llm_calls

    # ------------------------------------------------------------------
    # Summary getters
    # ------------------------------------------------------------------

    def get_event_summary(self) -> str:
        if self.global_event_summary:
            return self.global_event_summary
        if self.daily_event_summaries:
            return "\n".join(
                f"On {self.session_dates.get(sid, '?')}: {s}"
                for sid, s in sorted(self.daily_event_summaries.items())
            )
        return ""

    def get_user_portrait(self) -> str:
        if self.global_user_portrait:
            return self.global_user_portrait
        if self.daily_personality_summaries:
            return "\n".join(
                f"On {self.session_dates.get(sid, '?')}: {s}"
                for sid, s in sorted(self.daily_personality_summaries.items())
            )
        return ""

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_and_reset_token_counts_by_type(self) -> Dict:
        counts = {ct: dict(vals) for ct, vals in self._token_counts.items()}
        for ct in self._token_counts:
            self._token_counts[ct] = {"input": 0, "output": 0, "llm_calls": 0}
        return counts

    def get_memory_stats(self) -> Dict:
        num_memories = len(self.entries)
        num_daily_summaries = len(self.daily_summary_entries)
        total_content_tokens = sum(len(e.content) for e in self.entries) // 4
        return {
            "num_memories": num_memories,
            "num_daily_summaries": num_daily_summaries,
            "total_retrieval_docs": num_memories + num_daily_summaries,
            "total_content_tokens": total_content_tokens,
        }

    def get_and_reset_internal_stats(self) -> Dict:
        stats = {
            "forgetting_events_count": self._forgetting_events,
            "memories_forgotten": self._memories_forgotten,
        }
        self._forgetting_events = 0
        self._memories_forgotten = 0
        return stats

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
        self.daily_summary_entries.clear()
        self.daily_summary_retriever.corpus.clear()
        self.daily_summary_retriever.embeddings = None
        self.daily_event_summaries.clear()
        self.daily_personality_summaries.clear()
        self.session_dates.clear()
        self.global_event_summary = ""
        self.global_user_portrait = ""
        for ct in self._token_counts:
            self._token_counts[ct] = {"input": 0, "output": 0, "llm_calls": 0}
        self._forgetting_events = 0
        self._memories_forgotten = 0
        # Note: _llm_logger is intentionally preserved; set_llm_logger() manages it.

    def save_snapshot(self, directory: Path):
        """Save memory state to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "entries.json", "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

        self.retriever.save(directory)

        summaries = {
            "daily_event_summaries": {
                str(k): v for k, v in self.daily_event_summaries.items()
            },
            "daily_personality_summaries": {
                str(k): v for k, v in self.daily_personality_summaries.items()
            },
            "session_dates": {
                str(k): v for k, v in self.session_dates.items()
            },
            "daily_summary_entries": [asdict(e) for e in self.daily_summary_entries],
            "global_event_summary": self.global_event_summary,
            "global_user_portrait": self.global_user_portrait,
        }
        with open(directory / "summaries.json", "w") as f:
            json.dump(summaries, f, indent=2)

        metadata = {
            "total_memories": len(self.entries),
            "daily_summaries_count": len(self.daily_event_summaries),
            "total_retrieval_docs": len(self.entries) + len(self.daily_summary_entries),
            "has_global_summary": bool(self.global_event_summary),
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved memory snapshot to {directory} "
            f"({len(self.entries)} entries, "
            f"{len(self.daily_summary_entries)} daily summaries)"
        )
