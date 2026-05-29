"""
MemoryBank Memory System

Core module implementing the MemoryBank memory mechanism:
- Embedding store for dialogue turns (custom FAISS-equivalent via numpy)
- Ebbinghaus forgetting curve: probabilistic permanent deletion at conv_id boundaries
- Retrieval: raw cosine similarity top-k (no re-ranking; forgotten memories already removed)
- Hierarchical summarization (daily event + personality → global)
- No LLM calls at storage time (embedding only)

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
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import config as cfg
from retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)


# =============================================================================
# SUMMARIZATION PROMPTS (adapted from original MemoryBank paper)
# =============================================================================

# Aligned with original MemoryBank summarize_memory.py (English versions).
# summarize_content_prompt → DAILY_EVENT_PROMPT
DAILY_EVENT_PROMPT = """\
Please summarize the following dialogue as concisely as possible, extracting \
the main themes and key information. If there are multiple key events, you may \
summarize them separately. Dialogue content:
{dialogue}

Summarization:"""

# summarize_person_prompt → DAILY_PERSONALITY_PROMPT
DAILY_PERSONALITY_PROMPT = """\
Based on the following dialogue, please summarize the user's personality traits \
and emotions, and devise response strategies based on your speculation. \
Dialogue content:
{dialogue}

The user's personality traits, emotions, and AI Companion's response strategy are:"""

# summarize_overall_prompt → GLOBAL_EVENT_PROMPT
GLOBAL_EVENT_PROMPT = """\
Please provide a highly concise summary of the following events, capturing the \
essential key information as succinctly as possible. Summarize the event:
{daily_summaries}

Summarization:"""

# summarize_overall_personality → GLOBAL_PERSONALITY_PROMPT
GLOBAL_PERSONALITY_PROMPT = """\
The following are the user's exhibited personality traits and emotions throughout \
multiple dialogues, along with appropriate response strategies for the current situation:
{daily_personalities}

Please provide a highly concise and general summary of the user's personality \
and the most appropriate response strategy for the AI Companion, summarized as:"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryEntry:
    """Single memory unit in the MemoryBank."""
    id: str
    content: str
    timestamp: str          # "SSSS_CCCC_TTTT" format
    conv_id: int            # For forgetting curve (treated as "day")
    source_label: str       # Virtual-day label used in QA prompt/logs
    strength: int = 1       # Starts at 1, +1 on recall
    last_recall_conv_id: int = -1   # -1 = never recalled; updated on recall


@dataclass
class DailySummaryEntry:
    """Daily event summary used as a separate QA retrieval document."""
    id: str
    content: str
    conv_id: int
    source_label: str


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    formatted: str                              # Formatted string for prompt
    memo_dates: str = ""
    items: List[Dict] = field(default_factory=list)   # Detailed items for logging
    total_memories: int = 0
    total_daily_summaries: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)
    prompt_block_count: int = 0


# =============================================================================
# MEMORYBANK SYSTEM
# =============================================================================

class MemoryBankSystem:
    """
    Core MemoryBank memory system.

    Stores dialogue turns as embeddings, retrieves via cosine similarity
    over the currently retained memory set, applies Ebbinghaus-style
    forgetting separately at conv_id boundaries, and generates hierarchical
    summaries (daily event + personality -> global).
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

        # Retrieval stores:
        # - dialogue snippets (forgetting applies)
        # - daily event summaries (QA retrieval only; not forgotten)
        self.entries: List[MemoryEntry] = []
        self.retriever = EmbeddingRetriever(embedding_model)
        self.daily_summary_entries: List[DailySummaryEntry] = []
        self.daily_summary_retriever = EmbeddingRetriever(embedding_model)

        # Hierarchical summaries
        self.daily_event_summaries: Dict[int, str] = {}
        self.daily_personality_summaries: Dict[int, str] = {}
        self.global_event_summary: str = ""
        self.global_user_portrait: str = ""

        # Per-call-type token tracking for summarization LLM calls
        self._daily_event_tokens    = {"input": 0, "output": 0, "llm_calls": 0}
        self._daily_person_tokens   = {"input": 0, "output": 0, "llm_calls": 0}
        self._global_event_tokens   = {"input": 0, "output": 0, "llm_calls": 0}
        self._global_person_tokens  = {"input": 0, "output": 0, "llm_calls": 0}

        # Phase 1 internal statistics
        self._forgetting_events = 0
        self._memories_forgotten = 0

        # LLM call logger (injected from outside, optional)
        self._llm_logger = None

    def set_llm_logger(self, llm_logger):
        """Inject an LLMCallLogger instance for prompt/output logging."""
        self._llm_logger = llm_logger

    def log_summary_call(self, call_type: str, prompt: str, output) -> None:
        if self._llm_logger is None:
            return
        self._llm_logger.log(
            call_type=call_type,
            system_prompt=None,
            user_prompt=prompt,
            output=output,
        )

    @staticmethod
    def _virtual_day_index(conv_id: int) -> int:
        """Map conv_id to a 1-based virtual day index."""
        return (conv_id // cfg.CONVS_PER_DAY) + 1

    @classmethod
    def _source_label_for_conv_id(cls, conv_id: int) -> str:
        return f"Day {cls._virtual_day_index(conv_id)}"

    def _rebuild_daily_summary_index(self):
        """Rebuild the daily-summary embedding index from current entries."""
        self.daily_summary_retriever.corpus.clear()
        self.daily_summary_retriever.embeddings = None
        for entry in self.daily_summary_entries:
            self.daily_summary_retriever.add_document(entry.content)

    # ------------------------------------------------------------------
    # Forgetting curve
    # ------------------------------------------------------------------

    def _compute_retention(
        self, entry: MemoryEntry, current_conv_id: int
    ) -> float:
        """
        Ebbinghaus retention: R = exp(-day_gap / (DIVISOR * S))

        day_gap = conv_gap / CONVS_PER_DAY  (converts conv_ids to days).
        If never recalled, conv_gap is measured from the entry's creation conv_id.
        """
        if entry.last_recall_conv_id >= 0:
            conv_gap = max(current_conv_id - entry.last_recall_conv_id, 0)
        else:
            conv_gap = max(current_conv_id - entry.conv_id, 0)

        if conv_gap == 0:
            return 1.0

        day_gap = conv_gap / cfg.CONVS_PER_DAY
        return math.exp(-day_gap / (self.forgetting_divisor * entry.strength))

    # ------------------------------------------------------------------
    # Memory storage (embedding only, no LLM)
    # ------------------------------------------------------------------

    def add_memory(self, content: str, conv_id: int, timestamp: str):
        """Store a dialogue snippet as a retrievable memory entry."""
        source_label = self._source_label_for_conv_id(conv_id)
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=f"Conversation content on {source_label}: {content}",
            timestamp=timestamp,
            conv_id=conv_id,
            source_label=source_label,
        )
        self.entries.append(entry)
        self.retriever.add_document(entry.content)

    # ------------------------------------------------------------------
    # Memory retrieval over the retained memory set
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int,
        current_conv_id: int,
        update_strength: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve top-k memories by raw cosine similarity (original MemoryBank).

        Forgetting is handled separately via apply_forgetting() at conv_id
        boundaries — memories that were forgotten (deleted) simply no longer
        appear in retrieval results. No re-ranking at retrieval time.

        1. Search embedding store for top-k by cosine similarity
        2. Optionally update strength for recalled entries (S += 1)

        Returns:
            RetrievalResult with formatted prompt string and logging items.
        """
        total_memories = len(self.entries)
        total_daily_summaries = len(self.daily_summary_entries)
        if not self.entries and not self.daily_summary_entries:
            return RetrievalResult(
                formatted="",
                memo_dates="",
                items=[],
                total_memories=0,
                total_daily_summaries=0,
                type_counts={
                    "dialogue_snippet": 0,
                    "daily_summary": 0,
                },
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

        combined_results = []
        for idx, score in dialogue_results:
            combined_results.append({
                "memory_subtype": "dialogue_snippet",
                "idx": idx,
                "score": score,
            })
        for idx, score in summary_results:
            combined_results.append({
                "memory_subtype": "daily_summary",
                "idx": idx,
                "score": score,
            })

        if not combined_results:
            return RetrievalResult(
                formatted="",
                memo_dates="",
                items=[],
                total_memories=total_memories,
                total_daily_summaries=total_daily_summaries,
                type_counts={
                    "dialogue_snippet": 0,
                    "daily_summary": 0,
                },
                prompt_block_count=0,
            )

        combined_results.sort(key=lambda item: item["score"], reverse=True)
        selected_results = combined_results[: min(k, len(combined_results))]

        if update_strength:
            for result in selected_results:
                if result["memory_subtype"] != "dialogue_snippet":
                    continue
                entry = self.entries[result["idx"]]
                entry.strength += 1
                entry.last_recall_conv_id = current_conv_id

        selected_results_for_prompt = sorted(
            selected_results,
            key=lambda item: (
                self._prompt_sort_day_index(item),
                0 if item["memory_subtype"] == "dialogue_snippet" else 1,
            ),
        )

        type_counts = {
            "dialogue_snippet": 0,
            "daily_summary": 0,
        }
        items = []
        prompt_blocks = []
        memo_labels = []
        current_label = None
        current_lines = []

        for result in selected_results_for_prompt:
            doc = self._get_selected_document(result)
            subtype = result["memory_subtype"]
            type_counts[subtype] += 1

            if doc["source_label"] != current_label:
                if current_lines:
                    prompt_blocks.append("\n".join(current_lines))
                current_label = doc["source_label"]
                memo_labels.append(current_label)
                current_lines = [doc["content"]]
            else:
                current_lines.append(doc["content"])

            item = {
                "memory_id": doc["id"],
                "memory_subtype": subtype,
                "source_label": doc["source_label"],
                "content_preview": doc["content"][:200],
                "score": round(result["score"], 6),
            }
            if subtype == "dialogue_snippet":
                item["source_turn"] = self._parse_timestamp(doc["timestamp"])
                item["strength"] = doc["strength"]
            else:
                item["source_summary_conv_id"] = doc["conv_id"]
            items.append(item)

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

    def _prompt_sort_day_index(self, result: Dict) -> int:
        doc = self._get_selected_document(result)
        return self._virtual_day_index(doc["conv_id"])

    def _get_selected_document(self, result: Dict) -> Dict:
        subtype = result["memory_subtype"]
        if subtype == "dialogue_snippet":
            entry = self.entries[result["idx"]]
            return {
                "id": entry.id,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "conv_id": entry.conv_id,
                "source_label": entry.source_label,
                "strength": entry.strength,
            }

        entry = self.daily_summary_entries[result["idx"]]
        return {
            "id": entry.id,
            "content": entry.content,
            "conv_id": entry.conv_id,
            "source_label": entry.source_label,
        }

    def apply_forgetting(self, current_conv_id: int):
        """
        Probabilistically delete memories whose retention falls below random(),
        matching the original MemoryBank forget_memory.py behaviour.

        For each entry: retention = exp(-day_gap / (DIVISOR * S))
        If random.random() > retention → permanently delete.

        Both self.entries and self.retriever (corpus + embeddings) are updated
        together so their indices always stay in sync.
        """
        if not self.entries:
            return

        forget_indices = []
        for i, entry in enumerate(self.entries):
            retention = self._compute_retention(entry, current_conv_id)
            if random.random() > retention:
                forget_indices.append(i)

        if not forget_indices:
            return

        self._forgetting_events += 1
        self._memories_forgotten += len(forget_indices)

        # Remove from embedding retriever first (uses same indices)
        self.retriever.remove_by_indices(forget_indices)

        # Remove from entries list in reverse order to preserve correct indices
        for idx in sorted(forget_indices, reverse=True):
            self.entries.pop(idx)

        logger.info(
            f"apply_forgetting(conv_id={current_conv_id}): "
            f"removed {len(forget_indices)} memories, "
            f"{len(self.entries)} remaining"
        )

    @staticmethod
    def _parse_timestamp(timestamp: str) -> Dict:
        """Parse 'SSSS_CCCC_TTTT' → {session_id, conv_id, turn_id}."""
        try:
            parts = timestamp.split("_")
            return {
                "session_id": int(parts[0]),
                "conv_id": int(parts[1]),
                "turn_id": int(parts[2]),
            }
        except (IndexError, ValueError):
            return {"session_id": -1, "conv_id": -1, "turn_id": -1}

    # ------------------------------------------------------------------
    # Hierarchical summarization
    # ------------------------------------------------------------------

    def build_daily_summary_prompts(
        self, dialogue_text: str
    ) -> Tuple[str, str]:
        return (
            DAILY_EVENT_PROMPT.format(dialogue=dialogue_text),
            DAILY_PERSONALITY_PROMPT.format(dialogue=dialogue_text),
        )

    def apply_daily_summary_results(
        self,
        conv_id: int,
        event_summary: str,
        personality_summary: str,
    ):
        event_summary = (event_summary or "").strip()
        personality_summary = (personality_summary or "").strip()

        if event_summary:
            self.daily_event_summaries[conv_id] = event_summary
            self._upsert_daily_summary_memory(conv_id, event_summary)
        else:
            logger.warning(
                f"Daily event summary for conv_id {conv_id} was empty; "
                f"keeping previous value."
            )

        if personality_summary:
            self.daily_personality_summaries[conv_id] = personality_summary
        else:
            logger.warning(
                f"Daily personality summary for conv_id {conv_id} was empty; "
                f"keeping previous value."
            )

        logger.info(
            f"Daily summaries for conv_id {conv_id}: "
            f"event={len(event_summary)} chars, "
            f"personality={len(personality_summary)} chars"
        )

    def _upsert_daily_summary_memory(self, conv_id: int, event_summary: str):
        source_label = self._source_label_for_conv_id(conv_id)
        summary_content = (
            f"The summary of the conversation on {source_label} is: "
            f"{event_summary}"
        )
        for idx, entry in enumerate(self.daily_summary_entries):
            if entry.conv_id == conv_id:
                self.daily_summary_entries[idx] = DailySummaryEntry(
                    id=entry.id,
                    content=summary_content,
                    conv_id=conv_id,
                    source_label=source_label,
                )
                self._rebuild_daily_summary_index()
                return

        self.daily_summary_entries.append(
            DailySummaryEntry(
                id=str(uuid.uuid4()),
                content=summary_content,
                conv_id=conv_id,
                source_label=source_label,
            )
        )
        self.daily_summary_retriever.add_document(summary_content)

    def build_global_summary_prompts(self) -> Optional[Tuple[str, str]]:
        if not self.daily_event_summaries:
            logger.info("No daily summaries to synthesize.")
            return None

        daily_event_text = "\n".join(
            f"Day {cid}: {summary}"
            for cid, summary in sorted(self.daily_event_summaries.items())
        )
        daily_personality_text = "\n".join(
            f"Day {cid}: {summary}"
            for cid, summary in sorted(
                self.daily_personality_summaries.items()
            )
        )

        return (
            GLOBAL_EVENT_PROMPT.format(daily_summaries=daily_event_text),
            GLOBAL_PERSONALITY_PROMPT.format(
                daily_personalities=daily_personality_text
            ),
        )

    def apply_global_summary_results(
        self, event_summary: str, user_portrait: str
    ):
        event_summary = (event_summary or "").strip()
        user_portrait = (user_portrait or "").strip()

        if event_summary:
            self.global_event_summary = event_summary
        else:
            logger.warning(
                "Global event synthesis returned empty; keeping previous value."
            )

        if user_portrait:
            self.global_user_portrait = user_portrait
        else:
            logger.warning(
                "Global personality synthesis returned empty; keeping previous value."
            )

        logger.info(
            f"Global summaries synthesized: "
            f"event={len(self.global_event_summary)} chars, "
            f"portrait={len(self.global_user_portrait)} chars"
        )

    _CALL_TYPE_TO_BUCKET = {
        "call_2_daily_event":        "_daily_event_tokens",
        "call_3_daily_personality":  "_daily_person_tokens",
        "call_4_global_event":       "_global_event_tokens",
        "call_5_global_personality": "_global_person_tokens",
    }

    def accumulate_call_tokens(
        self,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Accumulate token counts for a specific summarization call type."""
        bucket_name = self._CALL_TYPE_TO_BUCKET.get(call_type)
        if bucket_name is None:
            return
        bucket = getattr(self, bucket_name)
        bucket["input"] += input_tokens
        bucket["output"] += output_tokens
        bucket["llm_calls"] += 1

    def summarize_daily(self, conv_id: int, dialogue_text: str):
        """
        Generate daily event and personality summaries for a batch of conv_ids.

        Makes 2 LLM calls: one for event summary, one for personality.
        Fallback: if the LLM call returns empty (e.g. thinking truncation),
        the previous value for that key is kept unchanged.
        """
        event_prompt, personality_prompt = self.build_daily_summary_prompts(
            dialogue_text
        )
        event_summary = self._call_llm_for_summary(
            event_prompt, call_type="call_2_daily_event"
        )
        personality_summary = self._call_llm_for_summary(
            personality_prompt, call_type="call_3_daily_personality"
        )
        self.apply_daily_summary_results(
            conv_id,
            event_summary=event_summary,
            personality_summary=personality_summary,
        )

    def synthesize_global(self):
        """
        Synthesize global summaries from all daily summaries.

        Makes 2 LLM calls: one for global event, one for global personality.
        Call between Phase 1 and Phase 2 so QA can use global summaries.
        """
        prompts = self.build_global_summary_prompts()
        if prompts is None:
            return

        event_prompt, personality_prompt = prompts
        event_summary = self._call_llm_for_summary(
            event_prompt, call_type="call_4_global_event"
        )
        user_portrait = self._call_llm_for_summary(
            personality_prompt, call_type="call_5_global_personality"
        )
        self.apply_global_summary_results(
            event_summary=event_summary,
            user_portrait=user_portrait,
        )

    def _call_llm_for_summary(self, prompt: str, call_type: str) -> str:
        """Call LLM for summarization with token tracking and call logging.

        Uses no system prompt. Any residual <think> blocks are stripped
        defensively from the returned text.
        Returns empty string on failure; callers are responsible for fallback.
        """
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                system_prompt=None,
                temperature=self.summarize_temperature,
                max_tokens=self.summarize_max_tokens,
                return_usage=True,
            )

            if isinstance(result, dict):
                if "_usage" in result:
                    usage = result["_usage"]
                    self.accumulate_call_tokens(
                        call_type,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )
                else:
                    self.accumulate_call_tokens(call_type, 0, 0)
                text = result.get("content", "")
            else:
                self.accumulate_call_tokens(call_type, 0, 0)
                text = str(result)

            # Defensive: strip any residual <think> blocks (complete or truncated)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
            text = text.strip()

            self.log_summary_call(call_type, prompt, text)

            return text

        except Exception as e:
            logger.error(f"Summarization LLM error: {e}")
            self.accumulate_call_tokens(call_type, 0, 0)

            self.log_summary_call(call_type, prompt, None)

            return ""

    # ------------------------------------------------------------------
    # Summary getters
    # ------------------------------------------------------------------

    def get_event_summary(self) -> str:
        """Return best available event summary (global > daily concatenation)."""
        if self.global_event_summary:
            return self.global_event_summary
        if self.daily_event_summaries:
            return "\n".join(
                f"Day {cid}: {s}"
                for cid, s in sorted(self.daily_event_summaries.items())
            )
        return ""

    def get_user_portrait(self) -> str:
        """Return best available user portrait (global > daily concatenation)."""
        if self.global_user_portrait:
            return self.global_user_portrait
        if self.daily_personality_summaries:
            return "\n".join(
                f"Day {cid}: {s}"
                for cid, s in sorted(self.daily_personality_summaries.items())
            )
        return ""

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_and_reset_token_counts_by_type(self) -> Dict:
        """Return and reset per-call-type summarization token counters."""
        counts = {
            "call_2_daily_event":        dict(self._daily_event_tokens),
            "call_3_daily_personality":  dict(self._daily_person_tokens),
            "call_4_global_event":       dict(self._global_event_tokens),
            "call_5_global_personality": dict(self._global_person_tokens),
        }
        self._daily_event_tokens   = {"input": 0, "output": 0, "llm_calls": 0}
        self._daily_person_tokens  = {"input": 0, "output": 0, "llm_calls": 0}
        self._global_event_tokens  = {"input": 0, "output": 0, "llm_calls": 0}
        self._global_person_tokens = {"input": 0, "output": 0, "llm_calls": 0}
        return counts

    def get_memory_stats(self) -> Dict:
        """Return current memory count and estimated total content tokens."""
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
        """Return and reset Phase 1 internal statistics."""
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
        """Reset all memory state for a new session."""
        self.entries.clear()
        self.retriever.corpus.clear()
        self.retriever.embeddings = None
        self.daily_summary_entries.clear()
        self.daily_summary_retriever.corpus.clear()
        self.daily_summary_retriever.embeddings = None
        self.daily_event_summaries.clear()
        self.daily_personality_summaries.clear()
        self.global_event_summary = ""
        self.global_user_portrait = ""
        self._daily_event_tokens   = {"input": 0, "output": 0, "llm_calls": 0}
        self._daily_person_tokens  = {"input": 0, "output": 0, "llm_calls": 0}
        self._global_event_tokens  = {"input": 0, "output": 0, "llm_calls": 0}
        self._global_person_tokens = {"input": 0, "output": 0, "llm_calls": 0}
        self._forgetting_events = 0
        self._memories_forgotten = 0
        self._llm_logger = None

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
            "daily_event_summaries": {
                str(k): v for k, v in self.daily_event_summaries.items()
            },
            "daily_personality_summaries": {
                str(k): v
                for k, v in self.daily_personality_summaries.items()
            },
            "global_event_summary": self.global_event_summary,
            "global_user_portrait": self.global_user_portrait,
        }
        with open(directory / "summaries.json", "w") as f:
            json.dump(summaries, f, indent=2)

        # Metadata
        metadata = {
            "total_memories": len(self.entries),
            "daily_summaries_count": len(self.daily_event_summaries),
            "has_global_summary": bool(self.global_event_summary),
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved memory snapshot to {directory} "
            f"({len(self.entries)} entries)"
        )

    def load_snapshot(self, directory: Path):
        """Load memory state from disk."""
        directory = Path(directory)

        # Entries
        entries_path = directory / "entries.json"
        if entries_path.exists():
            with open(entries_path) as f:
                entries_data = json.load(f)
            self.entries = [MemoryEntry(**d) for d in entries_data]

        # Embeddings + corpus
        self.retriever.load(directory)

        # Summaries
        summaries_path = directory / "summaries.json"
        if summaries_path.exists():
            with open(summaries_path) as f:
                summaries = json.load(f)
            self.daily_event_summaries = {
                int(k): v
                for k, v in summaries.get(
                    "daily_event_summaries", {}
                ).items()
            }
            self.daily_personality_summaries = {
                int(k): v
                for k, v in summaries.get(
                    "daily_personality_summaries", {}
                ).items()
            }
            self.global_event_summary = summaries.get(
                "global_event_summary", ""
            )
            self.global_user_portrait = summaries.get(
                "global_user_portrait", ""
            )
            self.daily_summary_entries = []
            for conv_id, summary_text in sorted(
                self.daily_event_summaries.items()
            ):
                source_label = self._source_label_for_conv_id(conv_id)
                self.daily_summary_entries.append(
                    DailySummaryEntry(
                        id=str(uuid.uuid4()),
                        content=(
                            f"The summary of the conversation on {source_label} "
                            f"is: {summary_text}"
                        ),
                        conv_id=conv_id,
                        source_label=source_label,
                    )
                )
            self._rebuild_daily_summary_index()

        logger.info(
            f"Loaded memory snapshot from {directory} "
            f"({len(self.entries)} entries)"
        )
