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
    strength: int = 1       # Starts at 1, +1 on recall
    last_recall_conv_id: int = -1   # -1 = never recalled; updated on recall


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    formatted: str                              # Formatted string for prompt
    items: List[Dict] = field(default_factory=list)   # Detailed items for logging
    total_memories: int = 0


# =============================================================================
# MEMORYBANK SYSTEM
# =============================================================================

class MemoryBankSystem:
    """
    Core MemoryBank memory system.

    Stores dialogue turns as embeddings, retrieves via cosine similarity
    weighted by Ebbinghaus forgetting curve, and generates hierarchical
    summaries (daily event + personality → global).
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

        # Memory store
        self.entries: List[MemoryEntry] = []
        self.retriever = EmbeddingRetriever(embedding_model)

        # Hierarchical summaries
        self.daily_event_summaries: Dict[int, str] = {}
        self.daily_personality_summaries: Dict[int, str] = {}
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
        """Store a dialogue turn as a memory entry."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=timestamp,
            conv_id=conv_id,
        )
        self.entries.append(entry)
        self.retriever.add_document(content)

    # ------------------------------------------------------------------
    # Memory retrieval with forgetting re-ranking
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
        if not self.entries:
            return RetrievalResult(formatted="", items=[], total_memories=0)

        k_actual = min(k, len(self.entries))
        raw_results = self.retriever.search_with_scores(query, k_actual)

        if not raw_results:
            return RetrievalResult(
                formatted="", items=[], total_memories=len(self.entries)
            )

        # Update strength for recalled memories
        if update_strength:
            for idx, _ in raw_results:
                self.entries[idx].strength += 1
                self.entries[idx].last_recall_conv_id = current_conv_id

        # Build items for logging
        items = []
        for idx, score in raw_results:
            entry = self.entries[idx]
            source_turn = self._parse_timestamp(entry.timestamp)
            items.append({
                "memory_id": entry.id,
                "content_preview": entry.content[:200],
                "score": round(score, 6),
                "source_turn": source_turn,
                "strength": entry.strength,
            })

        # Format for prompt injection
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

    def summarize_daily(self, conv_id: int, dialogue_text: str):
        """
        Generate daily event and personality summaries for a batch of conv_ids.

        Makes 2 LLM calls: one for event summary, one for personality.
        Fallback: if the LLM call returns empty (e.g. thinking truncation),
        the previous value for that key is kept unchanged.
        """
        # Event summary
        event_prompt = DAILY_EVENT_PROMPT.format(dialogue=dialogue_text)
        event_summary = self._call_llm_for_summary(
            event_prompt, call_type="call_2_daily_event"
        )
        if event_summary:
            self.daily_event_summaries[conv_id] = event_summary
        else:
            logger.warning(
                f"Daily event summary for conv_id {conv_id} was empty; "
                f"keeping previous value."
            )

        # Personality summary
        personality_prompt = DAILY_PERSONALITY_PROMPT.format(
            dialogue=dialogue_text
        )
        personality_summary = self._call_llm_for_summary(
            personality_prompt, call_type="call_3_daily_personality"
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

    def synthesize_global(self):
        """
        Synthesize global summaries from all daily summaries.

        Makes 2 LLM calls: one for global event, one for global personality.
        Call between Phase 1 and Phase 2 so QA can use global summaries.
        """
        if not self.daily_event_summaries:
            logger.info("No daily summaries to synthesize.")
            return

        # Global event summary
        daily_event_text = "\n".join(
            f"Day {cid}: {summary}"
            for cid, summary in sorted(self.daily_event_summaries.items())
        )
        event_prompt = GLOBAL_EVENT_PROMPT.format(
            daily_summaries=daily_event_text
        )
        new_event = self._call_llm_for_summary(
            event_prompt, call_type="call_4_global_event"
        )
        if new_event:
            self.global_event_summary = new_event
        else:
            logger.warning(
                "Global event synthesis returned empty; keeping previous value."
            )

        # Global user portrait
        daily_personality_text = "\n".join(
            f"Day {cid}: {summary}"
            for cid, summary in sorted(
                self.daily_personality_summaries.items()
            )
        )
        personality_prompt = GLOBAL_PERSONALITY_PROMPT.format(
            daily_personalities=daily_personality_text
        )
        new_portrait = self._call_llm_for_summary(
            personality_prompt, call_type="call_5_global_personality"
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
        """Call LLM for summarization with token tracking and call logging.

        Passes '/no_think' as the system prompt to suppress Qwen3 chain-of-thought
        reasoning traces, which would otherwise consume the token budget before
        any actual summary content is generated.
        Returns empty string on failure; callers are responsible for fallback.
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
                    self._summary_input_tokens += usage.get(
                        "prompt_tokens", 0
                    )
                    self._summary_output_tokens += usage.get(
                        "completion_tokens", 0
                    )
                text = result.get("content", "")
            else:
                text = str(result)

            # Defensive: strip any residual <think> blocks (complete or truncated)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
            text = text.strip()

            # Log the call if logger is available
            if self._llm_logger is not None:
                self._llm_logger.log(
                    call_type=call_type,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    output=text,
                )

            return text

        except Exception as e:
            logger.error(f"Summarization LLM error: {e}")
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
        """Reset all memory state for a new session."""
        self.entries.clear()
        self.retriever.corpus.clear()
        self.retriever.embeddings = None
        self.daily_event_summaries.clear()
        self.daily_personality_summaries.clear()
        self.global_event_summary = ""
        self.global_user_portrait = ""
        self._summary_input_tokens = 0
        self._summary_output_tokens = 0
        self._summary_api_calls = 0
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

        logger.info(
            f"Loaded memory snapshot from {directory} "
            f"({len(self.entries)} entries)"
        )
