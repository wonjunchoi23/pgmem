"""
Rolling context cache for GraphMem v5 on LoComo.
"""

from collections import deque
from pathlib import Path
import json
from typing import List, Tuple

from graph_store import format_elapsed_str


class ContextCache:
    """Stores the most recent k0 individual turns with timestamp metadata."""

    def __init__(self, k0: int) -> None:
        if k0 <= 0:
            raise ValueError(f"k0 must be positive, got {k0}")
        self._k0 = k0
        # each entry: (speaker, text, session_id, turn_idx, timestamp_seconds)
        self._turns = deque(maxlen=k0)

    def add_turn(
        self,
        speaker: str,
        text: str,
        session_id: int,
        turn_idx: int,
        timestamp_seconds: float,
    ) -> None:
        self._turns.append((speaker, text, session_id, turn_idx, timestamp_seconds))

    def get_formatted_context(
        self,
        current_timestamp_seconds: float,
        max_turns: int = None,
    ) -> str:
        if not self._turns:
            return ""
        turns = list(self._turns)
        if max_turns is not None and max_turns > 0:
            turns = turns[-max_turns:]
        lines: List[str] = []
        for speaker, text, _sid, _tidx, ts in turns:
            elapsed = format_elapsed_str(ts, current_timestamp_seconds)
            lines.append(f"[{elapsed}] {speaker} says: {text}")
        return "\n".join(lines)

    def get_turns(self) -> List[Tuple]:
        return list(self._turns)

    def get_prior_turns(
        self,
        current_session_id: int,
        current_turn_idx: int,
        n: int,
    ) -> List[Tuple[str, str, int, int, float]]:
        """Up to n most-recent prior turns, excluding the current
        (session_id, turn_idx). Used by ② state extraction (STATE_REF_CONTEXT_TURNS).
        Ignores session boundaries: the most recent n prior turns are returned
        regardless of whether they fall under a different session."""
        if n <= 0:
            return []
        out = [
            (sp, tx, sid, tidx, ts) for (sp, tx, sid, tidx, ts) in self._turns
            if (sid, tidx) != (current_session_id, current_turn_idx)
        ]
        return out[-n:]

    def clear(self) -> None:
        self._turns.clear()

    def save_snapshot(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "context_cache.json", "w", encoding="utf-8") as f:
            json.dump(list(self._turns), f, ensure_ascii=True, indent=2)

    def load_snapshot(self, directory: Path) -> None:
        path = Path(directory) / "context_cache.json"
        self.clear()
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                self._turns.append(tuple(item))

    def __len__(self) -> int:
        return len(self._turns)
