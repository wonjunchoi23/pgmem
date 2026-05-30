"""Rolling context cache for PGMem."""

from collections import deque
from pathlib import Path
import json
from typing import List, Tuple

from graph_store import format_elapsed_str


class ContextCache:
    """Stores the most recent k0 (user, gt_response) pairs with turn metadata."""

    def __init__(self, k0: int) -> None:
        if k0 <= 0:
            raise ValueError(f"k0 must be positive, got {k0}")
        self._k0 = k0
        self._pairs = deque(maxlen=k0)

    def add_turn(
        self,
        user_utterance: str,
        gt_response: str,
        conv_id: int,
        turn_id: int,
    ) -> None:
        self._pairs.append((user_utterance, gt_response, conv_id, turn_id))

    def get_formatted_context(
        self,
        current_conv_id: int,
        current_turn_id: int,
        cfg,
        max_pairs: int = None,
    ) -> str:
        if not self._pairs:
            return ""
        pairs = list(self._pairs)
        if max_pairs is not None and max_pairs > 0:
            pairs = pairs[-max_pairs:]
        lines: List[str] = []
        for user, agent, conv_id, turn_id in pairs:
            elapsed = format_elapsed_str(
                conv_id, turn_id,
                current_conv_id, current_turn_id,
                cfg.TIME_PER_CONV_ID_HOURS,
                cfg.TIME_PER_TURN_MINUTES,
            )
            lines.append(f"[{elapsed}] User: {user}")
            lines.append(f"[{elapsed}] Agent: {agent}")
        return "\n".join(lines)

    def get_pairs(self) -> List[Tuple[str, str, int, int]]:
        return list(self._pairs)

    def get_prior_pairs(
        self,
        current_conv_id: int,
        current_turn_id: int,
        n: int,
    ) -> List[Tuple[str, str, int, int]]:
        """Up to n most-recent (user, agent, conv_id, turn_id) pairs that are
        NOT the current (conv_id, turn_id). Used by ② state extraction for the
        PRIOR CONTEXT block. conv_id boundaries are ignored."""
        if n <= 0:
            return []
        out = [
            (u, a, c, t) for (u, a, c, t) in self._pairs
            if (c, t) != (current_conv_id, current_turn_id)
        ]
        return out[-n:]

    def clear(self) -> None:
        self._pairs.clear()

    def save_snapshot(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "context_cache.json", "w", encoding="utf-8") as f:
            json.dump(list(self._pairs), f, ensure_ascii=True, indent=2)

    def load_snapshot(self, directory: Path) -> None:
        path = Path(directory) / "context_cache.json"
        self.clear()
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                self._pairs.append(tuple(item))

    def __len__(self) -> int:
        return len(self._pairs)


