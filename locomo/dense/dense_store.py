"""
Dense Memory Store for the Dense Retrieval Baseline on LoCoMo.

Each stored memory corresponds to a single dialogue turn with speaker prefix,
for example:

    Speaker Caroline says: Hey Mel! Good to see you!
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryUnit:
    content: str
    embedding: np.ndarray
    sample_id: str
    session_id: int
    dia_id: str
    speaker: str
    date_time: str


class DenseMemoryStore:
    """
    Per-sample dense vector memory store.

    Embeddings are computed externally in batch and passed into `store()`.
    """

    def __init__(self, k: int):
        self._k = k
        self._memories: List[MemoryUnit] = []

    def __len__(self) -> int:
        return len(self._memories)

    def store(
        self,
        content: str,
        embedding: np.ndarray,
        sample_id: str,
        session_id: int,
        dia_id: str,
        speaker: str,
        date_time: str,
    ) -> None:
        self._memories.append(
            MemoryUnit(
                content=content,
                embedding=embedding,
                sample_id=sample_id,
                session_id=session_id,
                dia_id=dia_id,
                speaker=speaker,
                date_time=date_time,
            )
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        if not self._memories:
            return []

        effective_k = k if k is not None else self._k
        stored_embeddings = np.stack([memory.embedding for memory in self._memories], axis=0)
        scores = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings)[0]

        top_k = min(effective_k, len(self._memories))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._memories[idx], float(scores[idx])) for idx in top_indices]

    def get_memory_stats(self) -> Dict:
        return {
            "num_memories": len(self._memories),
            "total_content_tokens": sum(len(memory.content) // 4 for memory in self._memories),
        }

    def clear(self) -> None:
        self._memories = []

    def save_snapshot(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        snapshot = [
            {
                "content": memory.content,
                "embedding": memory.embedding.tolist(),
                "sample_id": memory.sample_id,
                "session_id": memory.session_id,
                "dia_id": memory.dia_id,
                "speaker": memory.speaker,
                "date_time": memory.date_time,
            }
            for memory in self._memories
        ]
        with open(path / "memories.json", "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=True)
