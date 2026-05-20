"""
Dense Memory Store for the Dense Retrieval Baseline

Stores conversation turn pairs as dense embeddings and retrieves them by
cosine similarity. No LLM calls are made during Phase 1.

Design:
    - MemoryUnit: stores one (user, assistant) turn pair with its embedding.
    - DenseMemoryStore: manages the list of MemoryUnit objects for one session.
      The SentenceTransformer model is shared across all sessions in a batch
      and passed in at construction time (not loaded per-session).
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class MemoryUnit:
    """One stored memory: a (user, assistant) pair with its dense embedding."""
    content: str          # "User: {utt}\nAssistant: {utt}" (or "User: {utt}" if no assistant)
    embedding: np.ndarray # shape (dim,)
    session_id: int
    conv_id: int
    turn_id: int          # user turn's turn_id


# =============================================================================
# STORE
# =============================================================================

class DenseMemoryStore:
    """
    Per-session dense vector memory store.

    The SentenceTransformer model is shared across all sessions in a batch.
    Embeddings are computed externally (in batch) and passed into store().
    """

    def __init__(self, embedding_model, k: int):
        """
        Args:
            embedding_model: A SentenceTransformer instance (shared, not owned).
            k: Default number of top memories to retrieve.
        """
        self._model = embedding_model
        self._k = k
        self._memories: List[MemoryUnit] = []

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        embedding: np.ndarray,
        session_id: int,
        conv_id: int,
        turn_id: int,
    ) -> None:
        """Add a pre-computed embedding + content to memory.

        The embedding is passed in (already computed in batch externally).
        """
        self._memories.append(
            MemoryUnit(
                content=content,
                embedding=embedding,
                session_id=session_id,
                conv_id=conv_id,
                turn_id=turn_id,
            )
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Cosine similarity retrieval. Returns top-k (MemoryUnit, score) pairs.

        Returns an empty list if the store is empty.

        Args:
            query_embedding: Shape (dim,) — a single query vector.
            k: Number of results to return. Defaults to self._k.

        Returns:
            List of (MemoryUnit, cosine_score) sorted descending by score,
            length min(k, num_memories).
        """
        if not self._memories:
            return []

        effective_k = k if k is not None else self._k

        # Stack all stored embeddings: shape (N, dim)
        stored_embeddings = np.stack([m.embedding for m in self._memories], axis=0)

        # query_embedding shape: (dim,) → reshape to (1, dim)
        query_2d = query_embedding.reshape(1, -1)

        # Cosine similarity: shape (1, N)
        scores = cosine_similarity(query_2d, stored_embeddings)[0]  # (N,)

        # Get top-k indices (sorted descending)
        top_k = min(effective_k, len(self._memories))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self._memories[idx], float(scores[idx])) for idx in top_indices]

    # ------------------------------------------------------------------
    # Stats / maintenance
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict:
        return {
            "num_memories": len(self._memories),
            "total_content_tokens": sum(len(m.content) // 4 for m in self._memories),
        }

    def clear(self) -> None:
        self._memories = []

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def save_snapshot(self, path: Path) -> None:
        """Save memory as JSON to path/memories.json.

        Each entry: {content, embedding (list), session_id, conv_id, turn_id}.
        """
        path.mkdir(parents=True, exist_ok=True)
        snapshot = [
            {
                "content":    m.content,
                "embedding":  m.embedding.tolist(),
                "session_id": m.session_id,
                "conv_id":    m.conv_id,
                "turn_id":    m.turn_id,
            }
            for m in self._memories
        ]
        with open(path / "memories.json", "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
