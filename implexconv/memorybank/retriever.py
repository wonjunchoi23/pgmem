"""
EmbeddingRetriever

Sentence-transformer based embedding store with cosine similarity search.
Extracted from memory_bank.py to follow the original MemoryBank module
separation pattern (analogous to local_doc_qa.py in the original code).
"""

import json
import numpy as np
from typing import Any, List, Optional, Tuple
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class EmbeddingRetriever:
    """Sentence-transformer based embedding retriever with cosine similarity."""

    def __init__(self, model_name_or_instance: Any = "all-MiniLM-L6-v2"):
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers required. pip install sentence-transformers"
            )
        if isinstance(model_name_or_instance, str):
            self.model = SentenceTransformer(model_name_or_instance)
            self.model_name = model_name_or_instance
        else:
            self.model = model_name_or_instance
            self.model_name = getattr(
                model_name_or_instance, "model_name_or_path", "shared"
            )
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_document(self, text: str):
        """Embed and append a document to the corpus."""
        embedding = self.model.encode([text], show_progress_bar=False)
        if self.embeddings is None:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.corpus.append(text)

    def search_with_scores(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Return top-k (index, cosine_similarity) pairs, descending by score."""
        if self.embeddings is None or len(self.corpus) == 0:
            return []

        query_emb = self.model.encode([query], show_progress_bar=False)  # (1, dim)

        # Cosine similarity: dot / (||a|| * ||b||)
        dot = (self.embeddings @ query_emb.T).squeeze()       # (N,)
        norms_c = np.linalg.norm(self.embeddings, axis=1)     # (N,)
        norm_q = np.linalg.norm(query_emb)
        denom = np.maximum(norms_c * norm_q, 1e-10)
        similarities = dot / denom                              # (N,)

        k = min(k, len(self.corpus))
        if k <= 0:
            return []

        top_indices = np.argsort(similarities)[-k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def remove_by_indices(self, indices: List[int]):
        """Remove entries at the given indices from corpus and embeddings.

        Indices must correspond to the same positions in self.corpus /
        self.embeddings rows as in MemoryBankSystem.entries — they are always
        kept in sync.
        """
        if not indices or self.embeddings is None:
            return
        keep_mask = np.ones(len(self.corpus), dtype=bool)
        for idx in indices:
            keep_mask[idx] = False
        self.corpus = [c for i, c in enumerate(self.corpus) if keep_mask[i]]
        self.embeddings = self.embeddings[keep_mask]
        if len(self.corpus) == 0:
            self.embeddings = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(
            directory / "embeddings.npy",
            self.embeddings if self.embeddings is not None else np.array([]),
        )
        with open(directory / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False)

    def load(self, directory: Path):
        directory = Path(directory)
        emb_path = directory / "embeddings.npy"
        corpus_path = directory / "corpus.json"
        if emb_path.exists():
            loaded = np.load(emb_path)
            self.embeddings = loaded if loaded.size > 0 else None
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
