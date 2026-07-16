"""
embed_store.py
Wraps sentence-transformers embeddings + a FAISS index so we can add
document chunks and run similarity search over them.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.ingest import Chunk

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Chunk] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.astype("float32")

    def build(self, chunks: List[Chunk]) -> None:
        """Build a fresh FAISS index from a list of chunks."""
        self.chunks = chunks
        vectors = self._embed([c.text for c in chunks])
        dim = vectors.shape[1]
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def add(self, chunks: List[Chunk]) -> None:
        """Add more chunks to an existing index (or create one if empty)."""
        if self.index is None:
            self.build(chunks)
            return
        vectors = self._embed([c.text for c in chunks])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """Return the top_k most similar chunks to the query with their scores."""
        if self.index is None or len(self.chunks) == 0:
            return []
        query_vec = self._embed([query])
        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Expose raw embedding for use in groundedness scoring."""
        return self._embed(sentences)
