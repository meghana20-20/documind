"""
qa_engine.py
The core RAG pipeline: retrieve relevant chunks, ask the LLM to answer using
only those chunks, then score how well-grounded each sentence of the answer
actually is in the retrieved source material.

This groundedness scoring step is what makes DocuMind different from a
bare-bones RAG tutorial: instead of blindly trusting the LLM's output, we
independently verify it against the retrieved evidence.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.embed_store import VectorStore
from src.llm_client import LLMClient
from src.ingest import Chunk

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using "
    "ONLY the information in the provided context. If the context does not "
    "contain enough information to answer, say so explicitly rather than "
    "guessing. Keep answers concise and factual."
)

# Sentences with groundedness below this are flagged as poorly supported.
GROUNDEDNESS_THRESHOLD = 0.45


@dataclass
class SentenceScore:
    sentence: str
    similarity: float
    supported: bool


@dataclass
class QAResult:
    answer: str
    sentence_scores: List[SentenceScore]
    overall_confidence: float
    sources: List[Chunk] = field(default_factory=list)


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter -- avoids pulling in nltk/spacy as a dependency.
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _build_context(chunks_with_scores) -> str:
    parts = []
    for chunk, score in chunks_with_scores:
        parts.append(f"[Source: {chunk.source}, chunk {chunk.chunk_id}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def score_groundedness(
    answer: str, source_chunks: List[Chunk], store: VectorStore
) -> tuple[List[SentenceScore], float]:
    """
    For each sentence in the answer, compute its max cosine similarity against
    the retrieved source chunks. This is a proxy for "is this sentence actually
    supported by the retrieved evidence, or did the model make it up?"
    """
    sentences = _split_sentences(answer)
    if not sentences or not source_chunks:
        return [], 0.0

    sentence_vecs = store.embed_sentences(sentences)
    source_vecs = store.embed_sentences([c.text for c in source_chunks])

    # cosine similarity via dot product (vectors are already normalized)
    sim_matrix = sentence_vecs @ source_vecs.T  # shape: (n_sentences, n_sources)
    max_sims = sim_matrix.max(axis=1)

    scores = [
        SentenceScore(
            sentence=sent,
            similarity=float(sim),
            supported=bool(sim >= GROUNDEDNESS_THRESHOLD),
        )
        for sent, sim in zip(sentences, max_sims)
    ]
    overall = float(np.mean(max_sims)) if len(max_sims) else 0.0
    return scores, overall


def answer_question(question: str, store: VectorStore, llm: LLMClient, top_k: int = 4) -> QAResult:
    """Run the full retrieve -> generate -> verify pipeline."""
    retrieved = store.search(question, top_k=top_k)

    if not retrieved:
        return QAResult(
            answer="I don't have any documents to search yet. Please upload some first.",
            sentence_scores=[],
            overall_confidence=0.0,
            sources=[],
        )

    context = _build_context(retrieved)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    raw_answer = llm.generate(SYSTEM_PROMPT, user_prompt)

    source_chunks = [c for c, _ in retrieved]
    sentence_scores, overall_confidence = score_groundedness(raw_answer, source_chunks, store)

    return QAResult(
        answer=raw_answer,
        sentence_scores=sentence_scores,
        overall_confidence=overall_confidence,
        sources=source_chunks,
    )
