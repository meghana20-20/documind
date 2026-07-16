"""
ingest.py
Loads documents (PDF, DOCX, TXT) and splits them into overlapping chunks
suitable for embedding + retrieval.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
import docx


@dataclass
class Chunk:
    """A single retrievable unit of text with metadata."""
    text: str
    source: str          # filename
    chunk_id: int        # position within the document


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def _read_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path: str) -> str:
    """Dispatch to the right reader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    elif ext == ".docx":
        return _read_docx(path)
    elif ext == ".txt":
        return _read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    """Collapse excessive whitespace/newlines produced by PDF extraction."""
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    chunk_size and overlap are measured in characters, which is a reasonable
    proxy for token count without pulling in a tokenizer dependency.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # try to break on a sentence boundary near the end for cleaner chunks
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + chunk_size // 2:
            end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c]


def ingest_documents(file_paths: List[str], chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    """
    Load and chunk a list of document file paths.
    Returns a flat list of Chunk objects ready for embedding.
    """
    all_chunks: List[Chunk] = []
    for path in file_paths:
        raw = load_document(path)
        cleaned = clean_text(raw)
        pieces = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        filename = os.path.basename(path)
        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(text=piece, source=filename, chunk_id=i))
    return all_chunks
