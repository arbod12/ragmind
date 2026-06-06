"""
Document loading + chunking.

This is the "structure messy data" part of the pipeline. Real corpora are a
pile of text files of wildly different lengths. The retriever cannot work on
whole documents (too coarse) so we cut them into overlapping chunks and
attach metadata to each one. The metadata (source file, chunk index) is what
later lets us show CITATIONS: when the model answers, we can say exactly
which document and which part it came from.

Key idea to understand for interviews:
A "chunk" is the atomic unit of retrieval. Everything downstream, embeddings,
BM25, grading, citations, operates on chunks, never on raw files. Get
chunking wrong and no amount of clever modeling downstream saves you. This is
why chunk_size/overlap live in config and are something you tune and measure.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class Chunk:
    """One retrievable piece of a document, plus where it came from.

    id:        stable identifier, e.g. "nist_csf.txt::3"
    text:      the chunk content
    source:    filename it came from (shown in citations)
    chunk_idx: position within that file (shown in citations)
    """
    id: str
    text: str
    source: str
    chunk_idx: int


def _clean(text: str) -> str:
    """Collapse runs of whitespace so chunk sizes are about *content*, not
    formatting. Without this, a document full of blank lines produces chunks
    that are mostly empty space and the size limits become meaningless."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split one document into overlapping character windows.

    Why overlap matters, concretely: imagine the sentence that answers the
    question sits right at character 900, the boundary. Without overlap it
    gets sliced in half and neither chunk contains the whole fact, so
    retrieval can match it but the model sees a fragment. Overlap means the
    last `overlap` characters of chunk N are repeated at the start of chunk
    N+1, so any sentence shorter than `overlap` always appears intact in at
    least one chunk.

    We try to break on paragraph/sentence boundaries near the target size
    rather than mid-word, which keeps chunks human-readable (important when
    you show them as citations).
    """
    text = _clean(text)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # If we are not at the very end, try to back up to a clean boundary
        # (paragraph break, then sentence end, then space) so we do not cut a
        # word in half. We only search the last ~200 chars of the window.
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                pos = window.rfind(sep)
                if pos != -1 and pos > chunk_size - 250:
                    end = start + pos + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                id=f"{source}::{idx}",
                text=piece,
                source=source,
                chunk_idx=idx,
            ))
            idx += 1

        if end >= n:
            break
        # Step forward, but leave `overlap` characters behind us.
        start = max(end - overlap, start + 1)

    return chunks


def load_corpus(folder: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Read every .txt and .md file in a folder and return all chunks.

    Returning a flat list of chunks (rather than nested per-file) is
    deliberate: the retrievers want one big pool to rank. The Chunk.source
    field preserves which file each came from, so we lose nothing.
    """
    base = Path(folder)
    all_chunks: list[Chunk] = []
    for path in sorted(base.glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(chunk_text(text, path.name, chunk_size, overlap))
    return all_chunks
