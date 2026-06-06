"""
Hybrid retrieval: dense (semantic) + sparse (BM25 keyword), fused.

This is the single most "enterprise-grade" piece of the system. Pure vector
search (just embeddings) is what most student projects do, and it has a
well-known failure: it is great at meaning but bad at exact tokens. Ask about
"CVE-2021-44228" and a dense model may return generically "vulnerability"
-flavored chunks while missing the chunk that literally contains that ID,
because the embedding blurs rare tokens. BM25 (keyword) is the opposite:
great at exact terms, blind to paraphrase. Real systems run BOTH and fuse the
scores. That is "hybrid search," and it is what shops like Elastic, Pinecone,
and Weaviate ship by default in 2026.

The two retrievers:

1. DENSE. Embed every chunk into a 384-dim vector once, up front. At query
   time, embed the query and rank chunks by cosine similarity. "Cosine
   similarity" = the cosine of the angle between two vectors; 1.0 means same
   direction (same meaning), 0 means unrelated. We normalize vectors to unit
   length so a plain dot product *is* the cosine, which is fast.

2. SPARSE (BM25). A classic information-retrieval formula that scores a chunk
   by how many query words it contains, weighted so that rare words count
   more (saying "Kerberoasting" is more informative than saying "the") and so
   that very long chunks do not win just by being long. We implement BM25
   directly, no black box, so she can explain every term.

FUSION. The two score lists are on different scales, so we cannot just add
them. We min-max normalize each to 0..1, then combine:
      score = alpha * dense_norm + (1 - alpha) * sparse_norm
alpha lives in config and is a great thing to tune in the eval harness.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .documents import Chunk


@dataclass
class Retrieved:
    """A chunk plus the scores that got it here. Keeping the component scores
    (not just the final) is what lets the UI explain *why* something ranked,
    and lets the eval harness diagnose whether dense or sparse is carrying."""
    chunk: Chunk
    score: float
    dense_score: float
    sparse_score: float


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer for BM25. We keep digits because in
    security text the identifiers (CVE numbers, port numbers) are exactly the
    high-value rare tokens BM25 is good at."""
    return _TOKEN.findall(text.lower())


class BM25:
    """Minimal, correct BM25 Okapi implementation.

    Parameters k1 and b are the standard BM25 knobs:
      k1 controls how fast term-frequency saturates (more occurrences help,
         but with diminishing returns).
      b  controls length normalization (how much to penalize long chunks).
    The values 1.5 and 0.75 are the textbook defaults used everywhere.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(toks) for toks in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        # Document frequency: in how many chunks does each term appear?
        df: dict[str, int] = {}
        for toks in corpus_tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1

        # Inverse document frequency (the "rare words matter more" weight).
        # This is the BM25 idf variant; the +0.5 smoothing avoids div-by-zero
        # and gives very common terms a near-zero (even slightly negative)
        # weight, which is what we want.
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

        # Precompute term frequencies per chunk for speed.
        self.tf: list[dict[str, int]] = []
        for toks in corpus_tokens:
            d: dict[str, int] = {}
            for t in toks:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)

    def scores(self, query: str) -> np.ndarray:
        q_terms = _tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)
        for term in q_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale an array to 0..1. Needed because dense cosine scores (~0..1) and
    BM25 scores (unbounded, can be 8+) live on different scales and cannot be
    blended until both are normalized. If everything ties, return zeros."""
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    """Builds both indexes once, then answers queries by fusing them."""

    def __init__(self, chunks: list[Chunk], embed_model_name: str,
                 hybrid_alpha: float, top_k_dense: int, top_k_sparse: int):
        self.chunks = chunks
        self.alpha = hybrid_alpha
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse

        # --- Build the dense index ---
        # Load the embedding model (downloads once, then cached locally).
        self.embedder = SentenceTransformer(embed_model_name)
        texts = [c.text for c in chunks]
        # normalize_embeddings=True gives unit vectors so dot == cosine.
        self.embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        # --- Build the sparse index ---
        self.bm25 = BM25([_tokenize(t) for t in texts])

    def retrieve(self, query: str) -> list[Retrieved]:
        # Dense: cosine similarity is just the matrix-vector dot product
        # because everything is unit-normalized.
        q_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        dense_raw = self.embeddings @ q_vec               # shape (num_chunks,)

        # Sparse: BM25 scores over the same chunks.
        sparse_raw = self.bm25.scores(query)

        # Normalize each to 0..1 so the weighted blend is meaningful.
        dense_n = _minmax(dense_raw)
        sparse_n = _minmax(sparse_raw)

        fused = self.alpha * dense_n + (1 - self.alpha) * sparse_n

        # Rank all chunks by fused score, return them with component scores
        # attached so callers can inspect/explain the ranking.
        order = np.argsort(-fused)
        results: list[Retrieved] = []
        for i in order:
            results.append(Retrieved(
                chunk=self.chunks[i],
                score=float(fused[i]),
                dense_score=float(dense_n[i]),
                sparse_score=float(sparse_n[i]),
            ))
        return results
