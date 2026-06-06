"""
Central configuration for the RAG engine.

Everything tunable lives here so the rest of the code never hard-codes a
magic number. In an interview you can point at this file and say "every
knob the system has is in one place, documented, with a reason." That is
exactly the kind of thing reviewers look for.

Why a dataclass instead of loose globals: it makes the config an *object*
you can copy, tweak, and pass into an evaluation run. That is how you test
"v1 vs v2" honestly. You build two Config objects and compare. If the
settings were scattered module-level globals you could not do that cleanly.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class Config:
    # ---- Embedding / retrieval ----
    # The embedding model turns text into vectors. all-MiniLM-L6-v2 is small
    # (80MB), fast, runs locally with no API call, and is the standard
    # baseline in industry for prototypes. 384-dimensional output.
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Chunking: how we cut documents into retrievable pieces. This is one of
    # the highest-leverage decisions in all of RAG. Too big -> the model gets
    # noise around the answer. Too small -> the answer gets split across
    # chunks and retrieval misses it. We chunk by characters with overlap so
    # a sentence spanning a boundary still appears whole in one chunk.
    chunk_size: int = 900          # characters per chunk
    chunk_overlap: int = 150       # characters shared between neighbors

    # ---- Hybrid search weighting ----
    # We blend two retrievers: dense (semantic, embedding similarity) and
    # sparse (BM25, keyword overlap). alpha = weight on the dense score.
    # 0.0 = pure keyword, 1.0 = pure semantic. 0.5 is a balanced default;
    # enterprises tune this per corpus. We expose it because tuning it is a
    # great evaluation experiment.
    hybrid_alpha: float = 0.5
    top_k_dense: int = 10          # candidates pulled from each retriever
    top_k_sparse: int = 10
    top_k_final: int = 4           # how many chunks we actually feed the LLM

    # ---- Agentic self-correction loop ----
    # If the grader judges the retrieved context insufficient, we rewrite the
    # query and retry. This caps how many times, so a hard question can never
    # spin forever (and never run up an unbounded API bill).
    max_retrieval_attempts: int = 3

    # ---- Generation ----
    gen_model_name: str = "gemini-2.5-flash"
    grader_model_name: str = "gemini-2.5-flash"  # cheap model is fine for grading
    temperature: float = 0.2       # low: factual, grounded answers, not creative
    max_output_tokens: int = 800

    # ---- Corpus selection ----
    # Which document set to load. The whole point of "swappable corpus" lives
    # here: change this one string and the system answers from a different
    # body of knowledge with no other code change.
    corpus: str = "security"       # "security" or "secondary"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_env() -> "Config":
        """Build a config, letting environment variables override defaults.
        This is how the same code runs locally and in deployment without
        editing source. Twelve-factor app style."""
        c = Config()
        if os.getenv("RAG_CORPUS"):
            c.corpus = os.environ["RAG_CORPUS"]
        if os.getenv("RAG_HYBRID_ALPHA"):
            c.hybrid_alpha = float(os.environ["RAG_HYBRID_ALPHA"])
        return c
