"""
The pipeline: the controller that wires everything into an agent.

This is the file to read last and understand best, because it is the "brain"
that decides what happens when. Everything else (retriever, grader, generator)
is a capability; this is the policy that USES those capabilities in a loop.

The agentic control flow, in plain English:

    1. Retrieve chunks for the query (hybrid search).
    2. Ask the grader: is this enough to answer?
    3. If YES  -> generate the cited answer and stop.
       If NO   -> rewrite the query into something better and go to step 1.
    4. Never loop more than max_retrieval_attempts times; if we exhaust them,
       answer with the best context we have AND flag that confidence is low.

Why this is "agentic" and not just a fancy if-statement: the system is making
its own decisions about whether its work is good enough and taking corrective
action (rewriting + retrying) without a human in the loop. That perceive ->
judge -> act -> repeat cycle is the definition of an agent. We keep it bounded
(the attempt cap) because an unbounded agent is how you get runaway API bills
and infinite loops. Bounded autonomy is the responsible-engineering version.

We also record a full TRACE of every step. The trace is not decoration: it is
what makes the system debuggable and what makes the UI able to show "here is
how I thought about this." Observability is a first-class production concern,
not an afterthought.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .config import Config
from .documents import load_corpus
from .retriever import HybridRetriever, Retrieved
from .grader import grade_context, rewrite_query
from .generator import generate_answer, Answer
from .llm import make_llm, LLMClient


@dataclass
class Step:
    """One iteration of the loop, captured for the trace/UI."""
    attempt: int
    query: str
    top_sources: list[str]
    sufficient: bool
    grade_reason: str


@dataclass
class PipelineResult:
    answer: Answer
    steps: list[Step] = field(default_factory=list)
    attempts_used: int = 0
    low_confidence: bool = False


# Map the config corpus name to its folder. Adding a new corpus = add a line.
_CORPUS_FOLDERS = {
    "security": "data/security",
    "secondary": "data/secondary",
}


class RAGPipeline:
    """Build once (loads + indexes the corpus), then answer many queries.

    Two ways to build:
    - Default: loads documents from the disk corpus named in config.corpus.
    - With `chunks=`: indexes a list of chunks you pass in directly. This is how
      the "upload your own document" feature works, the uploaded file is turned
      into chunks in memory and handed in here, never touching the disk corpus.
    """

    def __init__(self, config: Config, llm: LLMClient | None = None,
                 grader_llm: LLMClient | None = None,
                 chunks: list | None = None):
        self.cfg = config
        # Two LLM handles: a strong one for answers, a cheap one for grading.
        self.llm = llm or make_llm(config.gen_model_name)
        self.grader_llm = grader_llm or make_llm(config.grader_model_name)

        if chunks is None:
            folder = _CORPUS_FOLDERS[config.corpus]
            chunks = load_corpus(folder, config.chunk_size, config.chunk_overlap)
            if not chunks:
                raise RuntimeError(f"No documents found in {folder}. Add .txt/.md files.")
        self.retriever = HybridRetriever(
            chunks=chunks,
            embed_model_name=config.embed_model_name,
            hybrid_alpha=config.hybrid_alpha,
            top_k_dense=config.top_k_dense,
            top_k_sparse=config.top_k_sparse,
        )

    def answer(self, question: str) -> PipelineResult:
        cfg = self.cfg
        steps: list[Step] = []
        query = question
        contexts: list[Retrieved] = []

        for attempt in range(1, cfg.max_retrieval_attempts + 1):
            # Step 1: retrieve, keep the top-k final for the LLM.
            ranked = self.retriever.retrieve(query)
            contexts = ranked[: cfg.top_k_final]

            # Step 2: grade sufficiency.
            grade = grade_context(self.grader_llm, question, contexts)

            steps.append(Step(
                attempt=attempt,
                query=query,
                top_sources=[r.chunk.source for r in contexts],
                sufficient=grade.sufficient,
                grade_reason=grade.reason,
            ))

            # Step 3: decide.
            if grade.sufficient:
                ans = generate_answer(
                    self.llm, question, contexts,
                    temperature=cfg.temperature, max_tokens=cfg.max_output_tokens,
                )
                return PipelineResult(answer=ans, steps=steps,
                                      attempts_used=attempt, low_confidence=False)

            # Not sufficient and attempts remain -> rewrite and loop.
            if attempt < cfg.max_retrieval_attempts:
                query = rewrite_query(self.grader_llm, question, query, grade.reason)

        # Step 4: exhausted attempts. Answer with best available context but
        # flag low confidence so the UI can warn the user honestly.
        ans = generate_answer(
            self.llm, question, contexts,
            temperature=cfg.temperature, max_tokens=cfg.max_output_tokens,
        )
        return PipelineResult(answer=ans, steps=steps,
                              attempts_used=cfg.max_retrieval_attempts,
                              low_confidence=True)
