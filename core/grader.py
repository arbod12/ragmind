"""
The grader: the "agentic" judgment step.

This is what turns a plain RAG pipeline into an AGENT. A plain pipeline does:
retrieve -> stuff into prompt -> answer, and hopes the retrieval was good. An
agent adds a decision point: after retrieving, it asks an LLM to JUDGE whether
the retrieved context actually contains enough to answer. If yes, proceed. If
no, the controller (in pipeline.py) rewrites the query and retries. That
judge is this file.

Two separate LLM jobs live here:

1. grade_context(): a binary "is this enough to answer?" check. We force the
   model to return strict JSON so we get a machine-readable decision, not
   prose we have to parse with regex. Asking for JSON and parsing it is the
   robust pattern; "structured outputs" is a named skill enterprises want.

2. rewrite_query(): when grading fails, produce a better search query. The
   model often knows synonyms or more specific terms than the user typed
   ("my password got stolen" -> "credential theft mitigation"). This is the
   self-correction that makes the loop actually improve, not just retry the
   same failing search.

Why a cheap model for grading: grading is an easy classification task, so we
use the fast/cheap model (config.grader_model_name). You reserve the better
model for the final answer. Routing easy work to cheap models and hard work
to expensive ones is real cost-engineering that production teams care about.
"""

from __future__ import annotations
from dataclasses import dataclass
import json

from .llm import LLMClient
from .retriever import Retrieved


@dataclass
class Grade:
    sufficient: bool
    reason: str


_GRADER_SYSTEM = """You are a strict retrieval grader for a question-answering system.
You will be given a user QUESTION and a set of retrieved CONTEXT passages.
Your only job is to decide whether the CONTEXT contains enough information to
answer the QUESTION correctly and completely.

Be strict. If the context is only loosely related, or covers the topic but
not the specific thing asked, that is NOT sufficient. It is much better to
say insufficient and trigger another search than to let the system answer
from thin context and hallucinate.

Respond with ONLY a JSON object, no other text, in exactly this form:
{"sufficient": true or false, "reason": "one short sentence"}"""


def _extract_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in ```json fences or add a stray word. We pull
    out the first {...} block and parse it. Defensive parsing like this is the
    difference between a demo that breaks on the 20th query and one that does
    not."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in grader output: {text[:120]}")
    return json.loads(text[start:end + 1])


def grade_context(llm: LLMClient, question: str, contexts: list[Retrieved],
                  grader_model_temp: float = 0.0) -> Grade:
    """Ask the LLM whether the retrieved context can answer the question."""
    joined = "\n\n".join(
        f"[Passage {i+1} from {r.chunk.source}]\n{r.chunk.text}"
        for i, r in enumerate(contexts)
    )
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{joined}"
    raw = llm.complete(_GRADER_SYSTEM, user, temperature=grader_model_temp, max_tokens=200)
    try:
        obj = _extract_json(raw)
        return Grade(bool(obj.get("sufficient", False)),
                     str(obj.get("reason", "")).strip() or "(no reason given)")
    except Exception as e:
        # If grading itself fails, fail SAFE by treating context as sufficient
        # so the user still gets an answer rather than an error. We record the
        # reason so the failure is visible in the trace, not hidden.
        return Grade(True, f"grader-parse-failed-defaulting-to-proceed: {e}")


_REWRITE_SYSTEM = """You rewrite a user's question into a better search query for a
document retrieval system. The previous search did not find sufficient
information, so produce a DIFFERENT query that might retrieve better passages:
use more specific terminology, expand abbreviations, or add likely synonyms.

Respond with ONLY the rewritten query as a single line of plain text. No
quotes, no explanation, no formatting."""


def rewrite_query(llm: LLMClient, original_question: str,
                  previous_query: str, fail_reason: str) -> str:
    """Generate an improved query after a failed retrieval attempt."""
    user = (
        f"Original question: {original_question}\n"
        f"Previous search query that failed: {previous_query}\n"
        f"Why it was insufficient: {fail_reason}\n\n"
        f"Write one improved search query."
    )
    out = llm.complete(_REWRITE_SYSTEM, user, temperature=0.3, max_tokens=80)
    # Take the first non-empty line, strip stray quotes the model may add.
    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), original_question)
    return line.strip().strip('"').strip("'")
