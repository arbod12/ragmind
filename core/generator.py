"""
The generator: produce a grounded, CITED answer.

This is the final step, and the citation discipline here is what makes the
system trustworthy enough for a regulated company (finance, healthcare,
government, exactly the US corporate buyers you mentioned). The rule we
enforce through the prompt:

  - Answer ONLY from the provided context.
  - Cite each claim with the passage number it came from, like [1], [2].
  - If the context does not contain the answer, SAY SO rather than guessing.

That last rule is the anti-hallucination contract. A model told "answer from
this context and admit when it cannot" hallucinates far less than one asked
the bare question. We then verify the citations actually point at real
passages (a model can invent a [5] when only 4 were given), and we surface
the cited sources to the UI so a human can click and check. "Trust but
verify" baked into code.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

from .llm import LLMClient
from .retriever import Retrieved


@dataclass
class Answer:
    text: str
    cited_sources: list[str]          # human-readable source labels actually cited
    used_contexts: list[Retrieved]    # the passages we fed the model


_GEN_SYSTEM = """You are a precise question-answering assistant for a knowledge base.
Follow these rules without exception:

1. Answer the user's question using ONLY the information in the provided
   numbered CONTEXT passages. Do not use outside knowledge.
2. After each sentence or claim, cite the passage it came from using square
   brackets with the passage number, like [1] or [2][3]. Every factual claim
   must have a citation.
3. If the context does NOT contain enough information to answer, reply
   exactly: "I don't have enough information in the provided sources to answer
   that." Do not guess or fill gaps with general knowledge.
4. Be concise and factual. Write in plain text. Do not use markdown formatting,
   asterisks, or headers.
"""


def _verify_citations(text: str, n_passages: int) -> list[int]:
    """Find every [k] in the answer and keep only those that point at a real
    passage (1..n_passages). Returns the sorted unique valid passage numbers.
    This catches the model citing a passage that does not exist, a subtle but
    real failure mode you can only catch by checking."""
    nums = set()
    for m in re.finditer(r"\[(\d+)\]", text):
        k = int(m.group(1))
        if 1 <= k <= n_passages:
            nums.add(k)
    return sorted(nums)


def generate_answer(llm: LLMClient, question: str, contexts: list[Retrieved],
                    temperature: float, max_tokens: int) -> Answer:
    numbered = "\n\n".join(
        f"[{i+1}] (source: {r.chunk.source})\n{r.chunk.text}"
        for i, r in enumerate(contexts)
    )
    user = f"CONTEXT PASSAGES:\n{numbered}\n\nQUESTION:\n{question}"
    text = llm.complete(_GEN_SYSTEM, user, temperature=temperature, max_tokens=max_tokens)

    valid = _verify_citations(text, len(contexts))
    cited_sources = []
    seen = set()
    for k in valid:
        src = contexts[k - 1].chunk.source
        if src not in seen:
            cited_sources.append(src)
            seen.add(src)

    return Answer(text=text, cited_sources=cited_sources, used_contexts=contexts)
