"""
The evaluation harness: the part almost no student portfolio has.

Anyone can build a RAG demo. The thing that makes a hiring manager stop
scrolling is EVIDENCE that you know whether it works and can prove you made
it better. That evidence is what this file produces.

It measures four things over a fixed test set of questions whose answers we
know in advance (the "golden set" in eval/golden_set.json):

1. RETRIEVAL RECALL @ k
   Did the correct source document appear in the top-k retrieved chunks?
   This isolates the retriever from the generator. If recall is low, no LLM
   can save you, the right info never reached it. This is the metric that
   tells you whether your chunking/hybrid-alpha choices are working.

2. ANSWER CORRECTNESS (LLM-as-judge)
   We use an LLM to compare the system's answer against the known correct
   answer and score it. "LLM-as-judge" is the 2026 standard for grading
   free-text answers at scale because exact-string matching is hopeless for
   prose. We make the judge return strict JSON for a 0-1 score.

3. HALLUCINATION / GROUNDEDNESS
   For questions whose answer is deliberately NOT in the corpus, the system
   should refuse ("I don't have enough information..."). We measure how often
   it correctly abstains vs. makes something up. This is the single most
   important safety metric for enterprise use: a system that confidently
   fabricates is worse than useless in finance or healthcare.

4. CITATION VALIDITY
   Of the answers that made claims, what fraction cited at least one real
   source? Trustworthiness you can measure.

The output is a JSON report plus a printed summary table. The KEY workflow
this enables: run it on config A, run it on config B, compare. That is how
you produce the sentence "changing hybrid alpha from 1.0 (pure vector) to 0.5
(hybrid) raised retrieval recall from X% to Y%", which is the sentence that
gets callbacks.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path

from core.config import Config
from core.pipeline import RAGPipeline
from core.llm import make_llm, LLMClient


@dataclass
class CaseResult:
    question: str
    category: str               # "answerable" or "unanswerable"
    expected_source: str | None
    retrieval_hit: bool         # was expected_source in retrieved chunks?
    correctness: float          # 0..1 from LLM judge (answerable only)
    abstained: bool             # did the system refuse to answer?
    cited_real_source: bool
    attempts_used: int
    answer_text: str


_JUDGE_SYSTEM = """You grade whether a SYSTEM ANSWER correctly matches a REFERENCE
ANSWER for a given question. Focus on factual agreement, not wording. Partial
credit is allowed.

Return ONLY JSON: {"score": <number between 0 and 1>, "reason": "<short>"}
1.0 = fully correct and complete. 0.5 = partially correct or incomplete.
0.0 = wrong, irrelevant, or refused to answer."""


def _extract_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])


def judge_correctness(judge: LLMClient, question: str, reference: str,
                      system_answer: str) -> float:
    user = (f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference}\n\n"
            f"SYSTEM ANSWER: {system_answer}")
    try:
        raw = judge.complete(_JUDGE_SYSTEM, user, temperature=0.0, max_tokens=150)
        return float(_extract_json(raw).get("score", 0.0))
    except Exception:
        return 0.0


_ABSTAIN_MARKER = "don't have enough information"


def run_eval(config: Config, golden_path: str = "eval/golden_set.json",
             report_path: str = "eval/report.json") -> dict:
    pipeline = RAGPipeline(config)
    judge = make_llm("gemini-2.5-flash")
    golden = json.loads(Path(golden_path).read_text())

    results: list[CaseResult] = []
    for case in golden:
        q = case["question"]
        category = case["category"]
        expected = case.get("expected_source")
        reference = case.get("reference_answer", "")

        # On a free-tier rate limit, wait and retry rather than losing the run.
        from core.llm import RateLimitError
        for _attempt in range(3):
            try:
                res = pipeline.answer(q)
                break
            except RateLimitError:
                print(f"  rate limited on '{q[:40]}...', waiting 30s...")
                time.sleep(30)
        else:
            print(f"  skipping '{q[:40]}...' after repeated rate limits")
            continue
        ans = res.answer

        # Retrieval hit: did the expected source appear anywhere we retrieved?
        # We union the sources from every attempt's top results plus the
        # final contexts the model actually saw.
        retrieved_sources: set[str] = {r.chunk.source for r in ans.used_contexts}
        for s in res.steps:
            retrieved_sources.update(s.top_sources)
        hit = (expected in retrieved_sources) if expected else True

        abstained = _ABSTAIN_MARKER in ans.text.lower()

        if category == "answerable":
            correctness = 0.0 if abstained else judge_correctness(judge, q, reference, ans.text)
        else:
            # For unanswerable questions, "correct" behavior is abstaining.
            correctness = 1.0 if abstained else 0.0

        results.append(CaseResult(
            question=q, category=category, expected_source=expected,
            retrieval_hit=hit, correctness=correctness, abstained=abstained,
            cited_real_source=bool(ans.cited_sources),
            attempts_used=res.attempts_used, answer_text=ans.text,
        ))
        time.sleep(0.5)  # be gentle on the free-tier rate limit

    # ---- Aggregate metrics ----
    answerable = [r for r in results if r.category == "answerable"]
    unanswerable = [r for r in results if r.category == "unanswerable"]

    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else 0.0

    metrics = {
        "config": json.loads(config.to_json()),
        "n_cases": len(results),
        "retrieval_recall_pct": pct([r.retrieval_hit for r in answerable]),
        "avg_correctness_pct": round(
            100 * sum(r.correctness for r in answerable) / len(answerable), 1
        ) if answerable else 0.0,
        "correct_abstention_pct": pct([r.abstained for r in unanswerable]),
        "hallucination_pct": pct([not r.abstained for r in unanswerable]),
        "citation_validity_pct": pct(
            [r.cited_real_source for r in answerable if not r.abstained]
        ),
        "avg_attempts": round(
            sum(r.attempts_used for r in results) / len(results), 2
        ) if results else 0.0,
        "cases": [asdict(r) for r in results],
    }

    Path(report_path).write_text(json.dumps(metrics, indent=2))
    _print_summary(metrics)
    return metrics


def _print_summary(m: dict) -> None:
    print("\n" + "=" * 56)
    print("  RAG EVALUATION REPORT")
    print("=" * 56)
    print(f"  Corpus:                 {m['config']['corpus']}")
    print(f"  Hybrid alpha:           {m['config']['hybrid_alpha']}")
    print(f"  Test cases:             {m['n_cases']}")
    print("-" * 56)
    print(f"  Retrieval recall:       {m['retrieval_recall_pct']}%   (higher better)")
    print(f"  Answer correctness:     {m['avg_correctness_pct']}%   (higher better)")
    print(f"  Correct abstention:     {m['correct_abstention_pct']}%   (higher better)")
    print(f"  Hallucination rate:     {m['hallucination_pct']}%   (LOWER better)")
    print(f"  Citation validity:      {m['citation_validity_pct']}%   (higher better)")
    print(f"  Avg retrieval attempts: {m['avg_attempts']}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    import sys
    cfg = Config.from_env()
    if len(sys.argv) > 1:
        cfg.hybrid_alpha = float(sys.argv[1])
    run_eval(cfg)
