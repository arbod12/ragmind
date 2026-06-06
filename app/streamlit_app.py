"""
Streamlit UI: a THIN shell over the engine.

Read this and notice what is NOT here: there is no retrieval logic, no BM25,
no prompt engineering, no agentic loop. All of that lives in core/. This file
only does three things: collect a question, call pipeline.answer(), and
render the result nicely. That separation is the whole point of the
architecture. You could delete this file, write a FastAPI version, and the
engine would not change by one line.

The UI deliberately surfaces the AGENTIC TRACE (how many retrieval attempts,
what the grader decided each time) and the CITATIONS. Those are the two things
that make the system look like real engineering rather than a chatbot, so the
UI puts them front and center instead of hiding them.
"""

import os
import sys

# --- Import path bootstrap ---
# Streamlit runs this file from inside app/, which means Python does not
# automatically know where the project root (and therefore the `core` package)
# is. We add the parent directory of this file (the ragmind/ root) to the
# import path so `from core...` works no matter how the app is launched. This
# makes `streamlit run app/streamlit_app.py` work without needing PYTHONPATH.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from core.config import Config
from core.pipeline import RAGPipeline

st.set_page_config(page_title="RAGmind", page_icon="◆", layout="wide")

# ---- Minimal, distinctive styling (dark, technical, not generic) ----
st.markdown("""
<style>
  .stApp { background: #0e1117; }
  .ragmind-title { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem;
    font-weight: 700; color: #e8eaed; letter-spacing: -0.02em; }
  .ragmind-sub { font-family: 'IBM Plex Mono', monospace; color: #6ee7b7;
    font-size: 0.8rem; letter-spacing: 0.15em; text-transform: uppercase; }
  .trace-card { background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
  .pass { color: #6ee7b7; } .fail { color: #f0883e; }
  .src-pill { display: inline-block; background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb55; border-radius: 20px; padding: 2px 12px;
    margin: 3px; font-family: monospace; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ragmind-sub">agentic retrieval · hybrid search · evaluated</div>',
            unsafe_allow_html=True)
st.markdown('<div class="ragmind-title">◆ RAGmind</div>', unsafe_allow_html=True)
st.caption("Ask a question. The system retrieves with hybrid search, judges whether "
           "it found enough, retries if not, and answers only from cited sources.")

# ---- Sidebar: live controls that map directly to Config ----
with st.sidebar:
    st.subheader("Engine controls")
    corpus = st.selectbox("Knowledge corpus", ["security", "secondary"],
                          help="Swap the body of knowledge with zero code change.")
    alpha = st.slider("Hybrid alpha (0=keyword, 1=semantic)", 0.0, 1.0, 0.5, 0.1,
                      help="Blends BM25 keyword search with dense semantic search.")
    max_attempts = st.slider("Max retrieval attempts", 1, 5, 3,
                             help="How many times the agent may rewrite and retry.")
    st.divider()
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set. Add it in Streamlit secrets or your shell.")
    st.caption("Engine lives in core/. This page is only a UI shell.")


@st.cache_resource(show_spinner="Indexing corpus (one-time)...")
def get_pipeline(corpus_name: str, hybrid_alpha: float, max_att: int):
    """Cache the pipeline so we do not re-embed the corpus on every keystroke.
    The cache key is the tuple of args, so changing corpus/alpha rebuilds."""
    cfg = Config()
    cfg.corpus = corpus_name
    cfg.hybrid_alpha = hybrid_alpha
    cfg.max_retrieval_attempts = max_att
    return RAGPipeline(cfg)


question = st.text_input("Your question",
                         placeholder="e.g. What is Kerberoasting and how do I mitigate it?")

if st.button("Ask", type="primary") and question:
    if not os.getenv("GEMINI_API_KEY"):
        st.stop()
    pipeline = get_pipeline(corpus, alpha, max_attempts)
    with st.spinner("Retrieving, grading, answering..."):
        result = pipeline.answer(question)

    # ---- The answer ----
    st.markdown("### Answer")
    if result.low_confidence:
        st.warning("Low confidence: the system could not fully verify it found "
                   "enough context after all attempts. Treat the answer cautiously.")
    st.write(result.answer.text)

    # ---- Citations ----
    if result.answer.cited_sources:
        st.markdown("**Sources cited:**")
        pills = "".join(f'<span class="src-pill">{s}</span>'
                        for s in result.answer.cited_sources)
        st.markdown(pills, unsafe_allow_html=True)

    # ---- The agentic trace (the impressive part) ----
    st.markdown("### How it reasoned")
    st.caption(f"Used {result.attempts_used} retrieval attempt(s).")
    for step in result.steps:
        verdict = ('<span class="pass">SUFFICIENT</span>' if step.sufficient
                   else '<span class="fail">INSUFFICIENT → rewrote query</span>')
        st.markdown(f"""<div class="trace-card">
            <b>Attempt {step.attempt}</b> · query: <i>{step.query}</i><br>
            retrieved from: {', '.join(step.top_sources)}<br>
            grader: {verdict} — {step.grade_reason}
        </div>""", unsafe_allow_html=True)

    # ---- Retrieved context (transparency) ----
    with st.expander("Show retrieved passages (with scores)"):
        for i, r in enumerate(result.answer.used_contexts, 1):
            st.markdown(f"**[{i}] {r.chunk.source}** · "
                        f"fused={r.score:.3f} dense={r.dense_score:.3f} "
                        f"sparse={r.sparse_score:.3f}")
            st.text(r.chunk.text)
