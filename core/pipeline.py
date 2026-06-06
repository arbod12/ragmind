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
    st.subheader("How to use this")
    st.markdown(
        "1. Pick a knowledge source below.\n"
        "2. Type a question in the box.\n"
        "3. Click **Ask**.\n\n"
        "The system searches, checks whether it found enough, retries if not, "
        "and answers with sources cited, or tells you if the answer is not in "
        "the documents."
    )
    st.divider()
    st.subheader("Knowledge source")
    source_mode = st.radio(
        "What should it search?",
        ["Built-in cybersecurity docs", "Upload my own document"],
        help="Built-in: NIST, MITRE ATT&CK, and vulnerability notes. "
             "Upload: ask questions about your own file (used only for this "
             "session, never saved).",
    )

    uploaded = None
    if source_mode == "Upload my own document":
        uploaded = st.file_uploader(
            "Upload a document (PDF, Word, or text)",
            type=["pdf", "docx", "txt", "md"],
            help="Max 5 MB. Your file is used only for your questions in this "
                 "session and is not stored.",
        )
        st.caption("Try a published security PDF, a policy document, or any "
                   "text-based file, then ask questions about it.")

    st.divider()
    with st.expander("Advanced engine controls"):
        alpha = st.slider("Hybrid alpha (0=keyword, 1=semantic)", 0.0, 1.0, 0.5, 0.1,
                          help="Blends keyword search with semantic search.")
        max_attempts = st.slider("Max retrieval attempts", 1, 5, 3,
                                 help="How many times it may rewrite and retry.")
        corpus = st.selectbox("Built-in corpus", ["security", "secondary"],
                              help="Which built-in document set to use.")
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set. Add it in Streamlit secrets.")


@st.cache_resource(show_spinner="Indexing corpus (one-time)...")
def get_pipeline(corpus_name: str, hybrid_alpha: float, max_att: int):
    """Cache the built-in-corpus pipeline so we do not re-embed on every
    keystroke. The cache key is the tuple of args, so changing corpus/alpha
    rebuilds."""
    cfg = Config()
    cfg.corpus = corpus_name
    cfg.hybrid_alpha = hybrid_alpha
    cfg.max_retrieval_attempts = max_att
    return RAGPipeline(cfg)


@st.cache_resource(show_spinner="Reading and indexing your document...")
def get_uploaded_pipeline(file_name: str, file_hash: str, _data: bytes,
                          hybrid_alpha: float, max_att: int):
    """Build a pipeline from an uploaded document, in memory only.

    The file is parsed to text, chunked, and indexed for this session. It is
    never written to disk or added to the permanent corpus. We cache on
    (name, hash) so re-asking about the same file does not re-index it, but a
    different file rebuilds. The leading underscore on _data tells Streamlit
    not to try to hash the raw bytes for the cache key (we pass our own hash).
    """
    from core.loader import load_uploaded
    from core.documents import chunk_text

    cfg = Config()
    cfg.hybrid_alpha = hybrid_alpha
    cfg.max_retrieval_attempts = max_att

    doc = load_uploaded(file_name, _data)
    chunks = chunk_text(doc.text, doc.name, cfg.chunk_size, cfg.chunk_overlap)
    pipeline = RAGPipeline(cfg, chunks=chunks)
    return pipeline, doc.note


# ---- Decide which pipeline to use based on the source mode ----
active_pipeline = None
upload_note = ""
ready = True

if source_mode == "Upload my own document":
    if uploaded is None:
        st.info("Upload a document in the sidebar to ask questions about it, "
                "or switch to the built-in cybersecurity docs.")
        ready = False
    else:
        import hashlib
        data = uploaded.getvalue()
        fhash = hashlib.md5(data).hexdigest()
        try:
            active_pipeline, upload_note = get_uploaded_pipeline(
                uploaded.name, fhash, data, alpha, max_attempts)
            st.success(f"Indexed **{uploaded.name}**. Ask a question about it below.")
            if upload_note:
                st.caption(upload_note)
        except ValueError as e:
            st.error(str(e))
            ready = False
else:
    active_pipeline = get_pipeline(corpus, alpha, max_attempts)


placeholder = ("e.g. What is Kerberoasting and how do I mitigate it?"
               if source_mode == "Built-in cybersecurity docs"
               else "e.g. What are the main points of this document?")
question = st.text_input("Your question", placeholder=placeholder)

if st.button("Ask", type="primary") and question and ready and active_pipeline:
    if not os.getenv("GEMINI_API_KEY"):
        st.stop()
    from core.llm import RateLimitError
    try:
        with st.spinner("Retrieving, grading, answering..."):
            result = active_pipeline.answer(question)
    except RateLimitError:
        st.warning("The rate limit was hit (too many questions in a short "
                   "time). Wait about a minute, then ask again.")
        st.stop()

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
