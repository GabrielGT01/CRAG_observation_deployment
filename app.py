


"""
Streamlit front-end for the Corrective RAG (CRAG) pipeline.

Concurrency model
------------------
Streamlit's server (Tornado) runs each connected browser session in its own
thread and gives it an isolated `st.session_state`. Per-user state (retriever,
graph, chat history) lives in `st.session_state` -- never in module-level
globals -- so sessions can't step on each other, and an unhandled exception
in one user's script run doesn't take the process down for anyone else.

The only intentionally shared, mutable state is the rate limiter's request
log below, which is guarded by a `threading.Lock`.

Health check
------------
Streamlit ships a built-in health endpoint at `/_stcore/health` -- no extra
route needed. Point a Docker HEALTHCHECK or k8s probe at
`http://<host>:8501/_stcore/health`.

Rate limiting
-------------
In-memory sliding-window limiter, keyed by session id, guarding the expensive
path (LLM calls). Process-local -- fine for a single container. If you scale
to multiple replicas later, swap `_request_log` for a Redis-backed counter.
"""

import os
import time
import uuid
import shutil
import threading
from collections import defaultdict, deque

import streamlit as st
from dotenv import load_dotenv

from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.graph.graph_builder import GraphBuilder

load_dotenv()

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/crag_uploads")

st.set_page_config(page_title="CRAG Assistant", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------
# Shared, thread-safe rate limiter
# --------------------------------------------------------------------------
_rate_lock = threading.Lock()
_request_log: dict[str, deque] = defaultdict(deque)


def check_rate_limit(session_id: str) -> tuple[bool, int]:
    """Sliding-window rate limit check. Returns (allowed, seconds_until_retry)."""
    now = time.monotonic()
    with _rate_lock:
        log = _request_log[session_id]
        while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
            log.popleft()
        if len(log) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - log[0]))
            return False, max(retry_after, 1)
        log.append(now)
        return True, 0


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "graph_builder": None,
        "retriever_ready": False,
        "chat_history": [],  # list[(role, content)]
        "last_metrics": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# --------------------------------------------------------------------------
# Sidebar: document ingestion
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Knowledge source")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    url_input = st.text_input("...or a URL", placeholder="https://...")

    chunk_size = st.number_input("Chunk size", value=500, min_value=100, max_value=4000, step=50)
    chunk_overlap = st.number_input("Chunk overlap", value=50, min_value=0, max_value=1000, step=10)
    top_k = st.number_input("Top-k retrieved chunks", value=5, min_value=1, max_value=20)

    if st.button("Build knowledge base", type="primary", use_container_width=True):
        sources = []
        session_upload_dir = os.path.join(UPLOAD_DIR, st.session_state.session_id)
        os.makedirs(session_upload_dir, exist_ok=True)

        for f in uploaded_files or []:
            path = os.path.join(session_upload_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            sources.append(path)

        if url_input:
            sources.append(url_input.strip())

        if not sources:
            st.warning("Upload at least one file or provide a URL.")
        else:
            with st.spinner("Ingesting documents and building the retriever..."):
                try:
                    pipeline = IngestionPipeline(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=top_k
                    )
                    retriever = pipeline.build_retriever(sources)

                    graph_builder = GraphBuilder(retriever)
                    graph_builder.build()

                    # Files are only needed on disk long enough for the loaders
                    # to read them -- their content is now inside the FAISS
                    # index in memory, so the temp copies can go.
                    shutil.rmtree(session_upload_dir, ignore_errors=True)

                    st.session_state.graph_builder = graph_builder
                    st.session_state.retriever_ready = True
                    st.session_state.chat_history = []
                    st.session_state.last_metrics = None
                    st.success(f"Knowledge base ready ({len(sources)} source(s)).")
                except Exception as e:
                    st.session_state.retriever_ready = False
                    st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}`")
    st.caption(f"Rate limit: {RATE_LIMIT_MAX_REQUESTS} requests / {RATE_LIMIT_WINDOW_SECONDS}s")

# --------------------------------------------------------------------------
# Main: chat
# --------------------------------------------------------------------------
st.title("Corrective RAG Assistant")

if not st.session_state.retriever_ready:
    st.info("Build a knowledge base from the sidebar to start chatting.")
else:
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("Ask a question about your documents...")

    if question:
        allowed, retry_after = check_rate_limit(st.session_state.session_id)

        if not allowed:
            st.chat_message("assistant").warning(
                f"Rate limit reached. Try again in {retry_after}s."
            )
        else:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.graph_builder.run(
                            question, thread_id=st.session_state.session_id
                        )
                        answer = result.get("generation", "I couldn't generate an answer.")
                        st.markdown(answer)

                        st.session_state.last_metrics = {
                            "faithfulness": result.get("faithfulness"),
                            "answer_relevancy": result.get("answer_relevancy"),
                            "context_precision": result.get("context_precision"),
                        }
                        st.session_state.chat_history.append(("assistant", answer))
                    except Exception as e:
                        # Confined to this session's thread -- won't affect
                        # other users or crash the server process.
                        error_msg = f"Something went wrong answering that: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(("assistant", error_msg))

    if st.session_state.last_metrics:
        with st.expander("Last answer's RAG metrics"):
            m = st.session_state.last_metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness", "✅" if m["faithfulness"] else "❌")
            c2.metric("Answer relevancy", "✅" if m["answer_relevancy"] else "❌")
            precision = m["context_precision"]
            c3.metric("Context precision", f"{precision:.2f}" if precision is not None else "—")


