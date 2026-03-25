"""
Streamlit Frontend — AI Knowledge Assistant
Clean, production-grade UI for RAG-based document Q&A
"""

import time
import requests
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background: #0e0e12; }
.block-container { padding-top: 2rem; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

.rag-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.rag-header h1 {
    color: #e2e8f0;
    font-size: 1.8rem;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.rag-header p {
    color: #64748b;
    margin: 0;
    font-size: 0.95rem;
}
.accent { color: #818cf8; }

.stat-card {
    background: #13131f;
    border: 1px solid #1e1e35;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #818cf8;
}
.stat-label {
    font-size: 0.75rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

.answer-box {
    background: #13131f;
    border: 1px solid #1e1e35;
    border-left: 3px solid #818cf8;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.7;
}

.source-chip {
    display: inline-block;
    background: #1e1e35;
    border: 1px solid #2d2d50;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #94a3b8;
    margin: 0.2rem;
}
.score-high { border-color: #22c55e; color: #86efac; }
.score-mid  { border-color: #f59e0b; color: #fcd34d; }
.score-low  { border-color: #ef4444; color: #fca5a5; }

.chunk-card {
    background: #0e0e18;
    border: 1px solid #1e1e35;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
    color: #94a3b8;
}
.chunk-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    margin-bottom: 0.5rem;
}

.upload-zone {
    background: #0e0e18;
    border: 2px dashed #1e1e35;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.2s;
}

.stButton > button {
    background: #3730a3;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #4338ca; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #13131f !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.history-item {
    background: #13131f;
    border: 1px solid #1e1e35;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    font-size: 0.85rem;
    color: #94a3b8;
}
.history-item:hover { border-color: #3730a3; }

div[data-testid="stSidebar"] {
    background: #0a0a14 !important;
    border-right: 1px solid #1e1e35 !important;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e1e351a;
    font-size: 0.8rem;
    color: #64748b;
}
.step-icon { font-size: 1rem; }
.step-active { color: #818cf8; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "stats" not in st.session_state:
    st.session_state.stats = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        if r.ok:
            st.session_state.stats = r.json()
    except Exception:
        st.session_state.stats = None

def score_class(score: float) -> str:
    if score >= 0.7:   return "score-high"
    if score >= 0.45:  return "score-mid"
    return "score-low"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#818cf8;">🧠 RAG Assistant</span>
    </div>
    """, unsafe_allow_html=True)

    fetch_stats()
    stats = st.session_state.stats

    if stats:
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:1rem;">
            <div class="stat-number">{stats.get('total_vectors', 0):,}</div>
            <div class="stat-label">Vectors Indexed</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#475569;font-family:'IBM Plex Mono',monospace;line-height:2;">
            <div>Model: <span style="color:#94a3b8;">{stats.get('embedding_model','—')}</span></div>
            <div>Dim: <span style="color:#94a3b8;">{stats.get('embedding_dim','—')}</span></div>
            <div>LLM: <span style="color:#94a3b8;">{stats.get('llm_model','—')}</span></div>
            <div>Top-K: <span style="color:#94a3b8;">{stats.get('top_k','—')}</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠ API not reachable", icon="⚠")

    st.divider()
    st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("📄", "Document Ingestion"),
        ("✂️", "Chunking (512 tok)"),
        ("🔢", "Embedding (MiniLM)"),
        ("🗄️", "FAISS Indexing"),
        ("🔍", "Semantic Retrieval"),
        ("🤖", "LLM Generation"),
    ]
    for icon, label in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <span class="step-icon">{icon}</span>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑 Clear Index", use_container_width=True):
        try:
            r = requests.delete(f"{API_BASE}/index", timeout=10)
            if r.ok:
                st.success("Index cleared")
                fetch_stats()
        except Exception as e:
            st.error(str(e))

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>AI Knowledge Assistant <span class="accent">// RAG</span></h1>
    <p>Upload documents → Build knowledge base → Query with semantic precision</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬  Query", "📥  Ingest Documents", "📚  Knowledge Base"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — QUERY
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col_q, col_h = st.columns([2, 1], gap="large")

    with col_q:
        st.markdown("#### Ask a question")
        question = st.text_area(
            "Your question",
            placeholder="e.g. What are the key findings about transformer architecture?",
            height=120,
            label_visibility="collapsed",
        )

        col_btn, col_topk = st.columns([3, 1])
        with col_topk:
            top_k = st.slider("Retrieve K chunks", 1, 15, 5, label_visibility="collapsed")
        with col_btn:
            run_query = st.button("🔍  Search & Answer", use_container_width=True)

        if run_query and question.strip():
            with st.spinner("Retrieving context and generating answer..."):
                t0 = time.time()
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={"question": question, "top_k": top_k},
                        timeout=300,
                    )
                    elapsed = time.time() - t0

                    if resp.ok:
                        data = resp.json()

                        # Answer
                        st.markdown("#### Answer")
                        st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)

                        # Metadata
                        cols = st.columns(4)
                        cols[0].metric("Chunks Retrieved", data["chunks_retrieved"])
                        cols[1].metric("Response Time", f"{elapsed:.2f}s")
                        cols[2].metric("LLM", data["model_used"])
                        cols[3].metric("Top-K", top_k)

                        # Sources
                        if data["sources"]:
                            st.markdown("#### Retrieved Sources")
                            for i, src in enumerate(data["sources"]):
                                sc = score_class(src["score"])
                                with st.expander(f"Chunk {i+1} — {src['source']} (page {src['page']}) — score {src['score']:.3f}"):
                                    st.markdown(f"""
                                    <div class="chunk-meta">
                                        📄 {src['source']} &nbsp;|&nbsp; Page {src['page']} &nbsp;|&nbsp;
                                        <span class="source-chip {sc}">score {src['score']:.3f}</span>
                                    </div>
                                    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">{src['content']}</div>
                                    """, unsafe_allow_html=True)

                        # Save to history
                        st.session_state.history.insert(0, {
                            "q": question,
                            "a": data["answer"],
                            "chunks": data["chunks_retrieved"],
                        })

                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}\n\nMake sure the backend is running on {API_BASE}")

        elif run_query:
            st.warning("Please enter a question.")

    with col_h:
        st.markdown("#### Query History")
        if not st.session_state.history:
            st.markdown("<div style='color:#475569;font-size:0.85rem;'>No queries yet.</div>", unsafe_allow_html=True)
        for item in st.session_state.history[:10]:
            st.markdown(f"""
            <div class="history-item">
                <div style="color:#e2e8f0;margin-bottom:0.3rem;">{item['q'][:80]}{'…' if len(item['q'])>80 else ''}</div>
                <div style="color:#475569;font-size:0.75rem;">{item['chunks']} chunks retrieved</div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.history:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — INGEST
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Upload Documents")
    st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Supported: PDF, TXT, Markdown — up to 200 MB total</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s) selected:**")
        for f in uploaded:
            st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")

        if st.button("⚙️  Ingest Documents", use_container_width=False):
            with st.spinner("Ingesting, embedding, and indexing..."):
                try:
                    files_payload = [("files", (f.name, f.read(), f.type)) for f in uploaded]
                    resp = requests.post(f"{API_BASE}/ingest", files=files_payload, timeout=300)
                    if resp.ok:
                        result = resp.json()
                        st.success(f"✅ Ingested {result['documents_ingested']} document(s) → {result['chunks_created']} chunks → {result['total_vectors']} total vectors in index")
                        fetch_stats()
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()
    st.markdown("#### Or use a sample document")
    sample_text = st.text_area(
        "Paste text directly",
        height=200,
        placeholder="Paste any text content here and it will be ingested as a document...",
    )
    sample_name = st.text_input("Document name", value="custom_document.txt")
    if st.button("Ingest Text") and sample_text.strip():
        with st.spinner("Ingesting..."):
            try:
                files_payload = [("files", (sample_name, sample_text.encode(), "text/plain"))]
                resp = requests.post(f"{API_BASE}/ingest", files=files_payload, timeout=120)
                if resp.ok:
                    r = resp.json()
                    st.success(f"✅ Ingested → {r['chunks_created']} chunks created")
                    fetch_stats()
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Indexed Documents")
    if st.button("🔄 Refresh"):
        fetch_stats()

    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=5)
        if resp.ok:
            data = resp.json()
            docs = data.get("documents", [])
            total = data.get("total_chunks", 0)

            if docs:
                st.markdown(f"**{len(docs)} source(s) · {total:,} total chunks**")
                for d in docs:
                    pct = (d["chunks"] / total * 100) if total else 0
                    col1, col2, col3 = st.columns([3, 1, 3])
                    col1.markdown(f"📄 `{d['filename']}`")
                    col2.markdown(f"`{d['chunks']}` chunks")
                    col3.progress(pct / 100, text=f"{pct:.1f}%")
            else:
                st.info("No documents ingested yet. Head to **Ingest Documents** to add some.")
        else:
            st.error("Could not load document list from API.")
    except Exception as e:
        st.error(f"Connection error: {e}")

    st.divider()
    st.markdown("#### Architecture Overview")
    arch_cols = st.columns(5)
    steps = [
        ("📄", "Documents", "PDF / TXT / MD"),
        ("✂️", "Chunking", "512 token windows"),
        ("🔢", "Embeddings", "MiniLM-L6-v2"),
        ("🗄️", "FAISS Index", "Cosine similarity"),
        ("🤖", "LLM Answer", "Context-grounded"),
    ]
    for i, (icon, title, sub) in enumerate(steps):
        with arch_cols[i]:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:1.5rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="color:#e2e8f0;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;">{title}</div>
                <div style="color:#475569;font-size:0.7rem;margin-top:0.25rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
