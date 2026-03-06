"""
Streamlit UI for the Healthcare RAG System.

Run with: streamlit run app/ui/streamlit_app.py
Requires: Ollama running locally with qwen3:latest
"""

import re
import tempfile
from pathlib import Path

import streamlit as st

from app.generation.generator import generate_answer
from app.ingestion.pipeline import ingest
from app.embeddings.vector_store import add_chunks, get_collection, reset_collection
from app.retrieval.hybrid_retriever import reset_bm25


st.set_page_config(page_title="Healthcare RAG", page_icon="🏥", layout="wide")
st.title("Healthcare Document Q&A")
st.caption("Ask questions about clinical guidelines. Answers include citations to source documents.")

# --- Sidebar ---
with st.sidebar:
    # Loaded documents preview
    st.header("Loaded Documents")
    collection = get_collection()
    total_vectors = collection.count()
    if total_vectors == 0:
        st.info("No documents loaded. Upload one below.")
    else:
        results = collection.get(include=["metadatas"])
        docs = sorted({m["source_file"] for m in results["metadatas"]})
        st.success(f"{total_vectors} chunks indexed across {len(docs)} document(s)")
        for doc in docs:
            doc_chunks = [m for m in results["metadatas"] if m["source_file"] == doc]
            st.markdown(f"📄 **{doc}** — {len(doc_chunks)} chunks")
        if st.button("Clear all documents", type="secondary"):
            reset_collection()
            reset_bm25()
            st.rerun()

    st.divider()

    # Upload
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a clinical document (.pdf or .txt)", type=["pdf", "txt"])
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    tmp_dir = Path(tempfile.mkdtemp())
                    tmp_path = str(tmp_dir / uploaded_file.name)
                    with open(tmp_path, "wb") as tmp:
                        tmp.write(uploaded_file.read())
                    chunks = ingest(tmp_path)
                    add_chunks(chunks)
                    reset_bm25()
                    st.success(f"Ingested {len(chunks)} chunks from {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

# Session state for Q&A history
if "history" not in st.session_state:
    st.session_state.history = []

tab_qa, tab_docs = st.tabs(["Ask a Question", "Browse Documents"])

# ── Tab 1: Q&A ────────────────────────────────────────────────────────────
with tab_qa:
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input("Enter your question:", placeholder="e.g., What are the renal contraindications for metformin?")
        submitted = st.form_submit_button("Ask")

    if submitted and query.strip():
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = generate_answer(query.strip(), retrieval_top_k=10, rerank_top_k=5)
                st.session_state.history.append(result)
            except Exception as e:
                st.error(f"Error: {e}. Is Ollama running with qwen3:latest?")

    for i, result in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        st.divider()
        st.subheader(f"Q{idx}: {result['query']}")

        answer = result["answer"]
        answer_clean = re.sub(r"</?think>", "", answer).strip()

        if "SOURCES:" in answer_clean:
            answer_part, sources_part = answer_clean.split("SOURCES:", 1)
            answer_part = answer_part.replace("ANSWER:", "").strip()
        else:
            answer_part = answer_clean.replace("ANSWER:", "").strip()
            sources_part = None

        st.markdown(answer_part)

        if sources_part:
            with st.expander("Cited Sources"):
                st.text(sources_part.strip())

        chunks = result.get("context_chunks", [])
        if chunks:
            with st.expander(f"Retrieved Context ({len(chunks)} chunks)"):
                for j, chunk in enumerate(chunks):
                    meta = chunk.get("metadata", {})
                    doc = meta.get("source_file", "unknown")
                    page = meta.get("page_number", "?")
                    section = meta.get("section_title", "?")
                    st.markdown(f"**[Source {j+1}]** {doc} — Page {page} — {section}")
                    st.text(chunk.get("text", "")[:500])
                    if j < len(chunks) - 1:
                        st.markdown("---")

# ── Tab 2: Document Browser ───────────────────────────────────────────────
with tab_docs:
    collection = get_collection()
    if collection.count() == 0:
        st.info("No documents loaded yet. Upload one using the sidebar.")
    else:
        all_data = collection.get(include=["documents", "metadatas"])
        docs = sorted({m["source_file"] for m in all_data["metadatas"]})

        selected_doc = st.selectbox("Select document", docs)

        # Gather chunks for the selected doc, ordered by chunk_index
        paired = list(zip(all_data["documents"], all_data["metadatas"]))
        doc_chunks = sorted(
            [(text, meta) for text, meta in paired if meta["source_file"] == selected_doc],
            key=lambda x: x[1].get("chunk_index", 0),
        )

        st.caption(f"{len(doc_chunks)} chunks — scroll to read the full document")
        st.divider()

        current_section = None
        for text, meta in doc_chunks:
            section = meta.get("section_title", "")
            if section and section != current_section:
                st.subheader(section)
                current_section = section
            st.markdown(text)
            st.divider()
