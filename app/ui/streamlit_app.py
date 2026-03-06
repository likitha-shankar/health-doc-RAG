"""
Streamlit UI for the Healthcare RAG System.

Run with: streamlit run app/ui/streamlit_app.py
Requires: Ollama running locally with qwen3:latest
"""

import re
import tempfile
from pathlib import Path

import streamlit as st

# Palette for cycling citation card accent colours (one colour per unique source file)
_CITATION_COLOURS = [
    "#1976D2", "#388E3C", "#7B1FA2", "#E64A19",
    "#00838F", "#F57F17", "#5D4037", "#455A64",
]


def _colour_for_doc(doc_name: str, colour_map: dict) -> str:
    if doc_name not in colour_map:
        colour_map[doc_name] = _CITATION_COLOURS[len(colour_map) % len(_CITATION_COLOURS)]
    return colour_map[doc_name]


def render_answer_with_badges(answer_text: str) -> str:
    """Replace [Source N] / [Source N, Source M] markers with styled superscript badges."""
    def _badge(m):
        nums = re.findall(r"\d+", m.group(0))
        labels = ", ".join(f"[{n}]" for n in nums)
        return (
            f'<sup style="background:#1976D2;color:#fff;border-radius:3px;'
            f'padding:1px 5px;font-size:0.72em;font-weight:700;'
            f'margin-left:1px;">{labels}</sup>'
        )
    return re.sub(r"\[Source [^\]]+\]", _badge, answer_text)


def _fallback_answer_text(raw: str) -> str:
    """Strip think tags and SOURCES block from a raw answer string."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    if "SOURCES:" in cleaned:
        cleaned = cleaned.split("SOURCES:", 1)[0]
    return cleaned.replace("ANSWER:", "").strip()


def render_citation_card(citation: dict, colour: str) -> None:
    num = citation["number"]
    doc = citation.get("source_file", "Unknown document")
    page = citation.get("page_number", "?")
    section = citation.get("section_title", "")
    excerpt = citation.get("excerpt", "")

    section_tag = f"<span style='color:#555;'>&middot; {section}</span>" if section else ""

    if excerpt:
        excerpt_html = (
            "<blockquote style='border-left:3px solid "
            + colour
            + ";margin:8px 0 0 0;padding:6px 10px;background:#fafafa;"
            "font-style:italic;color:#333;border-radius:0 4px 4px 0;font-size:0.9em;'>"
            + excerpt
            + "</blockquote>"
        )
    else:
        excerpt_html = ""

    card = (
        f"<div style='border-left:4px solid {colour};border-radius:4px;"
        f"padding:10px 14px;margin:6px 0;background:#f5f8ff;'>"
        f"<div>"
        f"<span style='background:{colour};color:#fff;border-radius:3px;"
        f"padding:1px 7px;font-weight:700;font-size:0.85em;'>[{num}]</span>"
        f"&nbsp;<strong style='font-size:0.95em;'>{doc}</strong>"
        f"&nbsp;<span style='color:#777;font-size:0.85em;'>Page {page}</span>"
        f"&nbsp;{section_tag}"
        f"</div>"
        f"{excerpt_html}"
        f"</div>"
    )
    st.markdown(card, unsafe_allow_html=True)

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

        # Use pre-parsed fields when available (new format), fall back to raw parsing
        answer_text = result.get("answer_text") or _fallback_answer_text(result.get("answer", ""))
        citations = result.get("citations", [])

        # Render answer body with citation badges
        st.markdown(render_answer_with_badges(answer_text), unsafe_allow_html=True)

        # Citation cards — Abstractive Health style
        if citations:
            st.markdown("**Citations**")
            colour_map: dict = {}
            for citation in citations:
                colour = _colour_for_doc(citation.get("source_file", ""), colour_map)
                render_citation_card(citation, colour)

        # Full retrieved context in an expander (for debugging / power users)
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
