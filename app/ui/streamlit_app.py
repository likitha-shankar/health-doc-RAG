"""
Streamlit UI for the Healthcare RAG System.

Run with: streamlit run app/ui/streamlit_app.py
Requires: Ollama running locally with qwen3:latest
"""

import re
import streamlit as st

from app.generation.generator import generate_answer


st.set_page_config(page_title="Healthcare RAG", page_icon="🏥", layout="wide")
st.title("Healthcare Document Q&A")
st.caption("Ask questions about clinical guidelines. Answers include citations to source documents.")

# Session state for Q&A history
if "history" not in st.session_state:
    st.session_state.history = []

# Input form
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

# Display results (most recent first)
for i, result in enumerate(reversed(st.session_state.history)):
    idx = len(st.session_state.history) - i
    st.divider()
    st.subheader(f"Q{idx}: {result['query']}")

    # Parse and display the answer
    answer = result["answer"]
    # Strip thinking tags if present
    answer_clean = re.sub(r"</?think>", "", answer).strip()

    # Split into ANSWER and SOURCES sections
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

    # Expandable retrieved chunks
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
