"""
Streamlit UI for the Healthcare RAG System.

Run with: PYTHONPATH=. streamlit run app/ui/streamlit_app.py
Requires: Ollama running locally with qwen3:latest
"""

import os
import re
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Streamlit Secrets → Env Vars ─────────────────────────────────────────
# Streamlit Cloud stores secrets in st.secrets. Propagate them to env vars
# so that generator.py and other modules can read them via os.environ.
for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
    if key not in os.environ:
        val = st.secrets.get(key)
        if val:
            os.environ[key] = val

# ── Imports ───────────────────────────────────────────────────────────────
from app.generation.generator import generate_answer
from app.ingestion.pipeline import ingest
from app.embeddings.vector_store import add_chunks, get_collection, reset_collection
from app.retrieval.hybrid_retriever import reset_bm25

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Healthcare RAG", page_icon="🏥", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Status bar animation */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
@keyframes slide-right {
    0% { transform: translateX(0); opacity: 0.3; }
    50% { opacity: 1; }
    100% { transform: translateX(100%); opacity: 0.3; }
}
.status-bar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    z-index: 99999;
    background: linear-gradient(90deg, #1976D2, #42A5F5, #1976D2);
    background-size: 200% 100%;
    animation: slide-right 2s ease-in-out infinite;
}
.status-bar.idle {
    background: #1976D2;
    animation: none;
    opacity: 0.6;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.status-chip .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.status-chip.ready .dot { background: #4CAF50; animation: pulse-dot 2s infinite; }
.status-chip.ready { background: #E8F5E9; color: #2E7D32; }
.status-chip.working .dot { background: #FF9800; animation: pulse-dot 0.8s infinite; }
.status-chip.working { background: #FFF3E0; color: #E65100; }
.status-chip.error .dot { background: #F44336; }
.status-chip.error { background: #FFEBEE; color: #C62828; }

/* About page cards */
.about-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
    background: #fafbfc;
    transition: box-shadow 0.2s;
}
.about-card:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.about-card h4 {
    margin: 0 0 8px 0;
    font-size: 1.05em;
    color: #1976D2;
}
.about-card p {
    margin: 0;
    color: #444;
    font-size: 0.92em;
    line-height: 1.55;
}

/* Phase pipeline visual */
.phase-badge {
    display: inline-block;
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #1976D2;
    color: #fff;
    text-align: center;
    line-height: 32px;
    font-weight: 700;
    font-size: 0.85em;
    margin-right: 10px;
    flex-shrink: 0;
}
.phase-row {
    display: flex;
    align-items: flex-start;
    margin: 14px 0;
}
.phase-content {
    flex: 1;
}
.phase-content strong {
    font-size: 1em;
    color: #1a1a1a;
}
.phase-content .desc {
    color: #555;
    font-size: 0.88em;
    margin-top: 2px;
    line-height: 1.5;
}
.phase-connector {
    width: 2px;
    height: 16px;
    background: #1976D2;
    margin-left: 15px;
    opacity: 0.4;
}

/* Problem card */
.problem-card {
    border-radius: 10px;
    padding: 16px 18px;
    margin: 6px 0;
    border-left: 4px solid;
}
.problem-card.hallucination { background: #FFF3E0; border-color: #FF9800; }
.problem-card.stale { background: #E3F2FD; border-color: #1976D2; }
.problem-card.provenance { background: #F3E5F5; border-color: #7B1FA2; }
.problem-card.context { background: #FFEBEE; border-color: #F44336; }

/* Tech stack pill */
.tech-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    background: #E3F2FD;
    color: #1565C0;
    font-size: 0.82em;
    font-weight: 600;
    margin: 3px 4px;
}

/* Sample question button */
.sample-q {
    display: block;
    width: 100%;
    padding: 10px 14px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #fafbfc;
    text-align: left;
    font-size: 0.9em;
    color: #333;
    cursor: pointer;
    transition: all 0.15s;
    margin: 4px 0;
}
.sample-q:hover {
    border-color: #1976D2;
    background: #E3F2FD;
}

/* Source-highlighted answer spans */
.src-span {
    border-bottom: 2px solid var(--src-color-dim);
    padding: 1px 2px;
    border-radius: 2px;
    transition: all 0.15s ease;
    cursor: default;
    position: relative;
}
.src-span:hover {
    background: var(--src-color-bg);
    border-bottom-color: var(--src-color);
}
/* Tooltip */
.src-span .src-tip {
    display: none;
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #1a1a1a;
    color: #fff;
    font-size: 0.75em;
    padding: 5px 10px;
    border-radius: 6px;
    white-space: nowrap;
    z-index: 1000;
    pointer-events: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.src-span .src-tip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #1a1a1a;
}
.src-span:hover .src-tip {
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
_CITATION_COLOURS = [
    "#1976D2", "#388E3C", "#7B1FA2", "#E64A19",
    "#00838F", "#F57F17", "#5D4037", "#455A64",
]

SAMPLE_QUESTIONS = {
    "sample_clinical_guideline.txt": [
        "What are the diagnostic criteria for Type 2 Diabetes?",
        "What is the first-line drug for Type 2 Diabetes and what is the starting dose?",
        "What are the renal contraindications for metformin?",
        "When should metformin be temporarily stopped?",
        "What drugs are recommended for diabetic patients with heart failure?",
        "What is the HbA1c target for elderly patients?",
        "How should painful diabetic neuropathy be treated?",
        "What monitoring is recommended for diabetic patients?",
    ],
    "discharge_summary_james_wilson.txt": [
        "What was the patient's admission diagnosis?",
        "What was James Wilson's ejection fraction on the echocardiogram?",
        "What medications was the patient discharged on?",
        "Why was metformin stopped at discharge?",
        "What was the patient's eGFR on admission vs discharge?",
        "What follow-up appointments were scheduled?",
        "What red flag symptoms should the patient watch for?",
        "What was done to manage the hyperkalemia?",
    ],
    "lab_report_sarah_chen.txt": [
        "What were Sarah Chen's fasting glucose and HbA1c results?",
        "Does the lab report confirm a diabetes diagnosis?",
        "What is the patient's lipid panel and what does it indicate?",
        "Is there any evidence of kidney disease in the results?",
        "What is the HOMA-IR value and what does it mean?",
        "What actions are recommended based on these lab results?",
        "What was the Vitamin D level and what supplementation is needed?",
        "What did the urinalysis show?",
    ],
}


# ── Helper Functions ──────────────────────────────────────────────────────

def _colour_for_doc(doc_name: str, colour_map: dict) -> str:
    if doc_name not in colour_map:
        colour_map[doc_name] = _CITATION_COLOURS[len(colour_map) % len(_CITATION_COLOURS)]
    return colour_map[doc_name]


def render_answer_with_highlights(answer_text: str, citations: list) -> str:
    """Replace [Source N] markers with superscript badges and wrap preceding
    text segments in hover-highlighted spans with source-colour underlines
    and tooltips (Abstractive Health style)."""

    # Build a lookup: source number -> (colour, tooltip)
    cite_map: dict[int, tuple[str, str]] = {}
    colour_map: dict[str, str] = {}
    for c in citations:
        num = c.get("number")
        if num is None:
            continue
        doc = c.get("source_file", "Unknown")
        colour = _colour_for_doc(doc, colour_map)
        page = c.get("page_number", "?")
        section = c.get("section_title", "")
        tip_parts = [doc, f"Page {page}"]
        if section:
            tip_parts.append(section)
        cite_map[int(num)] = (colour, " — ".join(tip_parts))

    def _make_badge(source_text: str) -> str:
        nums = re.findall(r"\d+", source_text)
        labels = ", ".join(f"[{n}]" for n in nums)
        first_num = int(nums[0]) if nums else None
        colour = cite_map[first_num][0] if first_num and first_num in cite_map else "#1976D2"
        return (
            f'<sup style="background:{colour};color:#fff;border-radius:3px;'
            f'padding:1px 5px;font-size:0.72em;font-weight:700;'
            f'margin-left:1px;">{labels}</sup>'
        )

    parts = re.split(r"(\[Source [^\]]+\])", answer_text)
    html_parts: list[str] = []

    for i, part in enumerate(parts):
        if re.match(r"\[Source [^\]]+\]", part):
            # This is a source marker — render badge
            html_parts.append(_make_badge(part))
        else:
            # Text segment — check if the *next* part is a source marker
            next_is_source = (
                i + 1 < len(parts) and re.match(r"\[Source [^\]]+\]", parts[i + 1])
            )
            if next_is_source and part.strip():
                # Determine colour from the upcoming source marker
                next_nums = re.findall(r"\d+", parts[i + 1])
                first_num = int(next_nums[0]) if next_nums else None
                if first_num and first_num in cite_map:
                    colour, tooltip = cite_map[first_num]
                    html_parts.append(
                        f'<span class="src-span" style="'
                        f"--src-color:{colour};"
                        f"--src-color-dim:{colour}40;"
                        f'--src-color-bg:{colour}1A;">'
                        f'{part}<span class="src-tip">{tooltip}</span></span>'
                    )
                else:
                    html_parts.append(part)
            else:
                # Plain text (no following source marker)
                html_parts.append(part)

    return "".join(html_parts)


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


@st.cache_data(ttl=30)
def _check_llm_backend() -> bool:
    """Ping the LLM backend; cached for 30s to avoid blocking every rerun."""
    import os
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    try:
        import urllib.request
        if "localhost:11434" in base_url:
            # Ollama local server
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            urllib.request.urlopen(req, timeout=0.5)
        else:
            # Cloud API — ping the models endpoint
            api_key = os.environ.get("LLM_API_KEY", "")
            req = urllib.request.Request(f"{base_url}/models", method="GET")
            req.add_header("Authorization", f"Bearer {api_key}")
            urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


def _refresh_collection_cache() -> None:
    """Fetch collection metadata once and stash in session_state."""
    collection = get_collection()
    count = collection.count()
    if count > 0:
        results = collection.get(include=["documents", "metadatas"])
        st.session_state["_col_count"] = count
        st.session_state["_col_metadatas"] = results["metadatas"]
        st.session_state["_col_documents"] = results["documents"]
        st.session_state["_col_docs_list"] = sorted(
            {m["source_file"] for m in results["metadatas"]}
        )
    else:
        st.session_state["_col_count"] = 0
        st.session_state["_col_metadatas"] = []
        st.session_state["_col_documents"] = []
        st.session_state["_col_docs_list"] = []
    st.session_state["_col_dirty"] = False


def _get_system_status() -> tuple[str, str]:
    """Return (status_class, status_text) for the system."""
    import os
    llm_ok = _check_llm_backend()
    doc_count = st.session_state.get("_col_count", 0)
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    backend = "Ollama" if "localhost:11434" in base_url else "Cloud LLM"

    if not llm_ok:
        return "error", f"{backend} offline"
    if doc_count == 0:
        return "ready", f"Ready — no docs loaded"
    return "ready", f"Ready — {doc_count} chunks indexed"


# ── Auto-ingest sample docs on first run ──────────────────────────────────
# On Streamlit Cloud (or fresh installs), ChromaDB is empty. Auto-ingest
# bundled sample documents so the app is usable immediately.
if not st.session_state.get("_auto_ingested"):
    _col = get_collection()
    if _col.count() == 0:
        _raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        _sample_files = sorted(_raw_dir.glob("*.txt")) + sorted(_raw_dir.glob("*.pdf"))
        if _sample_files:
            with st.spinner("Loading sample documents for first run..."):
                for f in _sample_files:
                    try:
                        chunks = ingest(str(f))
                        add_chunks(chunks)
                        reset_bm25()
                    except Exception as e:
                        print(f"[Auto-ingest] Failed for {f.name}: {e}")
            st.session_state["_col_dirty"] = True
    st.session_state["_auto_ingested"] = True

# ── Status Bar ────────────────────────────────────────────────────────────
if st.session_state.get("pipeline_status") == "querying":
    st.session_state.pipeline_status = "idle"
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = "idle"

# Fetch collection data once per rerun (or when flagged dirty)
if st.session_state.get("_col_dirty", True):
    _refresh_collection_cache()

status_class, status_text = _get_system_status()

# Override if actively processing
if st.session_state.pipeline_status == "ingesting":
    status_class, status_text = "working", "Ingesting document..."
elif st.session_state.pipeline_status == "querying":
    status_class, status_text = "working", "Retrieving & generating answer..."

status_bar_class = "status-bar" if status_class == "working" else "status-bar idle"
st.markdown(f'<div class="{status_bar_class}"></div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
header_cols = st.columns([6, 2])
with header_cols[0]:
    st.title("Healthcare Document Q&A")
    st.caption("Upload clinical documents. Ask questions in plain English. Get cited, grounded answers.")
with header_cols[1]:
    st.markdown('<div style="padding-top:12px;"></div>', unsafe_allow_html=True)
    with st.popover(f"● {status_text}"):
        col_count = st.session_state.get("_col_count", 0)
        if col_count == 0:
            st.markdown("No documents loaded yet.")
        else:
            docs = st.session_state["_col_docs_list"]
            all_metadatas = st.session_state["_col_metadatas"]
            all_documents = st.session_state["_col_documents"]
            st.markdown(f"**{col_count} chunks** across **{len(docs)} document(s)**")
            for doc in docs:
                paired = [
                    (text, meta)
                    for text, meta in zip(all_documents, all_metadatas)
                    if meta["source_file"] == doc
                ]
                paired.sort(key=lambda x: x[1].get("chunk_index", 0))
                st.markdown(f"---\n**{doc}** — {len(paired)} chunks")
                for text, meta in paired[:10]:
                    section = meta.get("section_title", "")
                    preview = text[:150].replace("\n", " ")
                    label = f"*{section}* — " if section else ""
                    st.caption(f"{label}{preview}...")
                if len(paired) > 10:
                    st.caption(f"*...and {len(paired) - 10} more chunks*")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Loaded Documents")
    total_vectors = st.session_state.get("_col_count", 0)
    if total_vectors == 0:
        st.info("No documents loaded. Upload one below.")
    else:
        docs = st.session_state["_col_docs_list"]
        metadatas = st.session_state["_col_metadatas"]
        st.success(f"{total_vectors} chunks across {len(docs)} doc(s)")
        for doc in docs:
            doc_chunks = [m for m in metadatas if m["source_file"] == doc]
            st.markdown(f"**{doc}** — {len(doc_chunks)} chunks")
        if st.button("Clear all documents", type="secondary"):
            reset_collection()
            reset_bm25()
            st.session_state["_col_dirty"] = True
            st.rerun()

    st.divider()

    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a clinical document (.pdf or .txt)", type=["pdf", "txt"]
    )
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            st.session_state.pipeline_status = "ingesting"
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    tmp_dir = Path(tempfile.mkdtemp())
                    tmp_path = str(tmp_dir / uploaded_file.name)
                    with open(tmp_path, "wb") as tmp:
                        tmp.write(uploaded_file.read())
                    chunks = ingest(tmp_path)
                    add_chunks(chunks)
                    reset_bm25()
                    st.session_state.pipeline_status = "idle"
                    st.session_state["_col_dirty"] = True
                    st.success(f"Ingested {len(chunks)} chunks from {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.session_state.pipeline_status = "idle"
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # Sample questions
    st.header("Try a Question")
    if total_vectors > 0:
        loaded_docs = st.session_state["_col_docs_list"]
        available_qs = []
        for d in loaded_docs:
            available_qs.extend(SAMPLE_QUESTIONS.get(d, []))
        if available_qs:
            for q in available_qs[:6]:
                if st.button(q, key=f"sq_{q[:30]}"):
                    st.session_state["prefill_question"] = q
                    st.session_state["run_prefill"] = True
                    st.rerun()
        else:
            st.caption("Upload one of the sample documents to see suggested questions.")
    else:
        st.caption("Load a document first to see suggested questions.")

# ── Session State ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_qa, tab_docs, tab_about = st.tabs(["Ask a Question", "Browse Documents", "About"])

# ── Tab 1: Q&A ────────────────────────────────────────────────────────────
with tab_qa:
    prefill = st.session_state.pop("prefill_question", "")
    run_prefill = st.session_state.pop("run_prefill", False)

    with st.form("query_form", clear_on_submit=True):
        query = st.text_input(
            "Enter your question:",
            value=prefill,
            placeholder="e.g., What are the renal contraindications for metformin?",
            key="query_input",
        )
        submitted = st.form_submit_button("Ask")

    # clear_on_submit may clear query to the default (empty) on submit rerun.
    # Recover the actual submitted value from session state if needed.
    if submitted and not query.strip():
        query = st.session_state.get("query_input", "")

    # Run query from either form submit or prefill click
    active_query = None
    if submitted and query.strip():
        active_query = query.strip()
    elif run_prefill and prefill:
        active_query = prefill

    if active_query:
        st.session_state.pipeline_status = "querying"
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = generate_answer(active_query, retrieval_top_k=10, rerank_top_k=5)
                st.session_state.history.append(result)
                st.session_state.pipeline_status = "idle"
            except Exception as e:
                st.session_state.pipeline_status = "idle"
                st.error(f"Error: {e}. Check that your LLM backend is running.")

    for i, result in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        st.divider()
        st.subheader(f"Q{idx}: {result['query']}")

        answer_text = result.get("answer_text") or _fallback_answer_text(result.get("answer", ""))
        citations = result.get("citations", [])

        st.markdown(render_answer_with_highlights(answer_text, citations), unsafe_allow_html=True)

        if citations:
            st.markdown("**Citations**")
            colour_map: dict = {}
            for citation in citations:
                colour = _colour_for_doc(citation.get("source_file", ""), colour_map)
                render_citation_card(citation, colour)

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

# ── Tab 2: Document Browser ──────────────────────────────────────────────
with tab_docs:
    if st.session_state.get("_col_count", 0) == 0:
        st.info("No documents loaded yet. Upload one using the sidebar.")
    else:
        docs = st.session_state["_col_docs_list"]
        all_documents = st.session_state["_col_documents"]
        all_metadatas = st.session_state["_col_metadatas"]

        selected_doc = st.selectbox("Select document", docs)

        paired = list(zip(all_documents, all_metadatas))
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

# ── Tab 3: About ──────────────────────────────────────────────────────────
with tab_about:

    # Hero section
    st.markdown(
        '<div style="background:linear-gradient(135deg,#E3F2FD 0%,#f5f5ff 100%);'
        'border-radius:16px;padding:36px 32px;margin-bottom:24px;">'
        '<h2 style="margin:0 0 8px 0;color:#0D47A1;">Healthcare Document RAG</h2>'
        '<p style="font-size:1.1em;color:#333;margin:0 0 16px 0;line-height:1.6;">'
        'A production-grade <strong>Retrieval-Augmented Generation</strong> system '
        'purpose-built for clinical documents. Upload any health document, ask questions '
        'in plain English, and get <strong>cited, grounded answers</strong>. '
        'Everything runs <strong>fully offline</strong> using a local LLM.</p>'
        '<div>'
        '<span class="tech-pill">Fully Offline</span>'
        '<span class="tech-pill">Citation-Enforced</span>'
        '<span class="tech-pill">Structure-Aware Chunking</span>'
        '<span class="tech-pill">Hybrid Retrieval</span>'
        '<span class="tech-pill">Cross-Encoder Reranking</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── What is RAG? ──────────────────────────────────────────────────────
    st.markdown("### What is RAG?")
    st.markdown(
        "**RAG (Retrieval-Augmented Generation)** stops an LLM from making things up "
        "by forcing it to answer *only* from documents you provide. Think of it like "
        "an **open-book exam** — the model must cite the page it found the answer on."
    )

    flow_cols = st.columns(5)
    flow_items = [
        ("1", "Your Question", "Natural language query"),
        ("2", "Search", "Find relevant passages"),
        ("3", "Retrieve", "Vector + keyword matching"),
        ("4", "Generate", "LLM answers from context"),
        ("5", "Cite", "Every claim linked to source"),
    ]
    for col, (num, title, desc) in zip(flow_cols, flow_items):
        with col:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div style="width:40px;height:40px;border-radius:50%;background:#1976D2;'
                f'color:#fff;line-height:40px;font-weight:700;margin:0 auto 6px auto;">{num}</div>'
                f'<strong style="font-size:0.9em;">{title}</strong><br>'
                f'<span style="font-size:0.78em;color:#666;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Why RAG for Healthcare? ───────────────────────────────────────────
    st.markdown("### Why RAG for Healthcare?")
    st.markdown("Standard LLMs are **dangerous** for healthcare. Here's why:")

    prob_cols = st.columns(2)
    problems = [
        ("hallucination", "Hallucination",
         "LLM invents a dosage not in the guideline — patient receives wrong dose."),
        ("stale", "Stale Knowledge",
         "LLM trained in the past doesn't know the latest 2024 guideline update."),
        ("provenance", "No Provenance",
         "LLM gives an answer but can't say where it came from — clinician can't verify."),
        ("context", "Context Destruction",
         "Naive chunking separates a WARNING from its dosage — safety alert is missed."),
    ]
    for i, (cls, title, desc) in enumerate(problems):
        with prob_cols[i % 2]:
            st.markdown(
                f'<div class="problem-card {cls}">'
                f'<strong>{title}</strong><br>'
                f'<span style="font-size:0.9em;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.success(
        "**This system addresses all four:** answers come only from your uploaded documents, "
        "every claim is cited with [Source N], structure-aware chunking keeps warnings "
        "with their content, and you can upload the latest guideline any time."
    )

    st.markdown("")

    # ── The 5-Phase Pipeline ──────────────────────────────────────────────
    st.markdown("### The 5-Phase Pipeline")
    st.markdown(
        "Every question flows through five stages — each one narrows down to the "
        "most relevant, accurate answer."
    )

    phases = [
        ("1", "Ingestion",
         "Load PDF/TXT, detect section structure (headers, warnings, lists), "
         "split into metadata-rich chunks. Warnings are never separated from their content."),
        ("2", "Embedding",
         "Convert each chunk into a 384-dim vector using <strong>all-MiniLM-L6-v2</strong>. "
         "Store in <strong>ChromaDB</strong> with HNSW indexing for O(log n) search."),
        ("3", "Hybrid Retrieval",
         "<strong>Vector search</strong> (semantic similarity) + <strong>BM25</strong> "
         "(keyword matching) run in parallel. Results are merged with "
         "<strong>Reciprocal Rank Fusion</strong> — chunks in both lists get boosted."),
        ("4", "Reranking",
         "A <strong>cross-encoder</strong> reads each (query, chunk) pair together with "
         "full cross-attention. Much more accurate than bi-encoder, applied only to the "
         "top candidates for efficiency."),
        ("5", "Generation",
         "Top chunks are assembled into a citation-enforced prompt. "
         "<strong>Ollama/qwen3</strong> generates the answer locally — every claim "
         "must cite [Source N] or the model must say it doesn't know."),
    ]

    for i, (num, title, desc) in enumerate(phases):
        st.markdown(
            f'<div class="phase-row">'
            f'<div class="phase-badge">{num}</div>'
            f'<div class="phase-content">'
            f'<strong>{title}</strong>'
            f'<div class="desc">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if i < len(phases) - 1:
            st.markdown('<div class="phase-connector"></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Architecture ──────────────────────────────────────────────────────
    st.markdown("### Architecture")
    st.markdown(
        '<div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:12px;'
        'padding:20px 24px;font-family:monospace;font-size:0.85em;line-height:1.8;'
        'color:#333;overflow-x:auto;">'
        'Upload (PDF/TXT)<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>'
        'Structure-Aware Chunking &rarr; JSON (BM25 index)<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>'
        'Embed (MiniLM) &rarr; ChromaDB (vector store)<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>'
        'Query &rarr; Vector Search + BM25 Search<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        'v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reciprocal Rank Fusion (RRF)<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cross-Encoder Reranker<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LLM Generation (Ollama/qwen3)<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>Cited Answer with [Source N]</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Key Design Decisions ──────────────────────────────────────────────
    st.markdown("### Key Design Decisions")

    decisions = [
        ("Local Embeddings", "not OpenAI",
         "Clinical data never leaves the machine. Zero API cost, no network latency, "
         "and embeddings stay consistent across runs."),
        ("ChromaDB", "not Pinecone/Weaviate",
         "Runs embedded in the Python process with zero infrastructure. No account "
         "or API key — ideal for sensitive health data."),
        ("Hybrid Retrieval", "BM25 + Vector",
         "Vector search misses exact lab values; BM25 misses paraphrased questions. "
         "In healthcare, both matter. Hybrid retrieval captures both."),
        ("Structure-Aware Chunking", "not token-based",
         "A WARNING block before a dosage is inseparable. Generic chunking destroys "
         "this structure — structure-aware chunking preserves it."),
        ("Cross-Encoder Reranker", "funnel pattern",
         "Bi-encoder retrieval is fast but rough. Cross-encoder reads query and "
         "chunk together for precision — applied only to a small candidate set."),
    ]

    for title, subtitle, desc in decisions:
        st.markdown(
            f'<div class="about-card">'
            f'<h4>{title} <span style="color:#999;font-weight:400;">({subtitle})</span></h4>'
            f'<p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Tech Stack ────────────────────────────────────────────────────────
    st.markdown("### Tech Stack")

    tech_data = [
        ("Frontend", ["Streamlit"]),
        ("LLM", ["Ollama", "qwen3 (local)"]),
        ("Embeddings", ["all-MiniLM-L6-v2", "sentence-transformers"]),
        ("Vector Store", ["ChromaDB", "HNSW indexing"]),
        ("Retrieval", ["BM25", "Vector Search", "RRF Fusion"]),
        ("Reranking", ["ms-marco-MiniLM-L-6-v2", "cross-encoder"]),
        ("Evaluation", ["Custom heuristics", "Ragas (optional)"]),
        ("Data Validation", ["Pydantic"]),
    ]

    tech_cols = st.columns(4)
    for i, (category, items) in enumerate(tech_data):
        with tech_cols[i % 4]:
            pills = "".join(f'<span class="tech-pill">{item}</span>' for item in items)
            st.markdown(
                f'<div style="margin-bottom:16px;">'
                f'<strong style="font-size:0.85em;color:#666;">{category}</strong><br>'
                f'{pills}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── How to Use ────────────────────────────────────────────────────────
    st.markdown("### Quick Start")
    qstart_cols = st.columns(3)
    steps = [
        ("Step 1", "Upload a Document",
         "Use the sidebar to upload a .pdf or .txt clinical document, then click Ingest."),
        ("Step 2", "Ask a Question",
         "Type a question in plain English on the Ask a Question tab, or click a suggested question."),
        ("Step 3", "Review Citations",
         "Read the answer with inline [Source N] badges. Expand citations to see exact excerpts."),
    ]
    for col, (step, title, desc) in zip(qstart_cols, steps):
        with col:
            st.markdown(
                f'<div class="about-card" style="text-align:center;">'
                f'<div style="font-size:0.75em;color:#1976D2;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{step}</div>'
                f'<h4 style="color:#1a1a1a;">{title}</h4>'
                f'<p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#999;font-size:0.82em;padding:8px 0;">'
        'Built with Streamlit, ChromaDB, Ollama, and sentence-transformers. '
        'Fully offline — no data leaves your machine.'
        '</div>',
        unsafe_allow_html=True,
    )
