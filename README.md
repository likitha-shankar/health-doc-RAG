# Healthcare Document RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system purpose-built for clinical documents. Upload any health document — clinical guideline, discharge summary, lab report — ask questions in plain English, and get cited, grounded answers. Everything runs **fully offline** using a local LLM.

---

## Table of Contents

- [What is RAG?](#what-is-rag)
- [Why RAG for Healthcare?](#why-rag-for-healthcare)
- [System Architecture](#system-architecture)
- [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
  - [Phase 1 — Ingestion](#phase-1--ingestion)
  - [Phase 2 — Embedding](#phase-2--embedding)
  - [Phase 3 — Hybrid Retrieval](#phase-3--hybrid-retrieval)
  - [Phase 4 — Reranking](#phase-4--reranking)
  - [Phase 5 — Generation](#phase-5--generation)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [Evaluation](#evaluation)
- [CI Quality Gate](#ci-quality-gate)
- [Supported Document Types](#supported-document-types)
- [Glossary](#glossary)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that stops an LLM from making things up by forcing it to answer only from documents you provide.

Without RAG, an LLM answers from memory — which can be outdated, wrong, or hallucinated. With RAG:

```
Your question
     │
     ▼
Search your documents for the most relevant passages
     │
     ▼
Feed those passages + your question to the LLM
     │
     ▼
LLM answers using ONLY what's in the passages, with citations
```

Think of it like an open-book exam: the LLM is not allowed to guess — it must cite the page it found the answer on.

---

## Why RAG for Healthcare?

Healthcare documents have properties that make standard LLMs dangerous:

| Problem | Example | Consequence |
|---------|---------|-------------|
| **Hallucination** | LLM invents a dosage not in the guideline | Patient receives wrong dose |
| **Stale knowledge** | LLM trained in 2023 doesn't know a 2024 guideline update | Outdated treatment applied |
| **No provenance** | LLM gives an answer but can't say where it came from | Clinician can't verify |
| **Context destruction** | Naive chunking separates a WARNING from its dosage | Safety warning is missed |

This system addresses all four:
- Answers come only from your uploaded documents (no hallucination from training data)
- Upload the latest guideline and it immediately takes effect
- Every claim is cited with `[Source N]` linking back to the exact chunk
- Structure-aware chunking keeps warnings with their associated content

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│   ┌──────────────────┐          ┌──────────────────────────┐   │
│   │  Upload Sidebar  │          │   Ask a Question Tab     │   │
│   │  • File uploader │          │   • Question input       │   │
│   │  • Loaded docs   │          │   • Cited answer         │   │
│   │  • Clear all     │          │   • Retrieved chunks     │   │
│   └────────┬─────────┘          └──────────────┬───────────┘   │
└────────────│──────────────────────────────────│───────────────┘
             │ upload                            │ query
             ▼                                  ▼
┌────────────────────┐              ┌────────────────────────────┐
│  INGESTION         │              │  GENERATION PIPELINE       │
│  pipeline.py       │              │                            │
│  ┌──────────────┐  │              │  hybrid_retriever.py       │
│  │ Load (PDF/   │  │              │  ┌──────────┐ ┌─────────┐ │
│  │   txt)       │  │              │  │ Vector   │ │  BM25   │ │
│  └──────┬───────┘  │              │  │ Search   │ │ Search  │ │
│         ▼          │              │  └────┬─────┘ └────┬────┘ │
│  ┌──────────────┐  │              │       │            │       │
│  │ Structure-   │  │              │       └─────┬──────┘       │
│  │ Aware Chunk  │  │              │             ▼              │
│  └──────┬───────┘  │              │       RRF Fusion           │
│         ▼          │              │             │              │
│  ┌──────────────┐  │              │             ▼              │
│  │ Embed chunks │  │              │       reranker.py          │
│  │ (MiniLM)     │  │  vectors     │       (Cross-Encoder)      │
│  └──────┬───────┘  │              │             │              │
│         ▼          │              │             ▼              │
│  ┌──────────────┐  │              │       generator.py         │
│  │  ChromaDB    │◄─┘              │       (Ollama / qwen3)     │
│  │  (on disk)   │                 │             │              │
│  └──────────────┘                 │             ▼              │
│  ┌──────────────┐                 │    Cited Answer            │
│  │  BM25 index  │                 └────────────────────────────┘
│  │ (.json files)│
│  └──────────────┘
└────────────────────┘
```

---

## Full Pipeline Walkthrough

### Phase 1 — Ingestion

**File:** `app/ingestion/pipeline.py`, `app/ingestion/chunker.py`, `app/ingestion/loaders.py`

When you upload a document, the ingestion pipeline runs three steps:

#### Step 1.1: Load

The loader reads the file into pages. Supports `.txt` and `.pdf`.

```
raw file (PDF or TXT)
        │
        ▼
  list of pages
  [page_1_text, page_2_text, ...]
```

#### Step 1.2: Structure-Aware Chunking

This is the most critical step in healthcare RAG. **Generic chunking is dangerous.**

**Why generic chunking fails in medicine:**

```
Generic chunker splits at 500 tokens:

  Chunk A: "WARNING: Do not give Drug X if patient has renal impairment.
            Dosage:"

  Chunk B: "500mg twice daily for adults."

  Query: "What is the dose of Drug X?"
  Retrieved: Chunk B → "500mg twice daily"
  Missing: The renal impairment warning → DANGEROUS
```

**What this system does instead:**

```
Structure-aware chunker:

  Detects: section headers, WARNING blocks, numbered lists, procedure steps

  Keeps together: a WARNING and the content it refers to
  Labels every chunk with: section_title, page_number, source_file, chunk_index

  Result:
  Chunk A: "4.1 Dosage | WARNING: renal impairment contraindicated | 500mg twice daily"
  ← The warning stays with the dosage. Always.
```

Think of it like cutting a book into flashcards — you cut between chapters, not mid-sentence, and you write the chapter title on every card.

#### Step 1.3: Save JSON

Chunks are saved to `data/processed/<filename>.chunks.json` for:
- **Debugging** — you can inspect chunks visually before embedding
- **BM25 indexing** — the keyword search index is built from these files
- **Reproducibility** — compare chunk quality when you change chunking settings

Each chunk has this structure:

```json
{
  "chunk_id": "sample_clinical_guideline.txt_chunk_3",
  "text": "Metformin is contraindicated in patients with eGFR below 30...",
  "source_file": "sample_clinical_guideline.txt",
  "page_number": 1,
  "section_title": "4.1 First-Line Therapy",
  "chunk_index": 3
}
```

---

### Phase 2 — Embedding

**File:** `app/embeddings/embedder.py`, `app/embeddings/vector_store.py`

#### What is an embedding?

An embedding converts text into a list of numbers (a vector) that captures meaning. Similar meanings produce similar vectors.

```
"metformin kidney warning"  →  [0.23, -0.81, 0.14, ...]   ← 384 numbers
"renal contraindication"    →  [0.21, -0.79, 0.17, ...]   ← similar direction!
"capital of France"         →  [-0.54, 0.33, -0.62, ...]  ← very different
```

This system uses **`sentence-transformers/all-MiniLM-L6-v2`** — a lightweight model that produces 384-dimensional embeddings and runs fast on CPU.

#### Vector Store (ChromaDB)

Embeddings are stored in **ChromaDB**, a local vector database. It uses **HNSW (Hierarchical Navigable Small World)** indexing — a graph-based structure that finds nearest neighbours in O(log n) instead of scanning all vectors.

```
Naive search: compare query against all 10,000 chunks = SLOW
HNSW search:  traverse the graph, skip irrelevant clusters = FAST (milliseconds)
```

ChromaDB persists to disk at `data/chroma_db/` — vectors survive restarts without re-embedding.

---

### Phase 3 — Hybrid Retrieval

**File:** `app/retrieval/hybrid_retriever.py`, `app/retrieval/bm25_retriever.py`, `app/retrieval/fusion.py`

This system uses **two** search methods simultaneously, then combines their results.

#### Method A: Vector Search (Semantic)

Embeds the query and finds the chunks whose vectors are closest in meaning using cosine similarity.

```
Good at: "what can't diabetics take?" → finds "metformin is contraindicated..."
         (even though the words don't match — understands the meaning)

Bad at:  exact medical codes, drug names with unusual spelling
```

#### Method B: BM25 (Keyword)

BM25 is a classic information retrieval algorithm. It scores chunks based on term frequency and document frequency — essentially a smarter version of keyword matching.

```
Good at: "eGFR 30 mL/min" → finds exact numeric thresholds
         drug names, ICD codes, lab values

Bad at:  paraphrased questions where words don't appear verbatim
```

#### Fusion: Reciprocal Rank Fusion (RRF)

Neither method is perfect alone. RRF combines both ranked lists:

```
Vector search results:    BM25 results:
  1. Chunk A (score 0.91)   1. Chunk B (score 12.3)
  2. Chunk B (score 0.87)   2. Chunk A (score 11.1)
  3. Chunk C (score 0.84)   3. Chunk D (score 9.8)

RRF formula: score = Σ 1/(k + rank)   where k=60

Chunk A: 1/(60+1) + 1/(60+2) = 0.0328   ← appears in BOTH → boosted
Chunk B: 1/(60+2) + 1/(60+1) = 0.0328   ← appears in BOTH → boosted
Chunk C: 1/(60+3) + 0         = 0.0159
Chunk D: 0        + 1/(60+3)  = 0.0159
```

Chunks that appear in **both** lists get boosted. This catches the cases each method misses individually.

```
Wide net: retrieve top 15 from vector + top 15 from BM25 = 30 candidates
After RRF: top 10 fused candidates → passed to reranker
```

---

### Phase 4 — Reranking

**File:** `app/reranking/reranker.py`

The 10 fused candidates are good, but not perfectly ranked. The reranker applies precision scoring using a **cross-encoder** model.

#### Bi-encoder vs Cross-encoder

```
Bi-encoder (used in retrieval):
  query → [model] → query_vector
  chunk → [model] → chunk_vector
  similarity = cosine(query_vector, chunk_vector)
  ✓ Fast: encode once, compare with dot product
  ✗ No interaction: query and chunk are encoded independently

Cross-encoder (used in reranking):
  [query + chunk] → [model] → relevance_score
  ✓ Full cross-attention: query and chunk see each other
  ✓ Much more accurate relevance scoring
  ✗ Slow: must run for every (query, chunk) pair
```

The funnel pattern:

```
49 chunks in store
     │
     ▼ BM25 + Vector (fast, ~5ms)
15+15 candidates
     │
     ▼ RRF fusion
10 candidates
     │
     ▼ Cross-encoder (slower, ~200ms for 10 pairs)
5 best chunks → sent to LLM
```

Model used: **`cross-encoder/ms-marco-MiniLM-L-6-v2`** trained on MS MARCO passage ranking.

---

### Phase 5 — Generation

**File:** `app/generation/generator.py`, `app/generation/prompts.py`

The top 5 reranked chunks are assembled into a prompt and sent to the LLM.

#### Prompt Structure

```
SYSTEM PROMPT (persistent behavioral constraints):
  "You are a medical document assistant. Answer ONLY from the provided
   context. Every claim must be cited with [Source N]. If the answer
   is not in the context, say so explicitly."

USER PROMPT:
  [Source 1]: <chunk text from section 4.1>
  [Source 2]: <chunk text from section 4.2>
  [Source 3]: <chunk text from section 7.4>
  ...

  Question: What are the renal contraindications for metformin?
```

#### Why the system prompt matters

System prompts have a **privileged position** — the model treats them as persistent behavioral constraints rather than suggestions. Putting citation rules in the system prompt makes them much harder for the model to override, even with adversarial inputs like "ignore all previous instructions."

#### LLM: Ollama (local, offline)

The system uses **Ollama** with **qwen3:latest** via an OpenAI-compatible API on `localhost:11434`. Everything stays on your machine — no data is sent to any external service. The OpenAI SDK is used pointed at Ollama, meaning the backend can be swapped to any OpenAI-compatible provider (Anthropic, Azure, Together AI) by changing two lines.

#### Output format

```
ANSWER:
Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73m² [Source 1].
In patients with eGFR between 30-45, the dose should be reduced to a maximum of 1000mg
daily [Source 1]. Renal function should be monitored at least annually [Source 1].

SOURCES:
[Source 1] sample_clinical_guideline.txt — Page 1 — 4.1 First-Line Therapy
```

---

## Key Design Decisions

### Why local embeddings instead of OpenAI embeddings?

- **Privacy**: Clinical data never leaves the machine
- **Cost**: Zero API cost regardless of document volume
- **Speed**: No network latency; embeddings run in ~100ms locally
- **Reproducibility**: Embeddings don't change between API versions

### Why ChromaDB instead of Pinecone/Weaviate?

- **Zero infrastructure**: runs embedded in the Python process, persists to disk
- **No account/API key needed**: suitable for sensitive health data
- **Sufficient scale**: HNSW handles millions of chunks efficiently

### Why BM25 + Vector instead of just Vector?

Vector search misses exact matches. BM25 misses paraphrased questions. In healthcare, both matter:
- A doctor asking about "eGFR 30" needs exact number matching (BM25)
- A doctor asking "what kidney tests are needed" needs semantic search (vector)
Hybrid retrieval captures both.

### Why structure-aware chunking?

Medical documents have semantic structure — a WARNING block before a dosage is not separable from the dosage. Generic token-based chunking destroys this. Structure-aware chunking preserves it.

### Why a cross-encoder reranker?

The bi-encoder in retrieval encodes query and document independently — it can't model fine-grained relevance interactions. The cross-encoder reads both together and achieves much higher precision. The trade-off is speed, which is why we use the funnel: cheap retrieval → expensive reranking on a small set.

---

## Project Structure

```
health_doc_RAG/
├── app/
│   ├── ingestion/
│   │   ├── pipeline.py        # Entry point: load → chunk → save → return
│   │   ├── loaders.py         # PDF and TXT file loading
│   │   ├── chunker.py         # Structure-aware chunking logic
│   │   └── models.py          # DocumentChunk Pydantic model
│   ├── embeddings/
│   │   ├── embedder.py        # MiniLM sentence embedding wrapper
│   │   └── vector_store.py    # ChromaDB CRUD: add, search, reset
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Orchestrates vector + BM25 + RRF
│   │   ├── bm25_retriever.py    # BM25 keyword search over chunk JSONs
│   │   └── fusion.py            # Reciprocal Rank Fusion implementation
│   ├── reranking/
│   │   └── reranker.py          # Cross-encoder precision reranking
│   ├── generation/
│   │   ├── generator.py         # Full pipeline: retrieve→rerank→generate
│   │   └── prompts.py           # System prompt and user prompt builder
│   ├── evaluation/
│   │   ├── evaluate.py          # Local heuristic + Ragas evaluation
│   │   ├── heuristic_eval.py    # Faithfulness, relevancy, precision metrics
│   │   └── test_dataset.py      # Hand-crafted Q&A test set
│   └── ui/
│       └── streamlit_app.py     # Web interface
├── configs/
│   └── ingestion_config.yaml    # Chunking parameters
├── data/
│   ├── raw/                     # Original uploaded documents
│   ├── processed/               # Chunk JSONs (used by BM25)
│   ├── chroma_db/               # Persisted vector store
│   └── eval_results/            # Evaluation output JSON
├── .github/workflows/
│   └── eval.yml                 # CI quality gate
├── run_query.py                 # CLI query tool
└── requirements.txt
```

---

## Setup & Running

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running

### Installation

```bash
# Clone
git clone https://github.com/likitha-shankar/health-doc-RAG.git
cd health-doc-RAG

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull the LLM
ollama pull qwen3:latest
```

### Running the Web UI

```bash
streamlit run app/ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

1. Upload a document (`.pdf` or `.txt`) in the sidebar
2. Click **Ingest Document**
3. Switch to **Ask a Question** tab and type your question
4. Switch to **Browse Documents** tab to read the full document content

### Running via CLI

```bash
python run_query.py "What are the renal contraindications for metformin?"
```

### Ingesting a document programmatically

```python
from app.ingestion.pipeline import ingest
from app.embeddings.vector_store import add_chunks

chunks = ingest("data/raw/my_guideline.pdf")
add_chunks(chunks)
```

---

## Evaluation

The system includes a dual evaluation pipeline:

### Local Heuristic Evaluation (default, no API key needed)

Runs in ~2 minutes. Measures three metrics approximating the Ragas standard:

| Metric | What it measures | How | Threshold |
|--------|-----------------|-----|-----------|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Citation presence (45%) + 3-gram overlap with context (55%) | ≥ 0.65 |
| **Answer Relevancy** | Does the answer address the question? | Cosine similarity between question and answer embeddings | ≥ 0.70 |
| **Context Precision** | Did retrieval find the right chunks? | Phrase containment of ground truth in retrieved chunks | ≥ 0.50 |

```bash
python -m app.evaluation.evaluate
```

### Full Ragas Evaluation (requires Google API key)

Uses Google Gemini as an LLM judge for precise claim-level faithfulness and semantic relevancy scoring.

```bash
GOOGLE_API_KEY=your_key python -m app.evaluation.evaluate
```

Thresholds for the full Ragas suite: faithfulness ≥ 0.80, answer relevancy ≥ 0.70, context precision ≥ 0.70, context recall ≥ 0.70.

---

## CI Quality Gate

Every push to `main` and every pull request triggers a GitHub Actions workflow (`.github/workflows/eval.yml`) that:

1. Sets up Python and installs dependencies
2. Installs Ollama and pulls `qwen3:latest`
3. **Ingests the sample clinical guideline** into the vector store
4. Runs the evaluation pipeline against the test dataset
5. **Fails the build** if any metric falls below threshold

This prevents regressions — if a code change degrades retrieval or generation quality, the PR is blocked before merge.

```
push to main → CI → ingest → evaluate → PASS/FAIL
                                         │
                              FAIL = PR blocked
                              PASS = merge allowed
```

---

## Supported Document Types

The system ingests any clinical text document. Examples tested:

| Document Type | Example Questions |
|---------------|-------------------|
| Clinical Practice Guidelines | "What is the first-line treatment for Type 2 Diabetes?" |
| Discharge Summaries | "What medications was the patient discharged on?" |
| Lab Reports | "What was the patient's eGFR on admission?" |
| Radiology Reports | "What did the chest X-ray show?" |
| Operative Notes | "What procedure was performed and what were the findings?" |
| Drug Formularies | "What are the contraindications for drug X?" |

Multiple documents can be loaded simultaneously. Retrieval searches across all of them. The `[Source N]` citation tells you exactly which document the answer came from.

---

## Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — grounding LLM answers in a specific document set |
| **Chunk** | A passage of text extracted from a document, typically 200-1500 characters |
| **Embedding** | A fixed-size vector of numbers representing the meaning of text |
| **Vector Store** | A database optimised for storing and searching embedding vectors |
| **Cosine Similarity** | Measures the angle between two vectors — 1.0 = identical meaning, 0.0 = unrelated |
| **HNSW** | Hierarchical Navigable Small World — a graph index for fast approximate nearest-neighbour search |
| **BM25** | Best Match 25 — a classic keyword-based ranking algorithm used in search engines |
| **RRF** | Reciprocal Rank Fusion — a technique to merge multiple ranked lists into one |
| **Bi-encoder** | A model that embeds query and document independently for fast retrieval |
| **Cross-encoder** | A model that reads query and document together for accurate reranking |
| **Reranking** | A second-pass precision scoring of retrieval candidates |
| **Faithfulness** | The degree to which an answer is supported by the retrieved context |
| **Hallucination** | When an LLM generates information not present in or contradicted by its input |
| **eGFR** | Estimated Glomerular Filtration Rate — a measure of kidney function |
| **HbA1c** | Glycated haemoglobin — a measure of average blood glucose over ~3 months |
| **NYHA Class** | New York Heart Association classification of heart failure severity |
