# Healthcare RAG System — Phase Notes

A production-grade Retrieval-Augmented Generation system for healthcare document Q&A with citation-enforced answers.

---

## Phase 1 — Project Setup and Document Ingestion

### What We Built
- **Data Model** (`app/ingestion/models.py`): Pydantic schema for `DocumentChunk` — every chunk is validated to have `chunk_id`, `text`, `source_file`, `page_number`, `section_title`, and `chunk_index`.
- **Loaders** (`app/ingestion/loaders.py`): PDF (via PyMuPDF) and plain text extractors that return page-level dicts `[{"page": int, "text": str}]`.
- **Domain-Aware Chunker** (`app/ingestion/chunker.py`): Structure-aware splitting that detects section headers, keeps WARNING/CAUTION blocks intact, and respects sentence boundaries.
- **Pipeline** (`app/ingestion/pipeline.py`): Single entry point `ingest(file_path)` — load → chunk → save JSON → return chunks.
- **Config** (`configs/ingestion_config.yaml`): Tunable `max_chunk_size` (1500), `chunk_overlap` (200), `min_chunk_size` (100).

### Why It Matters
Generic chunking (split every N tokens) is dangerous in medical documents:
- A WARNING can get separated from the content it warns about
- Section context (e.g., "Pediatric Dosing") can get lost
- Sentence fragments like "Do not exceed" without the dosage value are dangerous

Our chunker splits at **section boundaries first**, then at **paragraph/sentence boundaries** within sections, and **never splits warning blocks**.

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Pydantic validation | Catches missing metadata at ingestion, not at query time |
| Page-level extraction | Enables page-number citations |
| Section title tracking | Every chunk knows its heading — critical for context and filtering |
| JSON output to `data/processed/` | Allows visual inspection of chunk quality before embedding |
| Warning block preservation | WARNING + content stay together as one indivisible unit |

### Results
- Sample clinical guideline (T2DM) → 13 chunks
- All 13 sections correctly identified (Introduction through Diabetic Neuropathy)
- Warnings confirmed to stay with their associated content

---

## Phase 2 — Embedding and Vector Store

### What We Built
- **Embedder** (`app/embeddings/embedder.py`): Wrapper around `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors). Caches model after first load. Separate `embed_texts()` (batch) and `embed_query()` (single) functions.
- **Vector Store** (`app/embeddings/vector_store.py`): ChromaDB wrapper with persistent storage at `data/chroma_db/`. Supports `add_chunks()`, `search()` with metadata filtering, and `reset_collection()`.

### Core Concepts

**Embeddings**: A text → vector mapping where semantic similarity = vector proximity. "Myocardial infarction" and "heart attack" should produce nearby vectors.

**Why model choice matters for clinical text**: General models underrepresent medical terminology (eGFR, HbA1c, SGLT2). A model trained on web text may not know that "renal impairment" and "kidney failure" are closely related. We compensate with hybrid search (Phase 3) and reranking (Phase 4).

**HNSW indexing**: ChromaDB uses Hierarchical Navigable Small World graphs for O(log n) nearest-neighbor search instead of brute-force O(n).

**Cosine similarity**: Measures the angle between vectors (direction of meaning), not magnitude. Robust to text length differences.

**Metadata filtering**: Search can be restricted by `source_file`, `section_title`, `page_number`, or `chunk_index` BEFORE vector comparison — faster and more precise.

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| `all-MiniLM-L6-v2` | Fast, free, local, good general-purpose baseline. Upgrade to BiomedBERT/MedCPT for production |
| ChromaDB with persistence | Survives restarts, no separate server needed |
| Upsert (not insert) | Re-ingesting a document updates chunks instead of duplicating |
| Batch embedding (100 at a time) | Memory-safe for large documents |
| Cosine distance metric | Standard for sentence embeddings |

### Search Quality Results
| Query | Top Result Section | Distance |
|-------|-------------------|----------|
| "Renal contraindications for metformin?" | 4.1 First-Line Therapy | 0.34 |
| "HbA1c target for elderly patients?" | 5. Glycemic Targets | 0.36 |
| "How to treat diabetic neuropathy?" | 7.4 Diabetic Neuropathy | 0.35 |

All correct. But results #2 and #3 are often loosely related — hybrid search and reranking will improve precision.

---

## Phase 3 — Hybrid Retrieval with BM25 and Vector Search

### What We Built
- **BM25 Retriever** (`app/retrieval/bm25_retriever.py`): Keyword-based sparse retrieval using BM25Okapi. Builds an index from processed chunk JSON files. Simple tokenizer (lowercase, alphanumeric split) that preserves numbers for lab values and dosages.
- **RRF Fusion** (`app/retrieval/fusion.py`): Reciprocal Rank Fusion merges ranked lists from multiple retrievers. Formula: `score = Σ 1/(k + rank)` with k=60.
- **Hybrid Retriever** (`app/retrieval/hybrid_retriever.py`): Single entry point `hybrid_search(query)` — runs vector search + BM25 in parallel, fuses with RRF, returns ranked candidates.

### Why Neither Search Alone Is Sufficient

| Scenario | Vector Search | BM25 |
|----------|--------------|------|
| Specific drug name ("empagliflozin") | May rank low — embeds as generic "diabetes medication" | Exact match — finds it immediately |
| Semantic query ("kidneys failing") | Finds "eGFR below 30", "renal impairment" by meaning | Misses — those words don't appear in query |
| Mixed ("HbA1c target elderly") | Finds semantically related glycemic content | Finds chunks containing "HbA1c" literally |

Hybrid retrieval = **best of both worlds**.

### How RRF Works (Intuition)
Two ranked lists, each with positions. RRF doesn't compare scores (they're on different scales). Instead, it scores each chunk by its **rank position** in each list:
- Chunk appearing as #1 in both lists → highest fused score
- Chunk appearing in only one list → lower score
- k=60 smoothing prevents the #1 result from dominating too aggressively

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Over-retrieve (15 from each, return 10) | Cheap to retrieve more; reranker in Phase 4 does the precision filtering |
| BM25 built from same JSON files as vector store | Ensures both retrievers search identical chunk sets |
| Simple tokenizer (no stemming/lemmatization) | Medical terms shouldn't be stemmed — "renal" ≠ "ren" |
| Zero-score BM25 results filtered out | No term overlap = not a keyword match, so exclude |

### Search Quality Observations
- **"empagliflozin"**: Hybrid boosted the correct chunk (RRF=0.0328) because both retrievers found it
- **"kidneys failing"**: Vector search found renal content; BM25 struggled (no literal keyword overlap)
- **"HbA1c target elderly"**: Both contributed useful results; hybrid merged them effectively
- Phase 4's reranker will further refine these rankings for precision

---

## Phase 4 — Cross-Encoder Reranking

### What We Built
- **Reranker** (`app/reranking/reranker.py`): Cross-encoder wrapper using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Takes hybrid retrieval candidates, scores each `(query, chunk)` pair together, returns top-K by relevance score.

### Bi-Encoder vs. Cross-Encoder

| Property | Bi-Encoder (Phase 2) | Cross-Encoder (Phase 4) |
|----------|---------------------|------------------------|
| How it works | Encodes query and chunk separately into vectors, compares by distance | Reads query + chunk together with full cross-attention |
| Speed | Fast — embed once, compare cheaply | Slow — must process each (query, chunk) pair |
| Accuracy | Good for broad retrieval | Much better for precise relevance scoring |
| Use case | Search 10,000+ chunks → narrow to ~10 | Rerank ~10 candidates → find the best 3-5 |
| Analogy | Speed dating (compare profile cards) | Job interview (deep evaluation per candidate) |

### The Funnel Architecture
```
10,000 chunks → [bi-encoder + BM25] → ~10 candidates → [cross-encoder] → 3-5 best
     fast, approximate                    slow, precise
```

### Score Interpretation
- **Positive scores** (e.g., 3.75, 5.36): Chunk likely answers the question
- **Negative scores** (e.g., -9.03): Chunk is probably irrelevant
- **Score gap**: Large gap between #1 and #2 = high confidence in the top result

### Reranking Results

| Query | Top Result | Score | #2 Score | Gap |
|-------|-----------|-------|----------|-----|
| "Renal contraindications for metformin?" | 4.1 First-Line Therapy | 3.75 | 1.13 | 2.62 |
| "How should diabetic neuropathy be treated?" | 7.4 Diabetic Neuropathy | 4.51 | -3.58 | 8.09 |
| "empagliflozin dosage for cardiovascular disease" | 4.2 Second-Line Therapy | 5.36 | -1.03 | 6.39 |

The reranker amplifies signal and suppresses noise — exactly what we need before feeding context to the LLM.

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Local cross-encoder (not Cohere API) | Free, no API key, fully offline-capable |
| Model cached at module level | Avoids 1-2s reload on every rerank call |
| Batch scoring (all pairs at once) | More efficient than scoring one at a time |
| Returns top_k=5 by default | Enough context for generation without overwhelming the LLM |

---

## Phase 5 — Citation-Enforced Generation

### What We Built
- **Prompt Templates** (`app/generation/prompts.py`): System prompt with strict citation rules + user prompt builder that labels chunks as `[Source N]` with full metadata.
- **Generator** (`app/generation/generator.py`): Full pipeline orchestrator — retrieves → reranks → builds prompt → calls LLM → returns cited answer.
- **Test Script** (`run_query.py`): CLI tool to run queries end-to-end.

### LLM Backend
Using **Ollama** (local, free) with `qwen3:latest` via the OpenAI-compatible API. The code uses the OpenAI SDK pointed at `localhost:11434` — swapping to Claude or GPT is a one-line change (base_url + model name).

### Prompt Engineering for Citation Faithfulness

The system prompt enforces 7 strict rules. The most critical:

| Rule | Why It Matters |
|------|---------------|
| "ONLY use information from provided context" | Prevents hallucination from model's training data |
| "Every claim MUST include [Source X]" | Forces traceability — every fact has a paper trail |
| "If context doesn't contain enough info, say so" | Teaches refusal over confabulation |
| "Do not combine chunks to create new claims" | Prevents false synthesis across documents |
| "Reproduce dosages/warnings exactly" | Medical precision — paraphrasing can change meaning |

### Output Format
```
ANSWER:
[Answer with inline [Source X] citations]

SOURCES:
[Source 1] Document: {file}, Page: {page}, Section: {section}
Excerpt: "{verbatim excerpt}"
```

### Test Results

| Query | Sources Cited | Accuracy |
|-------|--------------|----------|
| "Renal contraindications for metformin?" | [Source 1] = Section 4.1 | eGFR < 30 — exact match to document |
| "Starting/max dose, when to stop metformin?" | [Source 1] = Section 4.1 | 500mg → 2000mg, 48hr contrast media — exact |
| "Neuropathy treatment + dosing?" | [Source 1] = Section 7.4 | All 3 drugs + dose ranges + renal warning — exact |

Key observations:
- Model cites **only the chunks it uses** (not all 5 provided)
- Dosages reproduced **verbatim** from source text
- Warnings included when relevant (contrast media, renal adjustment)
- No hallucination detected across test queries

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Ollama + OpenAI SDK | Free, local, offline. Same SDK works with any OpenAI-compatible API |
| temperature=0.1 | Low creativity = low hallucination risk for factual medical Q&A |
| System prompt (not user prompt) for rules | Model treats system prompts as persistent constraints |
| Labeled chunks [Source N] | Creates a closed citation system — model can only cite what we gave it |
| Separate ANSWER/SOURCES sections | Makes answers parseable and citations auditable |

---

## Phase 6 — Evaluation Pipeline (Local Heuristic Metrics)

### What We Built
- **Test Dataset** (`app/evaluation/test_dataset.py`): 6 curated question–answer–context triples covering drug interactions, dosing, warnings, contraindications, and monitoring.
- **Metric Functions** (`app/evaluation/evaluate.py`): Three local heuristic metrics that approximate Ragas without needing an API key:
  - **Faithfulness**: Citation coverage (45%) + 3-gram context grounding (55%). Strips Qwen3 `/think` tags before scoring.
  - **Answer Relevancy**: Cosine similarity between question and answer embeddings via `all-MiniLM-L6-v2`.
  - **Context Precision**: Substring containment — checks what fraction of ground truth key phrases appear in retrieved chunks, with rank bonus for earlier matches.
- **Pipeline Runner**: Iterates test dataset, runs full RAG pipeline per question, computes metrics, saves results to `data/eval_results/ragas_results.json`.

### Why Local Heuristics (Not Full Ragas)
Full Ragas evaluation requires an LLM judge (e.g., GPT-4, Claude, Gemini) via API key. Our local metrics are:
- **Free and offline** — no API costs during development
- **Fast** — runs in ~2 minutes vs. ~10 minutes with API calls
- **Good enough to catch regressions** — if faithfulness drops from 0.7 to 0.3, something broke

For production evaluation, set `GOOGLE_API_KEY` to enable full Ragas with Gemini as the LLM judge.

### Metric Thresholds (Local Heuristic)
| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| faithfulness | ≥ 0.65 | 3-gram overlap is approximate; LLM judge would use ≥ 0.80 |
| answer_relevancy | ≥ 0.70 | Embedding similarity is reliable; same threshold as full Ragas |
| context_precision | ≥ 0.50 | Substring containment on short snippets; LLM judge would use ≥ 0.70 |

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| 3-grams (not 4-grams) for faithfulness | Medical term paraphrasing makes 4-gram overlap too strict |
| Strip `/think` tags | Qwen3 sometimes includes thinking blocks that pollute scoring |
| Substring containment for precision | SequenceMatcher.ratio() on full chunks vs short ground truth produces inherently low scores |
| Separate citation + grounding scores | Distinguishes "model cites sources" from "answer actually comes from context" |

### Results
All 6 test questions produce scores. All three metrics pass their thresholds:
- Faithfulness: PASS (≥ 0.65)
- Answer Relevancy: PASS (≥ 0.70)
- Context Precision: PASS (≥ 0.50)

---

## Phase 6b — Full Ragas Evaluation with Google Gemini

### What We Built
- **Ragas + Gemini integration** (`app/evaluation/evaluate.py`): `run_ragas_evaluation()` function that uses Google Gemini (free tier via AI Studio) as the LLM judge for proper Ragas evaluation.
- **Auto-mode selection**: The `__main__` block checks for `GOOGLE_API_KEY` — if set, runs full Ragas; otherwise falls back to local heuristics (Phase 6).
- **Environment setup**: `.env.example` documents available env vars, `.gitignore` prevents secrets and artifacts from being committed.

### Why Full Ragas Matters
Local heuristics (Phase 6) catch regressions, but they're approximations:
- **Faithfulness**: N-gram overlap ≠ actual claim verification. Ragas extracts individual claims from the answer and checks each against context using an LLM.
- **Context Precision/Recall**: Substring matching misses semantic equivalence. Ragas uses an LLM to judge whether retrieved context actually supports the reference answer.
- **Answer Relevancy**: Embedding similarity is a decent proxy, but Ragas generates hypothetical questions from the answer and checks alignment with the original question.

### How It Works
1. Checks for `GOOGLE_API_KEY` env var
2. Creates `ChatGoogleGenerativeAI(model="gemini-2.0-flash")` wrapped in `LangchainLLMWrapper`
3. Runs the full RAG pipeline for each test question (same as local mode)
4. Builds `SingleTurnSample` objects with `user_input`, `response`, `retrieved_contexts`, `reference`
5. Creates `EvaluationDataset` and calls `evaluate()` with four metrics
6. Reports per-sample and average scores

### Metric Thresholds (Full Ragas with Gemini)
| Metric | Threshold | vs Local Heuristic |
|--------|-----------|-------------------|
| faithfulness | ≥ 0.80 | Local uses ≥ 0.65 |
| answer_relevancy | ≥ 0.70 | Same threshold |
| context_precision | ≥ 0.70 | Local uses ≥ 0.50 |
| context_recall | ≥ 0.70 | Not available in local mode |

### How to Run
```bash
# Local heuristics (free, no API key)
python -m app.evaluation.evaluate

# Full Ragas with Gemini
GOOGLE_API_KEY=your_key python -m app.evaluation.evaluate
```

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Gemini 2.0 Flash (not Pro) | Free tier, fast, sufficient for evaluation judging |
| Auto-select via env var | No code changes needed — just set the key to upgrade |
| Local heuristics as fallback | Development stays free and offline; Ragas is opt-in |
| Higher thresholds for Ragas | LLM judge is more accurate, so we can set stricter standards |
| context_recall added | Only possible with an LLM judge — checks if retrieved context covers the reference answer |

---

## Phase 7 — CI Quality Gate

### What We Built
- **GitHub Actions Workflow** (`.github/workflows/eval.yml`): Runs the evaluation pipeline on every push and PR to `main`.
- Pipeline: Install dependencies → Start Ollama → Pull model → Run evaluation → Fail if thresholds not met.

### CI Requirements
The evaluation pipeline requires **Ollama** running with the `qwen3:latest` model. In CI:
- Ollama is installed and started as a background service
- The model is pulled before evaluation runs
- If Ollama is unavailable, the workflow will fail (intentional — generation quality can't be tested without the LLM)

### Alternative: Retrieval-Only CI
For environments where running Ollama in CI is impractical, the workflow could be adapted to only test retrieval and reranking metrics (no LLM generation). This would verify the search pipeline without needing a GPU.

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Fail on threshold breach | Prevents quality regressions from being merged |
| Full pipeline in CI | Tests the complete RAG chain, not just components |
| Ollama in CI | Matches local dev environment exactly |

---

## Phase 8 — Streamlit UI

### What We Built
- **Streamlit App** (`app/ui/streamlit_app.py`): Interactive web interface for querying the healthcare RAG system.
- Features:
  - Text input for natural language questions
  - Submit button to trigger the RAG pipeline
  - Answer display with inline `[Source N]` citations
  - Expandable source chunks showing document, page, section, and full text
  - Session state to preserve conversation across reruns

### How to Run
```bash
streamlit run app/ui/streamlit_app.py
```
Requires Ollama running locally with `qwen3:latest`.

### Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Streamlit (not Flask/FastAPI) | Fastest path to interactive UI — no frontend code needed |
| Expandable source chunks | Users can verify citations without cluttering the main answer |
| Session state for history | Preserves Q&A pairs across Streamlit reruns |
| Minimal styling | Focus on functionality; production UI would use a design system |
