"""
Cross-encoder reranker: precision scoring of candidate chunks.

WHAT THIS MODULE DOES:
Takes the ~10 candidate chunks from hybrid retrieval (Phase 3) and scores
each one against the query using a cross-encoder model. The cross-encoder
reads the query and chunk TOGETHER, with full cross-attention between them,
producing a relevance score that's far more accurate than the bi-encoder
similarity or BM25 score.

WHY THIS IS A SEPARATE STEP:
Cross-encoders are ~100x slower than bi-encoders per comparison. We can't
run them over 10,000 chunks. But we CAN run them over 10 candidates. This
is the classic "funnel" pattern in information retrieval:

    10,000 chunks → [fast retrieval] → 10 candidates → [slow reranker] → 3-5 best

THE MODEL:
We use `cross-encoder/ms-marco-MiniLM-L-6-v2` — a lightweight cross-encoder
trained on MS MARCO (a large-scale passage ranking dataset). It's not
healthcare-specific, but it understands question-passage relevance well.
In production, you'd use a domain-specific model or Cohere Rerank API.

WHAT "SCORE" MEANS:
The cross-encoder outputs a raw logit (not bounded 0-1). Higher = more
relevant. A score of 5.0 means "this chunk very likely answers this question."
A score of -2.0 means "probably irrelevant." We sort by score and take the top N.
"""

from sentence_transformers import CrossEncoder

# Module-level cache — same pattern as the embedder.
# Loading the model once avoids the 1-2 second startup on every rerank call.
_model: CrossEncoder | None = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    """Load the cross-encoder model (cached after first call)."""
    global _model
    if _model is None:
        print(f"[Reranker] Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
        print(f"[Reranker] Model loaded.")
    return _model


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Rerank candidate chunks using the cross-encoder.

    Args:
        query: The user's natural language question.
        candidates: List of result dicts from hybrid retrieval. Each must
                    have at least "text", "chunk_id", and "metadata" keys.
        top_k: How many reranked results to return.

    Returns:
        The top_k candidates sorted by cross-encoder relevance score
        (highest first). Each result gets a "rerank_score" field added.

    How it works internally:
    1. Form pairs: [(query, chunk1_text), (query, chunk2_text), ...]
    2. Feed ALL pairs through the cross-encoder in one batch
    3. Get back one score per pair
    4. Sort by score, return top_k
    """
    if not candidates:
        return []

    model = _get_model()

    # Step 1: Build query-chunk pairs for the cross-encoder.
    # Each pair is (query_text, chunk_text). The model reads both together.
    pairs = [(query, candidate.get("text", "")) for candidate in candidates]

    # Step 2: Score all pairs in a single batch.
    # This is much more efficient than scoring one at a time because the
    # model can parallelize across pairs using batch matrix operations.
    scores = model.predict(pairs)

    # Step 3: Attach scores to candidates and sort.
    scored_candidates = []
    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["rerank_score"] = float(score)
        scored_candidates.append(result)

    # Sort by rerank score (descending — higher is more relevant)
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_candidates[:top_k]
