"""
Hybrid retriever: combines vector search + BM25, fused with RRF.

This is the single entry point for retrieval in our RAG system. Downstream
components (reranker, generator) never call vector search or BM25 directly —
they call this module and get back a fused, ranked list of candidate chunks.

The two-stage retrieval strategy:
1. Cast a wide net: get top-K from vector search AND top-K from BM25
2. Merge with RRF: chunks that appear in BOTH lists get boosted
3. Return the fused top-N candidates for reranking (Phase 4)

Why retrieve more than we need? Because retrieval is cheap (milliseconds)
but reranking is expensive (requires a cross-encoder model pass). We
over-retrieve here and let the reranker in Phase 4 whittle down to the
truly best chunks.
"""

from app.embeddings.vector_store import search as vector_search
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.fusion import reciprocal_rank_fusion

# Module-level cache for the BM25 retriever
_bm25_retriever: BM25Retriever | None = None


def _get_bm25() -> BM25Retriever:
    """Lazily load the BM25 index from processed chunk files."""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever.from_processed_dir()
    return _bm25_retriever


def hybrid_search(
    query: str,
    top_k: int = 10,
    vector_top_k: int = 15,
    bm25_top_k: int = 15,
    rrf_k: int = 60,
) -> list[dict]:
    """Run hybrid retrieval: vector search + BM25, merged with RRF.

    Args:
        query: The user's natural language question
        top_k: How many final fused results to return
        vector_top_k: How many candidates to pull from vector search
        bm25_top_k: How many candidates to pull from BM25
        rrf_k: RRF smoothing constant

    Returns:
        List of result dicts sorted by RRF score (best first).
        Each result has: chunk_id, text, metadata, rrf_score
    """
    # Stage 1a: Vector search — finds semantically similar chunks
    vector_results = vector_search(query, top_k=vector_top_k)

    # Stage 1b: BM25 search — finds keyword-matching chunks
    bm25 = _get_bm25()
    bm25_results = bm25.search(query, top_k=bm25_top_k)

    # Stage 2: Fuse the two ranked lists with RRF
    fused = reciprocal_rank_fusion(
        ranked_lists=[vector_results, bm25_results],
        k=rrf_k,
        top_k=top_k,
    )

    return fused


def reset_bm25() -> None:
    """Force BM25 index to reload on next search.
    Call this after ingesting new documents."""
    global _bm25_retriever
    _bm25_retriever = None
