"""
Reciprocal Rank Fusion (RRF) — merging ranked lists from multiple retrievers.

THE PROBLEM:
We have two ranked lists of chunks:
- Vector search: ranked by cosine similarity (lower distance = better)
- BM25: ranked by term-frequency score (higher = better)

These scores are NOT comparable. A cosine distance of 0.3 and a BM25 score of 4.5
are on completely different scales. We can't just add them or average them.

THE SOLUTION — RRF:
Instead of comparing scores, RRF only looks at RANKS (positions in the list).
For each chunk, its fused score is:

    RRF_score = Σ  1 / (k + rank_i)
              for each list i where the chunk appears

Where k is a smoothing constant (default 60, from the original RRF paper).

WHY k=60?
- If k is too small (e.g., 1), the #1 result gets score 1/2 = 0.5 and #2 gets
  1/3 = 0.33 — a huge gap. This makes the fusion overly sensitive to rank position.
- If k is too large (e.g., 1000), #1 gets 1/1001 and #2 gets 1/1002 — nearly
  identical. This makes all ranks meaningless.
- k=60 is the sweet spot: #1 gets 1/61 ≈ 0.0164, #2 gets 1/62 ≈ 0.0161.
  There's a ranking preference, but it's not extreme.

EXAMPLE:
    Vector search: [ChunkA (#1), ChunkB (#2), ChunkC (#3)]
    BM25:          [ChunkB (#1), ChunkD (#2), ChunkA (#3)]

    ChunkA: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323  ← appears in both!
    ChunkB: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  ← appears in both!
    ChunkC: 1/(60+3)            = 0.0159                      ← only in vector
    ChunkD: 1/(60+2)            = 0.0161                      ← only in BM25

    Final ranking: ChunkB (0.0325), ChunkA (0.0323), ChunkD (0.0161), ChunkC (0.0159)

Chunks that appear in BOTH lists get boosted. This is exactly what we want —
if both keyword search AND semantic search agree a chunk is relevant, it's
very likely to be the right answer.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Merge multiple ranked result lists using RRF.

    Args:
        ranked_lists: List of result lists. Each result list contains dicts
                      with at least a "chunk_id" key. Results must be in
                      rank order (best first).
        k: Smoothing constant (default 60, per the original paper).
        top_k: How many fused results to return.

    Returns:
        Merged list of results, sorted by RRF score (highest first).
        Each result dict gets an added "rrf_score" field.
    """
    # Accumulate RRF scores by chunk_id
    rrf_scores: dict[str, float] = {}
    # Keep track of the full result data for each chunk_id
    # (we take the first occurrence we see)
    chunk_data: dict[str, dict] = {}

    for result_list in ranked_lists:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

            # Store the chunk data if we haven't seen it yet
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # Build final results
    fused_results = []
    for chunk_id in sorted_ids[:top_k]:
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = rrf_scores[chunk_id]
        fused_results.append(result)

    return fused_results
