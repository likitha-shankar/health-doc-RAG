"""
BM25 sparse retriever for keyword-based search.

BM25 (Best Matching 25) is a bag-of-words ranking function. For each document,
it scores how well it matches the query by looking at:
1. Term Frequency (TF): How often does each query word appear in the document?
   More occurrences → higher score, but with diminishing returns.
2. Inverse Document Frequency (IDF): How rare is each query word across ALL
   documents? Rare words (like "empagliflozin") get more weight than common
   words (like "the" or "patient").
3. Document Length normalization: Longer documents naturally contain more words,
   so BM25 adjusts for that to avoid unfairly favoring long chunks.

BM25 is NOT a neural model — it doesn't "understand" meaning. It's pure
word-counting with smart math. But it's extremely good at finding exact
matches, which is exactly what vector search struggles with.

We build the BM25 index over our stored chunks so it can be queried alongside
the vector store.
"""

import re
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.models import DocumentChunk

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Why not use a medical tokenizer? For BM25, simple tokenization works well.
    BM25 cares about word presence, not linguistic structure. Lowercasing
    ensures "Metformin" matches "metformin". Splitting on non-alphanumerics
    handles punctuation and special characters.

    We keep numbers intact because lab values (e.g., "126", "7.0") and
    dosages (e.g., "500mg") are critical search terms in healthcare.
    """
    text = text.lower()
    # Split on non-alphanumeric characters, keeping numbers and letters
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class BM25Retriever:
    """BM25 keyword retriever over document chunks.

    Usage:
        retriever = BM25Retriever.from_chunks(chunks)
        results = retriever.search("empagliflozin dosage", top_k=10)
    """

    def __init__(self, chunks: list[DocumentChunk], bm25: BM25Okapi):
        self._chunks = chunks
        self._bm25 = bm25

    @classmethod
    def from_chunks(cls, chunks: list[DocumentChunk]) -> "BM25Retriever":
        """Build a BM25 index from a list of DocumentChunks.

        This tokenizes every chunk and builds the BM25 index in memory.
        BM25 indices are lightweight (just word counts) — even 100k chunks
        would use minimal memory compared to vector embeddings.
        """
        tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"[BM25] Index built over {len(chunks)} chunks")
        return cls(chunks, bm25)

    @classmethod
    def from_processed_dir(cls) -> "BM25Retriever":
        """Build BM25 index from all processed chunk JSON files.

        This loads the same chunks we stored in Phase 1's ingestion pipeline,
        so BM25 and vector search are always operating over the same data.
        Keeping them in sync is critical — if one has chunks the other doesn't,
        RRF fusion will produce inconsistent results.
        """
        all_chunks = []
        for json_file in sorted(PROCESSED_DIR.glob("*.chunks.json")):
            with open(json_file) as f:
                data = json.load(f)
                chunks = [DocumentChunk(**item) for item in data]
                all_chunks.extend(chunks)
                print(f"[BM25] Loaded {len(chunks)} chunks from {json_file.name}")

        if not all_chunks:
            raise FileNotFoundError(
                f"No chunk files found in {PROCESSED_DIR}. Run ingestion first."
            )

        return cls.from_chunks(all_chunks)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for chunks matching the query using BM25 scoring.

        Args:
            query: The user's question
            top_k: Number of results to return

        Returns:
            List of dicts with: chunk_id, text, metadata, score
            Sorted by BM25 score (highest first).
        """
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get indices of top-k scores (argsort gives ascending, so we reverse)
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            score = float(scores[idx])
            # Skip zero-score results — they have no term overlap with the query
            if score <= 0:
                continue
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                },
                "score": score,
            })

        return results
