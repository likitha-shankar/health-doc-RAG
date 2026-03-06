"""
Document ingestion pipeline — the single entry point for processing documents.

Usage:
    from app.ingestion.pipeline import ingest
    chunks = ingest("data/raw/clinical_guideline.pdf")

This function:
1. Loads the document (PDF or text)
2. Chunks it with domain-aware splitting
3. Saves the chunks as JSON for inspection and debugging
4. Returns validated DocumentChunk objects

Why save to JSON? Two reasons:
- Debugging: you can visually inspect chunks to verify quality before embedding.
- Reproducibility: if you change your chunking strategy, you can compare outputs.
"""

import json
from pathlib import Path

from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_document
from app.ingestion.models import DocumentChunk

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def ingest(file_path: str, save_json: bool = True) -> list[DocumentChunk]:
    """Ingest a document end-to-end: load → chunk → save → return.

    Args:
        file_path: Path to the document (.pdf or .txt)
        save_json: Whether to save chunks as JSON for debugging

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    file_path = str(Path(file_path).resolve())
    source_filename = Path(file_path).name

    print(f"[Ingestion] Loading: {source_filename}")
    pages = load_document(file_path)
    print(f"[Ingestion] Extracted {len(pages)} page(s)")

    print(f"[Ingestion] Chunking with domain-aware splitter...")
    chunks = chunk_document(pages, source_filename)
    print(f"[Ingestion] Created {len(chunks)} chunks")

    if save_json:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIR / f"{source_filename}.chunks.json"
        chunks_data = [chunk.model_dump() for chunk in chunks]
        with open(output_path, "w") as f:
            json.dump(chunks_data, f, indent=2)
        print(f"[Ingestion] Saved chunks to {output_path}")

    # Print a preview of the first few chunks so you can spot-check quality
    print(f"\n{'='*60}")
    print(f"CHUNK PREVIEW (first 3 of {len(chunks)})")
    print(f"{'='*60}")
    for chunk in chunks[:3]:
        print(f"\n--- Chunk {chunk.chunk_index} ---")
        print(f"  Section: {chunk.section_title}")
        print(f"  Page:    {chunk.page_number}")
        print(f"  Length:  {len(chunk.text)} chars")
        print(f"  Text:    {chunk.text[:150]}...")

    return chunks
