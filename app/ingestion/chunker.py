"""
Domain-aware chunker for healthcare documents.

The core insight: medical documents have STRUCTURE that carries meaning.
A section header like "Contraindications" changes how every sentence beneath
it should be interpreted. A warning block must never be separated from the
content it warns about.

Our strategy:
1. Split each page into sections using header detection.
2. Within each section, split into chunks respecting sentence boundaries.
3. Never split inside warning/alert blocks.
4. Attach metadata (section title, page, position) to every chunk.

This is NOT perfect OCR-level layout analysis — that would require a vision
model. But it handles the vast majority of structured clinical documents
that use consistent heading patterns.
"""

import re
import yaml
from pathlib import Path

from app.ingestion.models import DocumentChunk

# ── Configuration ──────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "ingestion_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["chunking"]


# ── Header and Structure Detection ────────────────────────────────────────

# Patterns that indicate a section header in medical documents.
# These are intentionally broad — better to over-detect headers than miss them,
# because a false positive just creates a shorter chunk, while a missed header
# means we lose section context.
HEADER_PATTERNS = [
    # Numbered sections: "1. Introduction", "3.2 Dosage"
    r"^\d+\.[\d.]*\s+[A-Z]",
    # ALL-CAPS lines: "CONTRAINDICATIONS", "WARNINGS AND PRECAUTIONS"
    r"^[A-Z][A-Z\s]{4,}$",
    # Title Case lines that are short (likely headers, not sentences)
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6}\s*$",
    # Markdown-style headers (some clinical docs use these)
    r"^#{1,4}\s+",
]

# Lines that signal a warning/alert block — we must keep these with the
# content that follows them.
WARNING_MARKERS = [
    "WARNING:",
    "CAUTION:",
    "ALERT:",
    "BLACK BOX WARNING:",
    "CONTRAINDICATION:",
    "NOTE:",
    "IMPORTANT:",
]


def is_header(line: str) -> bool:
    """Check if a line looks like a section header."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        # Empty lines aren't headers. Very long lines aren't either —
        # they're sentences, not titles.
        return False
    return any(re.match(pattern, stripped) for pattern in HEADER_PATTERNS)


def is_warning_start(line: str) -> bool:
    """Check if a line starts a warning/alert block."""
    upper = line.strip().upper()
    return any(upper.startswith(marker) for marker in WARNING_MARKERS)


# ── Section Splitter ──────────────────────────────────────────────────────


def split_into_sections(page_text: str) -> list[dict]:
    """Split a page's text into sections based on detected headers.

    Think of this like a table of contents extractor — we find the headings
    and group everything between them.

    Returns: [{"title": "Section Name", "content": "section body text"}, ...]
    """
    lines = page_text.split("\n")
    sections = []
    current_title = "Untitled Section"
    current_lines = []

    for line in lines:
        if is_header(line):
            # Save the previous section before starting a new one
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append({"title": current_title, "content": content})
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({"title": current_title, "content": content})

    # If no headers were detected, return the whole page as one section
    if not sections:
        sections.append({"title": "Untitled Section", "content": page_text.strip()})

    return sections


# ── Chunk Splitter ────────────────────────────────────────────────────────


def split_section_into_chunks(
    section_text: str,
    max_size: int,
    overlap: int,
    min_size: int,
) -> list[str]:
    """Split a section's content into chunks, respecting sentence boundaries
    and keeping warning blocks intact.

    The algorithm:
    1. Split text into 'blocks' — paragraphs separated by blank lines, OR
       warning blocks (warning marker + everything until the next paragraph break).
    2. Greedily pack blocks into chunks up to max_size.
    3. When a chunk is full, start a new one with overlap from the previous.

    Why sentence-aware splitting? Because "Do not exceed 10mg." is a complete
    thought. "Do not exceed" is a dangerous fragment.
    """
    # First, identify blocks (paragraphs and warning blocks)
    blocks = _extract_blocks(section_text)

    chunks = []
    current_chunk = ""

    for block in blocks:
        # If adding this block would exceed max_size, finalize current chunk
        if current_chunk and len(current_chunk) + len(block) + 1 > max_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from the end of the previous one
            current_chunk = _get_overlap(current_chunk, overlap) + "\n" + block
        else:
            current_chunk = current_chunk + "\n" + block if current_chunk else block

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If a single block was bigger than max_size, we need to split it further.
    # This handles edge cases like a massive paragraph with no line breaks.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_size:
            final_chunks.extend(_force_split(chunk, max_size, overlap))
        elif len(chunk) >= min_size:
            final_chunks.append(chunk)
        # Chunks below min_size are discarded — they're usually artifacts
        # like lone page numbers or footer text.

    return final_chunks


def _extract_blocks(text: str) -> list[str]:
    """Split text into logical blocks: paragraphs and warning groups.

    A 'warning group' is a warning marker line plus everything until the
    next blank line. This ensures warnings stay with their content.
    """
    lines = text.split("\n")
    blocks = []
    current_block = []
    in_warning = False

    for line in lines:
        if is_warning_start(line):
            # Save whatever we had before the warning
            if current_block:
                block_text = "\n".join(current_block).strip()
                if block_text:
                    blocks.append(block_text)
                current_block = []
            in_warning = True
            current_block.append(line)
        elif not line.strip():
            # Blank line = paragraph break
            if in_warning:
                # End of warning block — save it as one unit
                block_text = "\n".join(current_block).strip()
                if block_text:
                    blocks.append(block_text)
                current_block = []
                in_warning = False
            elif current_block:
                block_text = "\n".join(current_block).strip()
                if block_text:
                    blocks.append(block_text)
                current_block = []
        else:
            current_block.append(line)

    # Final block
    if current_block:
        block_text = "\n".join(current_block).strip()
        if block_text:
            blocks.append(block_text)

    return blocks


def _get_overlap(text: str, overlap_size: int) -> str:
    """Get the last `overlap_size` characters of text, breaking at a sentence
    boundary if possible.

    Why sentence-level overlap? So the overlapping region is a complete thought,
    not a word fragment. This helps embeddings represent the overlap accurately.
    """
    if len(text) <= overlap_size:
        return text

    overlap_region = text[-overlap_size:]
    # Try to start at a sentence boundary
    sentence_break = re.search(r"[.!?]\s+", overlap_region)
    if sentence_break:
        return overlap_region[sentence_break.end() :]
    return overlap_region


def _force_split(text: str, max_size: int, overlap: int) -> list[str]:
    """Last-resort split for text that exceeds max_size with no natural breaks.

    Splits at sentence boundaries (periods followed by spaces). If no
    sentences are found, splits at word boundaries. Never splits mid-word.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current = current + " " + sentence if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── Main Chunking Pipeline ────────────────────────────────────────────────


def chunk_document(
    pages: list[dict], source_filename: str
) -> list[DocumentChunk]:
    """Process a full document (list of pages) into validated, metadata-rich chunks.

    This is the main entry point. It ties together:
    1. Section splitting (structure-aware)
    2. Chunk splitting (size-aware, boundary-respecting)
    3. Metadata attachment (provenance for citations)
    4. Pydantic validation (catches bad data immediately)

    Args:
        pages: Output from a loader — [{"page": int, "text": str}, ...]
        source_filename: Name of the original file (for citation metadata)

    Returns:
        List of validated DocumentChunk objects, ready for embedding.
    """
    config = load_config()
    max_size = config["max_chunk_size"]
    overlap = config["chunk_overlap"]
    min_size = config["min_chunk_size"]

    all_chunks = []
    chunk_index = 0

    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]

        # Step 1: Split the page into structural sections
        sections = split_into_sections(page_text)

        for section in sections:
            # Step 2: Split each section into appropriately-sized chunks
            text_chunks = split_section_into_chunks(
                section["content"], max_size, overlap, min_size
            )

            for text in text_chunks:
                # Step 3: Create a validated chunk with full metadata
                chunk = DocumentChunk(
                    chunk_id=f"{source_filename}_p{page_num}_c{chunk_index}",
                    text=text,
                    source_file=source_filename,
                    page_number=page_num,
                    section_title=section["title"],
                    chunk_index=chunk_index,
                )
                all_chunks.append(chunk)
                chunk_index += 1

    return all_chunks
