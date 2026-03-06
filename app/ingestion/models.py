"""
Data models for document chunks.

Why Pydantic here? In production, bad data is silent poison. A chunk missing
its source filename or page number means a citation that reads "Unknown source"
— useless in healthcare. Pydantic validates every chunk at creation time, so
we catch problems at ingestion, not at query time days later.
"""

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single chunk of text extracted from a healthcare document.

    Every field here exists because downstream components need it:
    - chunk_id: unique identifier for deduplication and citation linking
    - text: the actual content that gets embedded and searched
    - source_file: which document this came from (for citations)
    - page_number: where in the document (for citations)
    - section_title: the heading this chunk falls under (for context and filtering)
    - chunk_index: ordering within the document (for reconstructing context)
    """

    chunk_id: str = Field(description="Unique ID: {filename}_{page}_{chunk_index}")
    text: str = Field(description="The chunk content", min_length=1)
    source_file: str = Field(description="Original filename")
    page_number: int = Field(description="1-indexed page number", ge=1)
    section_title: str = Field(
        default="Untitled Section",
        description="Nearest section header above this chunk",
    )
    chunk_index: int = Field(
        description="0-indexed position of this chunk in the document", ge=0
    )
