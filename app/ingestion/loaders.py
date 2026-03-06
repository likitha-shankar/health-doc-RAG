"""
Document loaders: extract raw text from PDFs and plain text files.

Why separate loaders? Different file formats need different parsing strategies.
PDFs have pages (critical for citation page numbers). Plain text doesn't.
Keeping loaders modular means adding DOCX or HTML support later is just one
new function — no changes to the chunking logic.
"""

from pathlib import Path

import fitz  # PyMuPDF


def load_pdf(file_path: str) -> list[dict]:
    """Extract text from a PDF, one entry per page.

    We use PyMuPDF (fitz) because it's fast, handles complex layouts well,
    and gives us page-level granularity. Page numbers are essential for
    citations — "see page 12" is actionable, "see document.pdf" is not.

    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        # Skip blank pages — they add noise and waste embedding compute
        if text.strip():
            pages.append({"page": page_num, "text": text.strip()})
    doc.close()
    return pages


def load_text(file_path: str) -> list[dict]:
    """Load a plain text file as a single 'page'.

    Plain text has no concept of pages, so we treat the whole file as page 1.
    This keeps the interface consistent — downstream code always gets
    [{"page": int, "text": str}] regardless of file format.
    """
    text = Path(file_path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [{"page": 1, "text": text.strip()}]


def load_document(file_path: str) -> list[dict]:
    """Route to the correct loader based on file extension.

    This is the single entry point for ingestion. The rest of the pipeline
    never needs to know what format the original document was in.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_text(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Supported: .pdf, .txt"
        )
