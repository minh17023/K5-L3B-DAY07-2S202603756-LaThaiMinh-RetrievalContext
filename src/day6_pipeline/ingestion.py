"""Extract physical PDF pages while preserving traceable source locations."""

import re
import unicodedata
import logging
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Read every physical page; extraction failures are visible in metadata."""
    if not path.is_file():
        raise FileNotFoundError(path)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(path)
    pages: list[DocumentPage] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        metadata: dict[str, str | bool] = {}
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except Exception as exc:  # A damaged page must not make its location disappear.
            text = ""
            metadata["extraction_error"] = f"{type(exc).__name__}: {exc}"
        if not text:
            metadata["empty_text"] = True
        pages.append(DocumentPage(document_id, source, number, text, metadata))
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and line wrapping, retaining explicit paragraph breaks."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\u00ad", "")
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = [" ".join(part.split()) for part in paragraphs]
    return "\n\n".join(part for part in cleaned if part)
