"""Page-bounded fixed-size and sentence/paragraph-aware chunking."""

import re

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split one page into overlapping character windows."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    if not page.text.strip():
        return []
    chunks: list[Chunk] = []
    step = size - overlap
    for start in range(0, len(page.text), step):
        piece = page.text[start : start + size].strip()
        if piece:
            chunks.append(_chunk(page, f"fixed-{size}-{overlap}", len(chunks), piece))
        if start + size >= len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack complete paragraphs or sentences without crossing page boundaries."""
    if max_size <= 0:
        raise ValueError("Require max_size > 0")
    units = _units(page.text, max_size)
    chunks: list[Chunk] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > max_size:
            chunks.append(_chunk(page, f"paragraph-{max_size}", len(chunks), current))
            current = ""
        current = f"{current} {unit}" if current else unit
    if current:
        chunks.append(_chunk(page, f"paragraph-{max_size}", len(chunks), current))
    return chunks


def _chunk(page: DocumentPage, kind: str, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{page.document_id}:{page.page}:{kind}:{index}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=dict(page.metadata),
    )


def _units(text: str, max_size: int) -> list[str]:
    """Prefer paragraph, then sentence, then word boundaries."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= max_size:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
        for sentence in sentences:
            if len(sentence) <= max_size:
                units.append(sentence)
                continue
            current = ""
            for word in sentence.split():
                if len(word) > max_size:
                    if current:
                        units.append(current)
                        current = ""
                    units.extend(word[i : i + max_size] for i in range(0, len(word), max_size))
                elif len(current) + len(word) + bool(current) <= max_size:
                    current = f"{current} {word}" if current else word
                else:
                    units.append(current)
                    current = word
            if current:
                units.append(current)
    return units
