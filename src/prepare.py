"""Recreate Day 6 paragraph chunks locally and verify the source PDF."""

import argparse
import hashlib
import json
from pathlib import Path

from .day6_pipeline.chunking import by_paragraph
from .day6_pipeline.ingestion import load_pdf

ROOT = Path(__file__).resolve().parents[1]


def prepare(pdf: Path = ROOT / "data/CMO-October-2026.pdf") -> list[dict]:
    manifest = json.loads((ROOT / "data/day6_manifest.json").read_text(encoding="utf-8"))
    doc = manifest["documents"][0]
    actual_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if actual_hash != doc["sha256"]:
        raise ValueError(f"PDF hash differs from Day 6 manifest: {actual_hash}")
    pages = load_pdf(pdf, doc["document_id"], doc["source_path_or_url"])
    if len(pages) != doc["physical_page_count"]:
        raise ValueError("PDF page count differs from Day 6 manifest")
    first, last = doc["content_pdf_pages"]["first"], doc["content_pdf_pages"]["last"]
    chunks = []
    for page in pages[first - 1:last]:
        section = next(x["section"] for x in doc["sections"]
                       if x["pdf_page_first"] <= page.page <= x["pdf_page_last"])
        page.metadata.update({"section": section, "year": int(doc["data_cutoff_date"][:4]),
                              "printed_page": page.page - first + 1, "version": doc["version"]})
        for chunk in by_paragraph(page, 900):
            chunks.append({"id": chunk.chunk_id, "source_id": chunk.document_id,
                           "source": chunk.source, "page": chunk.page,
                           "text": chunk.text, "metadata": chunk.metadata})
    if len(chunks) != 340:
        raise ValueError(f"Expected 340 Day 6 B/C chunks, got {len(chunks)}")
    target = ROOT / "data/chunks.json"
    target.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/CMO-October-2026.pdf")
    args = parser.parse_args()
    output = prepare(args.pdf)
    print(f"Wrote {len(output)} verified Day 6 chunks to data/chunks.json")
