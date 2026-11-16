"""Seed an empty Day 6 strategy C pgvector index from verified Day 6 data."""

import gzip
import json
import os

from .day6_pipeline.models import Chunk
from .day6_pipeline.store import PgVectorStore
from .prepare import prepare
from .retrieval import DIMENSIONS, EMBEDDING_MODEL, ROOT, load_chunks, load_environment


def seed() -> int:
    load_environment()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    chunks = load_chunks()
    if prepare() != chunks:
        raise ValueError("Saved chunks differ from hash-verified Day 6 PDF")
    cache = ROOT / "data/vectors.json.gz"
    with gzip.open(cache, "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    if saved.get("model") != EMBEDDING_MODEL or saved.get("ids") != [c["id"] for c in chunks]:
        raise ValueError("Embedding cache does not match Day 6 chunks/model")
    vectors = saved.get("vectors")
    if not isinstance(vectors, list) or len(vectors) != len(chunks):
        raise ValueError("Embedding cache has the wrong number of vectors")
    store = PgVectorStore(database_url, DIMENSIONS, EMBEDDING_MODEL)
    store.create_schema()
    existing = store.count("C")
    if existing:
        raise RuntimeError(f"Strategy C already contains {existing} chunks; refusing to overwrite")
    records = [Chunk(chunk_id=c["id"], document_id=c["source_id"], source=c["source"],
                     page=c["page"], text=c["text"], metadata=c["metadata"])
               for c in chunks]
    store.upsert("C", records, vectors)
    actual = store.count("C")
    if actual != len(chunks):
        raise RuntimeError(f"Expected {len(chunks)} indexed chunks, found {actual}")
    return actual


if __name__ == "__main__":
    print(json.dumps({"strategy": "C", "indexed_chunks": seed(), "model": EMBEDDING_MODEL}))
