"""Exact cosine search in PostgreSQL + pgvector using a pure-Python driver."""

import json
import math
from contextlib import closing, contextmanager
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote, urlsplit

import pg8000.dbapi

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, model_name: str = "") -> None:
        """Retain a PostgreSQL URL, fixed vector width, and model identity."""
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname or not parsed.path.strip("/"):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL with host and database")
        if not 1 <= dimensions <= 2000:
            raise ValueError("pgvector vector dimensions must be between 1 and 2000")
        self._connection_args = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "database": unquote(parsed.path.lstrip("/")),
            "timeout": 5,
        }
        self.dimensions = dimensions
        self.model_name = model_name

    @contextmanager
    def _connection(self) -> Iterator[pg8000.dbapi.Connection]:
        connection = pg8000.dbapi.connect(**self._connection_args)
        try:
            yield connection
        finally:
            connection.close()

    def _vector(self, values: list[float]) -> str:
        if len(values) != self.dimensions or any(not math.isfinite(value) for value in values):
            raise ValueError("Vector length or values are invalid")
        return "[" + ",".join(str(float(value)) for value in values) + "]"

    def create_schema(self) -> None:
        """Create a dedicated table; fail if an existing table has another width."""
        with self._connection() as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if cursor.fetchone() is None:
                    raise RuntimeError("pgvector extension is not enabled")
                cursor.execute(
                    f"""CREATE TABLE IF NOT EXISTS lab_chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        section TEXT,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        model_name TEXT NOT NULL,
                        embedding VECTOR({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )"""
                )
                cursor.execute(
                    """SELECT format_type(a.atttypid, a.atttypmod)
                       FROM pg_attribute a
                       WHERE a.attrelid = 'lab_chunks'::regclass
                         AND a.attname = 'embedding'"""
                )
                actual = cursor.fetchone()[0]
                if actual != f"vector({self.dimensions})":
                    raise RuntimeError(f"Existing lab_chunks uses {actual}, expected vector({self.dimensions})")
                cursor.execute("CREATE INDEX IF NOT EXISTS lab_chunks_section_idx ON lab_chunks (strategy, section)")
            connection.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Atomically replace one strategy, including obsolete chunks."""
        if strategy not in {"A", "B", "C"}:
            raise ValueError("Strategy must be A, B or C")
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("Need one vector per nonempty chunk")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within a strategy")
        serialized = [self._vector(vector) for vector in vectors]
        with self._connection() as connection:
            try:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("DELETE FROM lab_chunks WHERE strategy = %s", (strategy,))
                    for chunk, vector in zip(chunks, serialized, strict=True):
                        cursor.execute(
                            """INSERT INTO lab_chunks
                               (strategy, chunk_id, document_id, source, page, section,
                                content, metadata, model_name, embedding)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS jsonb),
                                       %s, CAST(%s AS vector))""",
                            (
                                strategy, chunk.chunk_id, chunk.document_id, chunk.source,
                                chunk.page, chunk.metadata.get("section"), chunk.text,
                                json.dumps(chunk.metadata, ensure_ascii=False), self.model_name,
                                vector,
                            ),
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Apply an allowlisted section filter in SQL, then rank by cosine distance."""
        if strategy not in {"A", "B", "C"} or top_k <= 0:
            raise ValueError("Invalid strategy or top_k")
        if filters and set(filters) != {"section"}:
            raise ValueError("Only the source-derived section filter is supported")
        params: list[Any] = [self._vector(query_vector), strategy, self.model_name]
        sql = """SELECT chunk_id, document_id, source, page, content, metadata,
                        embedding <=> CAST(%s AS vector) AS distance
                 FROM lab_chunks WHERE strategy = %s AND model_name = %s"""
        if filters:
            sql += " AND section = %s"
            params.append(filters["section"])
        sql += " ORDER BY distance ASC LIMIT %s"
        params.append(top_k)
        with self._connection() as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5] if isinstance(row[5], dict) else json.loads(row[5]),
                "score": 1 - float(row[6]),
            }
            for row in rows
        ]

    def count(self, strategy: str) -> int:
        """Number of chunks currently stored for the configured model."""
        with self._connection() as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM lab_chunks WHERE strategy = %s AND model_name = %s",
                    (strategy, self.model_name),
                )
                return int(cursor.fetchone()[0])
