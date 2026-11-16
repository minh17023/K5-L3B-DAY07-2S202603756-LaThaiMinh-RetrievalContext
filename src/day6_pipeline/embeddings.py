"""Real OpenRouter embeddings with one model for documents and queries."""

import json
import os
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Call OpenRouter's embedding endpoint; never substitute mock vectors."""

    def __init__(self, model_name: str, dimensions: int = 1536) -> None:
        if not model_name or dimensions <= 0:
            raise ValueError("A model name and positive dimensions are required")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing from starter/.env")
        self.model_name = model_name
        self.dimensions = dimensions
        self._api_key = api_key
        self.prompt_tokens = 0
        self.requests = 0

    def _embed(self, texts: list[str], input_type: str | None) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        payload: dict[str, object] = {
            "model": self.model_name,
            "input": texts,
            "dimensions": self.dimensions,
        }
        if input_type:
            payload["input_type"] = input_type
        request = Request(
            "https://openrouter.ai/api/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=60) as response:
                    result = json.load(response)
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(f"Embedding API returned HTTP {exc.code}") from exc
                time.sleep(2**attempt)
            except URLError as exc:
                raise RuntimeError("Embedding API connection failed") from exc
        data = result.get("data", [])
        if len(data) != len(texts):
            raise RuntimeError("Embedding API returned an unexpected number of vectors")
        data.sort(key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError("Embedding dimensions do not match the configured value")
        self.prompt_tokens += int(result.get("usage", {}).get("prompt_tokens", 0))
        self.requests += 1
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        input_type = os.getenv("EMBEDDING_DOCUMENT_INPUT_TYPE") or None
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            vectors.extend(self._embed(texts[start : start + 32], input_type))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        input_type = os.getenv("EMBEDDING_QUERY_INPUT_TYPE") or None
        return self._embed([question], input_type)[0]
