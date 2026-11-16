"""Checks that the Day 6 evidence contract and Day 7 tool loop stay honest."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent import run_agent
from src.retrieval import Retriever, load_chunks
from src.run import evidence_hit

ROOT = Path(__file__).resolve().parents[1]


class Day7Tests(unittest.TestCase):
    def test_benchmark_split_and_gold_resolve_to_day6_chunks(self):
        chunks = load_chunks()
        cases = json.loads((ROOT / "data/cases.json").read_text(encoding="utf-8"))
        self.assertEqual((len(chunks), len(cases)), (340, 24))
        self.assertEqual((sum(c["split"] == "dev" for c in cases),
                          sum(c["split"] == "test" for c in cases)), (17, 7))
        for case in cases:
            if case["answerable"]:
                self.assertTrue(any(evidence_hit(c, case["gold"]) for c in chunks), case["id"])

    def test_tfidf_returns_real_chunk_ids(self):
        retriever = Retriever(load_chunks(), "local")
        hits = retriever.search("tfidf", "global natural gas production", top_k=3)
        self.assertTrue(hits)
        self.assertEqual([r["rank"] for r in hits], list(range(1, len(hits) + 1)))
        self.assertTrue(all(r["id"] in retriever.by_id for r in hits))

    def test_agent_calls_tool_then_cites_context(self):
        chunk = {"id": "doc:1:chunk", "text": "The price rose by 3 percent.",
                 "source_id": "doc", "source": "report.pdf", "page": 1,
                 "metadata": {}, "score": 0.8, "rank": 1}

        class FakeRetriever:
            backend = "local"
            def search(self, method, query, top_k=3, filters=None):
                return [chunk]

        messages = [
            ({"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
              "function": {"name": "search_report", "arguments": '{"query":"price"}'}}]}, {}),
            ({"role": "assistant", "content": "The price rose by 3 percent [doc:1:chunk]."}, {}),
        ]
        with patch("src.agent._chat", side_effect=messages):
            result = run_agent("How much?", FakeRetriever(), "tfidf", "test-model")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], ["doc:1:chunk"])
        self.assertEqual(len(result["events"]), 1)

    def test_agent_rejects_citation_outside_context(self):
        class EmptyRetriever:
            def search(self, method, query, top_k=3, filters=None):
                return []
        messages = [
            ({"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
              "function": {"name": "search_report", "arguments": '{"query":"missing"}'}}]}, {}),
            ({"role": "assistant", "content": "Claim [fake]."}, {}),
        ]
        with patch("src.agent._chat", side_effect=messages):
            result = run_agent("Unknown?", EmptyRetriever(), "tfidf", "test-model")
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
