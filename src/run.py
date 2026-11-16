"""Day 7 commands: index, compare retrieval, and run a real tool-calling agent."""

import argparse
import json
import time
import unicodedata
from pathlib import Path

from starter.baseline import pack_context
from .agent import run_agent
from .retrieval import Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()


def evidence_hit(row: dict, gold: list[dict]) -> bool:
    return any(row["source_id"] == g["source_id"] and row["page"] == g["page"]
               and normalized(g["evidence"]) in normalized(row["text"]) for g in gold)


def save(name: str, value: object) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def retrieval_eval(retriever: Retriever, cases: list[dict], split: str, method: str,
                   top_k: int, budget: int) -> dict:
    traces = []
    for case in cases:
        if case["split"] != split or not case["answerable"]:
            continue
        start = time.perf_counter()
        results = retriever.search(method, case["question"], top_k=max(3, top_k),
                                   filters=case.get("filter"))
        context, ids = pack_context(results[:top_k], budget)
        marked = [{**r, "evidence_hit": evidence_hit(r, case["gold"])} for r in results]
        traces.append({"query_id": case["id"], "split": split, "method": method,
                       "query": case["question"], "filter": case.get("filter"),
                       "results": marked, "context": context, "context_ids": ids,
                       "hit1": bool(marked and marked[0]["evidence_hit"]),
                       "hit3": any(r["evidence_hit"] for r in marked[:3]),
                       "context_hit": any(r["evidence_hit"] and r["id"] in ids for r in marked),
                       "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)})
    n = len(traces)
    if not n:
        raise ValueError("No answerable cases in split")
    summary = {"split": split, "method": method, "backend": retriever.backend,
               "top_k": top_k, "budget_chars": budget, "N": n,
               "hit1": {"n": sum(t["hit1"] for t in traces), "N": n},
               "hit3": {"n": sum(t["hit3"] for t in traces), "N": n},
               "context_evidence_rate": {"n": sum(t["context_hit"] for t in traces), "N": n},
               "mean_ms": round(sum(t["elapsed_ms"] for t in traces) / n, 2),
               "query_embedding_tokens": retriever.embedder.prompt_tokens if retriever.embedder else 0}
    name = f"retrieval-{split}-{retriever.backend}-{method}-k{top_k}-b{budget}"
    save(f"{name}-traces.json", traces)
    save(f"{name}-summary.json", summary)
    return summary


def agent_eval(retriever: Retriever, cases: list[dict], split: str, method: str,
               model: str, top_k: int, budget: int, max_searches: int) -> dict:
    traces = []
    name = f"agent-{split}-{retriever.backend}-{method}-k{top_k}-b{budget}-s{max_searches}"
    for case in cases:
        if case["split"] != split:
            continue
        try:
            trace = run_agent(case["question"], retriever, method, model,
                              filters=case.get("filter"), top_k=top_k,
                              budget_chars=budget, max_searches=max_searches)
        except Exception as exc:
            trace = {"status": "error", "answer": "", "citations": [], "events": [],
                     "error": f"{type(exc).__name__}: {exc}"}
        trace.update({"query_id": case["id"], "split": split, "question": case["question"],
                      "method": method, "backend": retriever.backend,
                      "top_k": top_k, "budget_chars": budget,
                      "max_searches": max_searches})
        traces.append(trace)
        print(f"{case['id']}: {trace['status']}", flush=True)
        save(f"{name}-traces.json", traces)
    summary = {"split": split, "method": method, "model": model, "N": len(traces),
               "answered": sum(t["status"] == "answered" for t in traces),
               "abstained": sum(t["status"] == "abstain" for t in traces),
               "errors": sum(t["status"] == "error" for t in traces),
               "tool_use": sum(any(e.get("event") == "search" for e in t["events"]) for t in traces),
               "prompt_tokens": sum(t.get("usage", {}).get("prompt_tokens", 0) for t in traces),
               "completion_tokens": sum(t.get("usage", {}).get("completion_tokens", 0) for t in traces)}
    save(f"{name}-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("index", "retrieval", "agent"))
    parser.add_argument("--backend", choices=("local", "pgvector"), default="local")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--method", choices=("dense", "tfidf", "hybrid"), default="dense")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=3000, help="Final context character budget")
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--model", default="openai/gpt-4o-mini-2024-07-18")
    parser.add_argument("--smoke", action="store_true", help="Run only the first dev case without writing a benchmark")
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1 or args.max_searches < 1:
        parser.error("top-k, budget and max-searches must be positive")
    retriever = Retriever(load_chunks(), args.backend)
    if args.stage == "index":
        print(json.dumps({"document_embedding_tokens": retriever.index_local()}))
        return
    cases = json.loads((ROOT / "data/cases.json").read_text(encoding="utf-8"))
    if args.smoke:
        if args.stage != "agent" or args.split != "dev":
            parser.error("--smoke is only allowed for a dev agent run")
        case = next(c for c in cases if c["split"] == "dev" and c["answerable"])
        trace = run_agent(case["question"], retriever, args.method, args.model,
                          filters=case.get("filter"), top_k=args.top_k,
                          budget_chars=args.budget, max_searches=args.max_searches)
        print(json.dumps({"query_id": case["id"], "status": trace["status"],
                          "answer": trace["answer"], "searches": len(trace["events"])},
                         ensure_ascii=False, indent=2))
        return
    if args.stage == "retrieval":
        summary = retrieval_eval(retriever, cases, args.split, args.method,
                                 args.top_k, args.budget)
    else:
        summary = agent_eval(retriever, cases, args.split, args.method, args.model,
                             args.top_k, args.budget, args.max_searches)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
