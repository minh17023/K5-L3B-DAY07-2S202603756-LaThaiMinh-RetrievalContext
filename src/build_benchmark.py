"""Build the Day 7 benchmark from Day 6 dev labels and new PDF evidence."""

import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "world-bank-cmo-2026-10"


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()


NEW_TEST = [
    ("t01", "How many oil reserve barrels did IEA members agree in early October to release through January 2027?",
     "100 million barrels, initially focused on diesel.", 10,
     "release 100 million barrels of oil reserves through January 2027, initially focused on diesel"),
    ("t02", "What growth in global natural gas production is forecast for 2027, in percent and bcm?",
     "About 2.5 percent, or 110 bcm.", 29,
     "Global production is forecast to increase by about 2.5 percent (110 bcm) in 2027"),
    ("t03", "How large are total grain supplies projected to be in 2026–27 relative to the record and two years earlier?",
     "The second largest on record and about 3 percent higher than two years earlier.", 35,
     "total grain supplies (production and beginning stocks combined) in 2026-27 are projected to be the second largest on record and about 3 percent higher than two years earlier"),
    ("t04", "What year-on-year copper price increase is projected for 2026 before declines in 2027 and 2028?",
     "33 percent year on year.", 50,
     "After a projected 33 percent (y/y) increase to a record annual average in 2026, copper prices are forecast to decline by 2 percent in 2027 and 4 percent in 2028"),
    ("t05", "In the Special Focus, which commodity groups are net transmitters and which are net receivers of price volatility?",
     "Industrial metals and food are net transmitters; livestock and soft commodities are net receivers.", 58,
     "Industrial metals and food are net transmitters, livestock and soft commodities are net receivers"),
]

MISSING = [
    ("d_missing_01", "dev", "What Brent crude price does this report forecast for 2030, in dollars per barrel?",
     "The report's price forecasts cover 2026–28, not an exact Brent forecast for 2030."),
    ("d_missing_02", "dev", "What monthly cocoa price does the report forecast for July 2030?",
     "The report has no month-specific cocoa forecast for July 2030."),
    ("t_missing_01", "test", "What exact urea price does the report forecast for December 2031?",
     "The report has no month-specific urea forecast for December 2031."),
    ("t_missing_02", "test", "What lithium price does the report forecast for the year 2032?",
     "The report has no lithium price forecast for 2032."),
]


def build() -> tuple[list[dict], list[dict]]:
    chunks = json.loads((ROOT / "data/chunks.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (ROOT / "data/day6_questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = []
    for row in rows:
        if row["split"] != "dev":
            continue  # The old Day 6 test results are already visible.
        answerable.append({"id": row["id"], "split": "dev", "question": row["question"],
                           "expected_answer": row["expected_answer"], "answerable": True,
                           "gold": [{"source_id": row["gold_source_id"], "page": row["gold_page"],
                                     "evidence": row["evidence_quote"]}], "filter": row.get("filter")})
    for qid, question, answer, page, evidence in NEW_TEST:
        answerable.append({"id": qid, "split": "test", "question": question,
                           "expected_answer": answer, "answerable": True,
                           "gold": [{"source_id": SOURCE_ID, "page": page, "evidence": evidence}],
                           "filter": None})
    missing = []
    for qid, split, question, reason in MISSING:
        missing.append({"id": qid, "split": split, "question": question,
                        "answerable": False, "expected_behavior": "search_and_abstain",
                        "absence_reason": reason, "filter": None})
    for case in answerable:
        for gold in case["gold"]:
            if not any(c["source_id"] == gold["source_id"] and c["page"] == gold["page"]
                       and normalize(gold["evidence"]) in normalize(c["text"]) for c in chunks):
                # A Day 6 quote may straddle a paragraph chunk; preserve it but make
                # that validation failure explicit instead of manufacturing a hit.
                case["gold_not_in_single_chunk"] = True
    all_cases = answerable + missing
    assert len([c for c in all_cases if c["split"] == "dev"]) == 17
    assert len([c for c in all_cases if c["split"] == "test"]) == 7
    (ROOT / "data/questions_answerable.json").write_text(json.dumps(answerable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "data/cases.json").write_text(json.dumps(all_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return answerable, all_cases


if __name__ == "__main__":
    answerable, all_cases = build()
    print(f"Wrote {len(answerable)} answerable and {len(all_cases)} total cases")
    print("Gold crossing chunk boundary:", [q["id"] for q in answerable if q.get("gold_not_in_single_chunk")])
