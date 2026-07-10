"""Retrieval evaluation — so accuracy is measured, not guessed.

Metrics (pure-Python, no extra dependency): recall@k, nDCG@k, MRR over a labeled set
of (query -> relevant page ids). The gold ids reuse store.page_id() so they line up
exactly with what search() returns; the whole indexed corpus is the negative pool per
query (ViDoRe convention).

Data format — eval.jsonl, one object per line:
    {"query": "...", "gold_page_ids": ["report.pdf::p3", ...]}

Usage:
    colpali-rag eval run --eval eval.jsonl --k 1,5,10 [--rerank] [--report report.json]

Honest scope: the numbers are only as good as your labeled set. A model-free unit
test checks the metric MATH on tiny fixtures; a real retrieval number requires a real
index + a real labeled set (an offline job, not the fast CI suite).
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(set(ranked_ids[:k]) & gold)
    return hit / len(gold)


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    dcg = sum((1.0 / math.log2(i + 2)) for i, pid in enumerate(ranked_ids[:k]) if pid in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return (dcg / ideal) if ideal else 0.0


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    for i, pid in enumerate(ranked_ids):
        if pid in gold:
            return 1.0 / (i + 1)
    return 0.0


def load_eval(path) -> list[dict]:
    """Load eval.jsonl; skips blank lines, raises on malformed JSON with the line no."""
    cases = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
        if "query" not in obj or "gold_page_ids" not in obj:
            raise ValueError(f"{path}:{n}: each line needs 'query' and 'gold_page_ids'")
        cases.append(obj)
    return cases


def run_eval(cases: list[dict], retrieve_fn, ks=(1, 5, 10)) -> dict:
    """retrieve_fn(query, top_k) -> [(Page, score, page_id)]. Returns a metrics report."""
    maxk = max(ks)
    per_query, agg = [], {f"recall@{k}": [] for k in ks}
    for k in ks:
        agg[f"ndcg@{k}"] = []
    agg["mrr"] = []
    for c in cases:
        gold = set(c["gold_page_ids"])
        ranked = [pid for _p, _s, pid in retrieve_fn(c["query"], maxk)]
        row = {"query": c["query"], "mrr": round(reciprocal_rank(ranked, gold), 4)}
        for k in ks:
            row[f"recall@{k}"] = round(recall_at_k(ranked, gold, k), 4)
            row[f"ndcg@{k}"] = round(ndcg_at_k(ranked, gold, k), 4)
        for key in agg:
            agg[key].append(row[key])
        per_query.append(row)
    means = {key: round(sum(v) / len(v), 4) if v else 0.0 for key, v in agg.items()}
    return {"n": len(cases), "ks": list(ks), "means": means, "per_query": per_query}


def format_report(report: dict) -> str:
    m = report["means"]
    lines = [f"eval over {report['n']} queries", "-" * 32]
    for key in sorted(m):
        lines.append(f"  {key:12} {m[key]:.4f}")
    return "\n".join(lines)
