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
import random
from pathlib import Path


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(set(ranked_ids[:k]) & gold)
    return hit / len(gold)


def coverage_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """1.0 iff EVERY gold id is in the top-k — the 'retrieve ALL of it' signal. Unlike recall@k
    (which is fractional), this is the all-or-nothing metric that matters when a downstream task
    needs the complete set, not most of it."""
    if not gold:
        return 0.0
    return 1.0 if gold <= set(ranked_ids[:k]) else 0.0


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = sum((1.0 / math.log2(i + 2)) for i, pid in enumerate(ranked_ids[:k]) if pid in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return (dcg / ideal) if ideal else 0.0


def graded_ndcg_at_k(ranked_ids: list[str], gains: dict, k: int) -> float:
    """Graded nDCG@k over a {page_id: gain} map. With uniform gain 1.0 this equals ndcg_at_k."""
    dcg = sum(float(gains.get(pid, 0.0)) / math.log2(i + 2) for i, pid in enumerate(ranked_ids[:k]))
    ideal_gains = sorted((float(g) for g in gains.values()), reverse=True)[:k]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return (dcg / ideal) if ideal else 0.0


def average_precision(ranked_ids: list[str], gold: set[str]) -> float:
    """AP = mean of precision@rank at each relevant hit, divided by |gold| (unretrieved gold
    contributes 0). Mean AP across queries is MAP."""
    if not gold:
        return 0.0
    hits, ap = 0, 0.0
    for i, pid in enumerate(ranked_ids, start=1):
        if pid in gold:
            hits += 1
            ap += hits / i
    return ap / len(gold)


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    for i, pid in enumerate(ranked_ids):
        if pid in gold:
            return 1.0 / (i + 1)
    return 0.0


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0,1], over an already-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_vals[int(idx)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def paired_bootstrap(values_a, values_b, *, n_boot: int = 10000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Paired bootstrap over per-query metric values (same queries, two systems). Pairing removes
    query-difficulty variance. Returns the mean delta (b - a), a (1-alpha) CI, a two-sided
    bootstrap p-value, and `significant` (the CI excludes 0 AND there are >= 2 queries — a single
    query can't establish significance). Deterministic given `seed`."""
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("paired_bootstrap needs two equal-length, non-empty samples")
    n = len(values_a)
    deltas = [b - a for a, b in zip(values_a, values_b)]
    obs = sum(deltas) / n
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        boot.append(s / n)
    boot.sort()
    # `+ 0.0` normalizes a rounded -0.0 to 0.0 so the reported CI never spuriously straddles 0.
    ci_low = round(_percentile(boot, alpha / 2), 4) + 0.0
    ci_high = round(_percentile(boot, 1 - alpha / 2), 4) + 0.0
    le = sum(1 for x in boot if x <= 0.0) / n_boot
    ge = sum(1 for x in boot if x >= 0.0) / n_boot
    p = min(1.0, 2.0 * min(le, ge))
    # Derive `significant` from the REPORTED CI (so the flag can never contradict the interval a
    # caller sees) and only with >= 2 paired observations (one query cannot establish it). This
    # is the ship/no-ship criterion; p_value is a complementary approximate two-sided estimate
    # that can differ marginally at the threshold.
    significant = bool(n >= 2 and (ci_low > 0.0 or ci_high < 0.0))
    return {"n": n, "mean_delta": round(obs, 4) + 0.0, "ci_low": ci_low, "ci_high": ci_high,
            "p_value": round(p, 4), "significant": significant}


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


def run_eval(cases: list[dict], retrieve_fn, ks=(1, 5, 10), full_recall_k: int = 100) -> dict:
    """retrieve_fn(query, top_k) -> [(Page, score, page_id)]. Returns a metrics report with, per
    cutoff k: recall@k, graded nDCG@k, coverage@k; plus recall@{full_recall_k},
    coverage@{full_recall_k}, MAP, and MRR. A case may carry an optional {"gold_gains": {id:gain}}
    for graded relevance; otherwise every gold id gets gain 1.0."""
    fetch = max(max(ks), full_recall_k)
    keys = []
    for k in ks:
        keys += [f"recall@{k}", f"ndcg@{k}", f"coverage@{k}"]
    keys += [f"recall@{full_recall_k}", f"coverage@{full_recall_k}", "map", "mrr"]
    agg = {key: [] for key in keys}
    per_query = []
    for c in cases:
        raw_gains = c.get("gold_gains")
        gains = ({pid: float(g) for pid, g in raw_gains.items()} if raw_gains
                 else {pid: 1.0 for pid in c["gold_page_ids"]})
        gold = {pid for pid, g in gains.items() if g > 0}   # relevant = positive gain; a gain-0
        #                                                     entry is a judged-nonrelevant marker
        ranked = [pid for _p, _s, pid in retrieve_fn(c["query"], fetch)]
        row = {"query": c["query"], "map": round(average_precision(ranked, gold), 4),
               "mrr": round(reciprocal_rank(ranked, gold), 4)}
        for k in ks:
            row[f"recall@{k}"] = round(recall_at_k(ranked, gold, k), 4)
            row[f"ndcg@{k}"] = round(graded_ndcg_at_k(ranked, gains, k), 4)
            row[f"coverage@{k}"] = round(coverage_at_k(ranked, gold, k), 4)
        row[f"recall@{full_recall_k}"] = round(recall_at_k(ranked, gold, full_recall_k), 4)
        row[f"coverage@{full_recall_k}"] = round(coverage_at_k(ranked, gold, full_recall_k), 4)
        for key in keys:
            agg[key].append(row[key])
        per_query.append(row)
    means = {key: round(sum(v) / len(v), 4) if v else 0.0 for key, v in agg.items()}
    return {"n": len(cases), "ks": list(ks), "full_recall_k": full_recall_k,
            "means": means, "per_query": per_query}


def compare_runs(cases, retrieve_a, retrieve_b, *, metric: str, ks=(1, 5, 10),
                 full_recall_k: int = 100, n_boot: int = 10000, seed: int = 0) -> dict:
    """Run two retrievers on the SAME cases and paired-bootstrap the per-query delta of `metric`
    (e.g. 'coverage@5', 'map', 'recall@100'). Stops changes shipping on noise: a +0.03 on 30
    queries is usually not significant. Deterministic given `seed`."""
    ra = run_eval(cases, retrieve_a, ks=ks, full_recall_k=full_recall_k)
    rb = run_eval(cases, retrieve_b, ks=ks, full_recall_k=full_recall_k)
    if metric not in ra["means"]:
        raise ValueError(f"unknown metric {metric!r}; available: {sorted(ra['means'])}")
    va = [row[metric] for row in ra["per_query"]]
    vb = [row[metric] for row in rb["per_query"]]
    stats = paired_bootstrap(va, vb, n_boot=n_boot, seed=seed)
    return {"metric": metric, "mean_a": ra["means"][metric], "mean_b": rb["means"][metric], **stats}


def format_report(report: dict) -> str:
    m = report["means"]
    lines = [f"eval over {report['n']} queries", "-" * 32]
    for key in sorted(m):
        lines.append(f"  {key:12} {m[key]:.4f}")
    return "\n".join(lines)
