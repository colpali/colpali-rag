"""Index health checks.

Two silent-failure modes this guards against:

1. **Non-unit-norm embeddings.** The in-memory store scores with a raw dot product; Qdrant scores
   with cosine. They agree *only* when the model's output vectors are already L2-normalized. A
   checkpoint (or a fine-tune) whose head drops the normalization makes the two backends rank
   differently and shifts the score scale under the gates — silently. `unit_norm_deviation`
   measures `mean |‖v‖ - 1|`; `check_unit_norm` turns it into a pass/fail.
2. **Backend disagreement.** `rank_agreement` / `probe_backend_agreement` compare the rankings two
   stores return for the same queries (overlap@k + Kendall tau), so a memory-vs-Qdrant divergence
   shows up as a number instead of a surprise in production.

Pure-Python and dependency-free (works on lists, numpy arrays, or torch tensors), so it can be
unit-tested without a model.
"""

from __future__ import annotations

import math


def _rows(multivector):
    """Yield each patch vector (as a sequence of floats) from a multivector that may be a torch
    tensor, a numpy array, or a plain list-of-lists."""
    if hasattr(multivector, "tolist"):
        multivector = multivector.tolist()
    return iter(multivector)


def unit_norm_deviation(multivectors, *, sample: int = 4000) -> dict:
    """`mean` and `max` of `|‖v‖ - 1|` over up to `sample` patch vectors. 0.0 == perfectly
    unit-norm. `multivectors` is an iterable of per-page multivectors."""
    devs: list[float] = []
    for mv in multivectors:
        for row in _rows(mv):
            n = math.sqrt(sum(float(x) * float(x) for x in row))
            devs.append(abs(n - 1.0))
            if len(devs) >= sample:
                break
        if len(devs) >= sample:
            break
    if not devs:
        return {"mean_dev": 0.0, "max_dev": 0.0, "n": 0}
    return {"mean_dev": sum(devs) / len(devs), "max_dev": max(devs), "n": len(devs)}


def check_unit_norm(multivectors, *, tol: float = 1e-3, sample: int = 4000):
    """Return (ok, stats). ok is True when mean deviation from unit norm is within `tol`."""
    stats = unit_norm_deviation(multivectors, sample=sample)
    return (stats["mean_dev"] <= tol), stats


def rank_agreement(a_ids, b_ids, k: int = 10) -> dict:
    """Agreement between two ranked id lists (best-first): overlap@k and a Kendall-tau over the
    items common to both top-k. tau=1.0 == same relative order; -1.0 == fully reversed."""
    a, b = list(a_ids)[:k], list(b_ids)[:k]
    sb = set(b)
    overlap = (len(set(a) & sb) / k) if k else 0.0
    common = [x for x in a if x in sb]
    rank_b = {x: i for i, x in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if rank_b[common[i]] < rank_b[common[j]]:
                conc += 1
            else:
                disc += 1
    tau = ((conc - disc) / (conc + disc)) if (conc + disc) else 1.0
    return {"overlap@k": round(overlap, 4), "kendall_tau": round(tau, 4),
            "k": k, "n_common": len(common)}


def probe_backend_agreement(store_a, store_b, queries, k: int = 10) -> dict:
    """Run the same queries against two stores and average their ranking agreement. Use it to
    confirm the in-memory and Qdrant backends rank a corpus the same way. Each store needs a
    `.search(query, top_k) -> [(record, score, id)]`."""
    per_query = []
    for q in queries:
        a_ids = [pid for _r, _s, pid in store_a.search(q, top_k=k)]
        b_ids = [pid for _r, _s, pid in store_b.search(q, top_k=k)]
        per_query.append(rank_agreement(a_ids, b_ids, k=k))
    n = len(per_query) or 1
    return {
        "queries": len(per_query),
        "mean_overlap": round(sum(r["overlap@k"] for r in per_query) / n, 4),
        "mean_kendall": round(sum(r["kendall_tau"] for r in per_query) / n, 4),
        "per_query": per_query,
    }
