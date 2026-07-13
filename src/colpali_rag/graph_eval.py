"""Structured-output evaluation — vocabulary adherence and required coverage.

The retrieval side is scored by `eval.py`. This scores the *structured output*: given a set
of node labels a model proposed and the labels that survived projection onto a closed
vocabulary, how faithful was the model, and did the required items make it through?

- **P_adh** (adherence) — fraction of RAW proposed nodes that resolve to the vocabulary. Scored
  on the raw labels, because after projection every surviving node is in the vocabulary by
  construction (adherence is then tautologically 1.0). Measuring the raw labels is what makes
  this a real quality signal for the generator.
- **HPR** (hallucinated-part rate) = 1 - P_adh. The number to drive to zero.
- **C_req** (required coverage) — fraction of the required set present in the emitted (post-
  projection) output. Reported *alongside* P_adh so that "win by dropping everything" stays
  visible: dropping a required item raises adherence but sinks C_req. It is measured on the
  canonical ids that survived, never on the raw labels (which may be non-canonical).

Deliberately dependency-pure: it takes plain collections and an `accept` predicate, so it never
imports the catalog machinery (or anything domain-specific). A caller wires a compiled
`Catalog.accept` in. Neutral, generic, no external data.
"""

from __future__ import annotations


def adherence(raw_labels, accept) -> float:
    """P_adh: fraction of raw labels that resolve to the vocabulary (via `accept`)."""
    labels = list(raw_labels)
    if not labels:
        return 0.0
    return sum(1 for l in labels if accept(l)) / len(labels)


def hallucinated_part_rate(raw_labels, accept) -> float:
    """HPR = 1 - P_adh."""
    return 1.0 - adherence(raw_labels, accept)


def required_coverage(present, required) -> float:
    """C_req: fraction of the required set present among `present`. Compare like with like —
    pass canonical/verbatim ids on both sides. Empty required set => 1.0 (nothing to cover)."""
    req = set(required)
    if not req:
        return 1.0
    return len(req & set(present)) / len(req)


def graph_report(raw_labels, repaired_ids, *, accept, required) -> dict:
    """One structured-output scorecard.

    raw_labels    — node labels the model proposed (pre-projection).
    repaired_ids  — canonical ids that survived projection (the emitted diagram's nodes).
    accept        — predicate: does a raw label resolve to the vocabulary?
    required      — the required id set (Req).

    P_adh/HPR score the RAW output; C_req scores the emitted (canonical) output. The "delta"
    the constraint buys is HPR on the raw labels falling to zero on the emitted diagram (every
    surviving node is in the vocabulary by construction) while C_req shows nothing required was
    dropped to get there.
    """
    raw = list(raw_labels)
    rep = list(repaired_ids)
    return {
        "n": len(raw),
        "p_adh": round(adherence(raw, accept), 4),
        "hpr": round(hallucinated_part_rate(raw, accept), 4),
        "c_req": round(required_coverage(rep, required), 4),
        "n_dropped": len(raw) - len(rep),
    }


def format_graph_report(r: dict) -> str:
    return (
        f"nodes={r['n']}  P_adh={r['p_adh']:.4f}  HPR={r['hpr']:.4f}  "
        f"C_req={r['c_req']:.4f}  dropped={r['n_dropped']}"
    )
