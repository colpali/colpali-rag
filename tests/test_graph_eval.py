"""Structured-output metric tests — model-free, deterministic, generic ids only.

Demonstrates the committable before/after "eval delta": raw model output has a nonzero
hallucinated-part rate; after projection every emitted node is in the vocabulary, and C_req
is reported alongside so adherence can't be gamed by dropping required items.
"""

from colpali_rag.graph_eval import (
    adherence,
    format_graph_report,
    graph_report,
    hallucinated_part_rate,
    required_coverage,
)
from colpali_rag.studio.catalog import Catalog, canon


def _cat(ids, required=()):
    return Catalog(keys={canon(i): i for i in ids}, canonical=set(ids), required=set(required))


def _repaired(cat, raw):
    out = []
    for label in raw:
        m = cat.match(label)
        if m.status in ("exact", "remap"):
            out.append(m.canonical)
    return out


def test_primitives():
    cat = _cat(["AX-100", "BX-200"])
    assert adherence(["AX-100", "ZX-9"], cat.accept) == 0.5
    assert hallucinated_part_rate(["AX-100", "ZX-9"], cat.accept) == 0.5
    assert adherence([], cat.accept) == 0.0
    assert required_coverage(["AX-100"], set()) == 1.0        # nothing required -> covered
    assert required_coverage(["AX-100"], {"AX-100", "BX-200"}) == 0.5


def test_before_after_delta():
    V = ["AX-100", "AX-101", "BX-200", "BX-201", "CX-300"]
    req = {"AX-100", "BX-200"}
    cat = _cat(V, req)
    raw = ["AX-100", "BX-200", "ZX-999"]                       # one hallucinated node
    repaired = _repaired(cat, raw)
    assert repaired == ["AX-100", "BX-200"]

    r = graph_report(raw, repaired, accept=cat.accept, required=req)
    assert r == {"n": 3, "p_adh": 0.6667, "hpr": 0.3333, "c_req": 1.0, "n_dropped": 1}
    # the delta the change buys: HPR 0.3333 -> 0.0 while C_req holds at 1.0
    assert hallucinated_part_rate(repaired, cat.accept) == 0.0
    assert "HPR=0.3333" in format_graph_report(r)


def test_c_req_exposes_dropped_required():
    V = ["AX-100", "BX-200", "CX-300"]
    req = {"AX-100", "BX-200"}
    cat = _cat(V, req)
    raw = ["AX-100", "ZX-999"]                                 # BX-200 (required) never proposed
    repaired = _repaired(cat, raw)
    assert repaired == ["AX-100"]

    r = graph_report(raw, repaired, accept=cat.accept, required=req)
    assert r["hpr"] == 0.5                                     # 1 of 2 raw nodes hallucinated
    assert r["c_req"] == 0.5                                   # missing required item stays visible
