"""Index health diagnostics — unit-norm deviation, rank agreement, the backend-agreement probe,
and the adapter-aware identity guard. Model-free."""

import types

import pytest

from colpali_rag.diagnostics import (
    check_unit_norm,
    probe_backend_agreement,
    rank_agreement,
    unit_norm_deviation,
)


def test_unit_norm_deviation_zero_for_unit_vectors():
    unit = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]          # one page, two unit patch vectors
    d = unit_norm_deviation(unit)
    assert d["mean_dev"] == 0.0 and d["max_dev"] == 0.0 and d["n"] == 2
    assert check_unit_norm(unit)[0] is True


def test_unit_norm_deviation_flags_non_unit():
    mv = [[[2.0, 0.0], [0.6, 0.8]]]                       # norms 2.0 (dev 1.0) and 1.0 (dev 0.0)
    d = unit_norm_deviation(mv)
    assert d["max_dev"] == 1.0 and d["mean_dev"] == 0.5
    assert check_unit_norm(mv, tol=1e-3)[0] is False


def test_rank_agreement():
    assert rank_agreement(["a", "b", "c"], ["a", "b", "c"], k=3)["kendall_tau"] == 1.0
    assert rank_agreement(["a", "b", "c"], ["c", "b", "a"], k=3)["kendall_tau"] == -1.0
    assert rank_agreement(["a", "b", "c", "d"], ["a", "b", "x", "y"], k=4)["overlap@k"] == 0.5


def test_probe_backend_agreement():
    class _S:
        def __init__(self, order):
            self.order = order

        def search(self, q, top_k=10):
            return [(None, 1.0, i) for i in self.order[:top_k]]

    a, b = _S(["p1", "p2", "p3"]), _S(["p1", "p2", "p3"])
    rep = probe_backend_agreement(a, b, ["q1", "q2"], k=3)
    assert rep["mean_overlap"] == 1.0 and rep["mean_kendall"] == 1.0
    rep2 = probe_backend_agreement(a, _S(["p3", "p2", "p1"]), ["q"], k=3)
    assert rep2["mean_kendall"] == -1.0


def test_adapter_aware_identity_guard():
    from colpali_rag.errors import IndexModelMismatch
    from colpali_rag.store import check_identity

    base = types.SimpleNamespace(model_id="m", adapter="")
    check_identity({"model": "m", "adapter": "", "schema_version": 1}, base)     # base<->base ok
    check_identity({"model": "m", "schema_version": 1}, base)                     # old index (no key) ok
    with pytest.raises(IndexModelMismatch):                                       # ft index, base query
        check_identity({"model": "m", "adapter": "ft-v1", "schema_version": 1}, base)
    ft = types.SimpleNamespace(model_id="m", adapter="ft-v1")
    check_identity({"model": "m", "adapter": "ft-v1", "schema_version": 1}, ft)   # matching ft ok
    with pytest.raises(IndexModelMismatch):                                       # base index, ft query
        check_identity({"model": "m", "adapter": "", "schema_version": 1}, ft)
