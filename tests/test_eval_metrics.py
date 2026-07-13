"""Retrieval-eval extensions — coverage@k, MAP, graded nDCG, and the paired bootstrap.
Model-free and deterministic (the bootstrap seed is passed in, never wall-clock)."""

from colpali_rag.eval import (
    average_precision,
    compare_runs,
    coverage_at_k,
    graded_ndcg_at_k,
    ndcg_at_k,
    paired_bootstrap,
    run_eval,
)


def test_coverage_is_all_or_nothing():
    assert coverage_at_k(["a", "b", "c"], {"a", "b"}, 2) == 1.0     # every gold in top-2
    assert coverage_at_k(["a", "c", "b"], {"a", "b"}, 2) == 0.0     # one gold at rank 3
    assert coverage_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert coverage_at_k(["a"], set(), 5) == 0.0


def test_average_precision():
    assert average_precision(["a", "x", "b", "y"], {"a", "b"}) == (1.0 + 2 / 3) / 2  # hits @1,@3
    assert average_precision(["x", "y"], {"a"}) == 0.0             # gold never retrieved


def test_graded_ndcg_reduces_to_binary_and_grades():
    ranked = ["a", "b", "c"]
    assert graded_ndcg_at_k(ranked, {"b": 1.0}, 3) == ndcg_at_k(ranked, {"b"}, 3)
    assert 0.0 < graded_ndcg_at_k(["c", "x", "a"], {"a": 3.0, "c": 1.0}, 3) < 1.0  # high-gain low = penalized


def test_run_eval_reports_new_metrics():
    ranked = ["a", "b", "c", "d"]
    cases = [{"query": "q", "gold_page_ids": ["b", "d"]}]
    rep = run_eval(cases, lambda q, k: [("p", 1.0, x) for x in ranked[:k]], ks=(1, 2), full_recall_k=4)
    m = rep["means"]
    assert m["coverage@2"] == 0.0 and m["coverage@4"] == 1.0       # d only reachable within top-4
    assert m["recall@4"] == 1.0 and "map" in m


def test_paired_bootstrap_significance_and_determinism():
    r1 = paired_bootstrap([0.0] * 20, [1.0] * 20, n_boot=2000, seed=7)
    r2 = paired_bootstrap([0.0] * 20, [1.0] * 20, n_boot=2000, seed=7)
    assert r1 == r2                                                # same seed -> identical
    assert r1["mean_delta"] == 1.0 and r1["significant"] is True and r1["p_value"] < 0.05
    flat = paired_bootstrap([0.5] * 20, [0.5] * 20, n_boot=2000, seed=7)
    assert flat["significant"] is False                           # no real difference


def test_compare_runs_ab():
    cases = [{"query": f"q{i}", "gold_page_ids": ["g"]} for i in range(15)]
    worse = lambda q, k: [("p", 1.0, "x")]                        # never finds gold
    better = lambda q, k: [("p", 1.0, "g")]                       # always ranks gold first
    cmp = compare_runs(cases, worse, better, metric="mrr", n_boot=1000, seed=1)
    assert cmp["mean_a"] == 0.0 and cmp["mean_b"] == 1.0
    assert cmp["mean_delta"] == 1.0 and cmp["significant"] is True


def test_gold_gains_zero_gain_is_nonrelevant():
    # 'a' judged non-relevant (gain 0), 'b' relevant (gain 2); retrieval returns only 'b'
    rep = run_eval([{"query": "q", "gold_gains": {"a": 0.0, "b": 2.0}}],
                   lambda q, k: [("p", 1.0, x) for x in ["b", "z"]][:k], ks=(2,), full_recall_k=2)
    m = rep["means"]
    assert m["recall@2"] == 1.0 and m["coverage@2"] == 1.0 and m["map"] == 1.0 and m["ndcg@2"] == 1.0


def test_gold_gains_id_coercion_consistent():
    f = lambda q, k: [("p", 1.0, i) for i in [7, 8, 9]][:k]        # non-string ids
    a = run_eval([{"query": "q", "gold_page_ids": [7]}], f, ks=(1,), full_recall_k=3)["means"]["recall@1"]
    b = run_eval([{"query": "q", "gold_gains": {7: 1.0}}], f, ks=(1,), full_recall_k=3)["means"]["recall@1"]
    assert a == b == 1.0                                           # both paths agree on int ids


def test_bootstrap_single_query_not_significant():
    r = paired_bootstrap([0.1], [0.9], n_boot=500, seed=1)
    assert r["n"] == 1 and r["significant"] is False              # one query can't establish it


def test_bootstrap_significant_never_contradicts_ci():
    for seed in range(6):
        a = [0.9] * 6 + [0.1] * 6
        b = [0.1] * 6 + [0.9] * 6
        r = paired_bootstrap(a, b, n_boot=800, seed=seed)
        straddles = r["ci_low"] <= 0.0 <= r["ci_high"]
        assert r["significant"] == (not straddles)               # flag agrees with the reported CI
        assert "-0.0" not in (repr(r["ci_low"]) + repr(r["ci_high"]))  # no negative zero
