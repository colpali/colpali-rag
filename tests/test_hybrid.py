"""Hybrid visual + lexical retrieval — the lexical BM25 channel, RRF fusion, coverage gating,
and a measured A/B. Model-free: a fake store supplies a fixed (deliberately weak) visual ranking
so the lexical channel and the fusion are what's under test, with no torch involved."""

import types

from colpali_rag.engine import _rrf, retrieve
from colpali_rag.lexical import LexicalIndex
from colpali_rag.pdf import Page
from colpali_rag.store import page_id

# distinct codes (no shared n-grams) so a bare-code query isolates its one page lexically
CODES = ["AX-100", "BQ-217", "C7-350", "DK-902", "EN-118", "FZ-543", "GT-806", "HM-274"]


class _FakeStore:
    """Its visual search returns a FIXED order (page i at rank i), independent of the query —
    a stand-in for 'the visual channel didn't surface the right page at the top'."""

    def __init__(self, pages, visual_order):
        self.records = [Page(doc=d, page=p, text=t) for d, p, t in pages]
        self.ids = [page_id(r.doc, r.page) for r in self.records]
        self._vo = visual_order

    def search(self, query, top_k=12):
        order = self._vo[:top_k]
        return [(self.records[i], float(len(order) - rank), self.ids[i])
                for rank, i in enumerate(order)]


def _pages(scanned=False):
    return [("doc.pdf", i + 1, ("" if scanned else f"unit {c} datasheet: rated voltage, pinout."))
            for i, c in enumerate(CODES)]


def _cfg(on, **kw):
    base = dict(hybrid_enabled=on, hybrid_kappa=60, hybrid_fetch=100, hybrid_min_coverage=0.5,
                hybrid_ngram_min=3, hybrid_ngram_max=5)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_lexical_index_ranks_exact_id_first():
    docs = list(zip([page_id("d", i + 1) for i in range(len(CODES))],
                    [f"unit {c} datasheet" for c in CODES]))
    res = LexicalIndex(docs, ngram=(3, 5)).search("DK-902", top_k=3)
    assert res and res[0][0] == page_id("d", 4)          # DK-902 is the 4th doc


def test_rrf_pulls_up_a_lexical_hit():
    # 'g' is visual-rank 4 but lexical-rank 1 -> fused above the visual leader 'a'
    fused = _rrf([["a", "b", "c", "g"], ["g"]], kappa=60, top_k=2)
    assert fused[0][0] == "g"


def test_hybrid_recovers_exact_id_the_visual_channel_missed():
    store = _FakeStore(_pages(), visual_order=list(range(len(CODES))))
    gold = page_id("doc.pdf", 6)                          # FZ-543 -> page 6, poorly ranked visually
    assert retrieve(store, "FZ-543", 1, settings=_cfg(False))[0][2] != gold
    assert retrieve(store, "FZ-543", 1, settings=_cfg(True))[0][2] == gold


def test_hybrid_degrades_to_visual_on_scanned_corpus():
    store = _FakeStore(_pages(scanned=True), visual_order=list(range(len(CODES))))
    hyb = retrieve(store, "FZ-543", 3, settings=_cfg(True))
    vis = retrieve(store, "FZ-543", 3, settings=_cfg(False))
    assert [h[2] for h in hyb] == [v[2] for v in vis]    # no text -> lexical off -> identical


def test_hybrid_ab_is_a_significant_win():
    from colpali_rag.eval import compare_runs

    store = _FakeStore(_pages(), visual_order=list(range(len(CODES))))
    cases = [{"query": c, "gold_page_ids": [page_id("doc.pdf", i + 1)]} for i, c in enumerate(CODES)]
    visual = lambda q, k: retrieve(store, q, k, settings=_cfg(False))
    hybrid = lambda q, k: retrieve(store, q, k, settings=_cfg(True))
    cmp = compare_runs(cases, visual, hybrid, metric="coverage@1", ks=(1,), n_boot=1000, seed=3)
    assert cmp["mean_b"] > cmp["mean_a"] and cmp["significant"] is True


def test_lexical_ngram_lo_clamped():
    # lo=0 must be clamped to >=1 so empty-string grams don't match EVERY document; a doc that
    # shares no characters with the query must not appear (the empty-gram bug would surface it)
    idx = LexicalIndex([("a", "apple"), ("b", "zzzzz")], ngram=(0, 2))
    hits = dict(idx.search("xyz", 10))
    assert "a" not in hits                               # 'apple' shares no chars with 'xyz'
    assert idx.lo == 1                                   # clamped


def test_hybrid_gives_reranker_a_wide_pool():
    store = _FakeStore(_pages(), visual_order=list(range(len(CODES))))
    seen = {}

    class _RR:
        def rerank(self, query, hits, store, top_k):
            seen["n"] = len(hits)                        # candidates the reranker received
            return [hits[-1]] + hits[:-1]                # promote the deepest candidate

    out = retrieve(store, "AX-100", top_k=2, reranker=_RR(), settings=_cfg(True))
    assert seen["n"] == len(CODES)                       # wide pool (whole corpus), not just top_k=2
    assert out[0][2] == store.ids[-1]                    # the reranker's promotion survived
