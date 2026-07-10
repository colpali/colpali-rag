"""Correctness + unit tests for the v2 accuracy/architecture work. All model-free
(fakes/stubs) so they run fast without downloading a ColPali model. Each pins a
specific verified bug the redesign fixed."""

import json
import types

import pytest
import torch

# --------------------------------------------------------------------------- registry
from colpali_rag import models_registry as R
from colpali_rag.errors import HeatmapUnsupported, IndexModelMismatch, UnsupportedModel


@pytest.mark.parametrize("mid,model_cls,lic", [
    ("vidore/colSmol-500M", "ColIdefics3", "apache-2.0"),
    ("vidore/colqwen2-v1.0", "ColQwen2", "apache-2.0"),        # was mis-routed to ColPali
    ("vidore/colqwen2.5-v0.2", "ColQwen2_5", "research-nc"),   # non-commercial base flagged
    ("ModernVBERT/colmodernvbert", "ColModernVBert", "mit"),
    ("OpenSearch-AI/Ops-Colqwen3-4B", "ColQwen3", "apache-2.0"),
    ("nomic-ai/colnomic-embed-multimodal-7b", "ColQwen2_5", "varies"),  # was mis-routed
    ("vidore/colpali-v1.3", "ColPali", "gemma"),
])
def test_registry_dispatch(mid, model_cls, lic):
    spec = R.resolve(mid)
    assert spec.model_cls == model_cls and spec.license == lic


def test_registry_unknown_raises():
    with pytest.raises(UnsupportedModel):
        R.resolve("acme/not-a-colvision-model")


def test_registry_family_override():
    assert R.resolve("acme/custom", family_override="qwen2").model_cls == "ColQwen2"


def test_registry_contract_against_real_engine():
    """The critic's #1: assert every registered class name exists in the INSTALLED
    colpali-engine (not a fake). Fast — just getattr, no model load."""
    for spec in R.REGISTRY:
        R.load_classes(spec)  # raises EngineCapabilityError if the engine renamed a class


# --------------------------------------------------------------------------- embedder (fakes)
class _Batch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def convert_ids_to_tokens(self, ids):
        return ["what", "colour", "<pad>"][: len(ids)]


class _FakeProcNoMaps:
    """Exposes only get_image_mask + get_n_patches (like ColQwen2) — forces the
    vendored einsum path and must NOT raise AttributeError."""
    def __init__(self, n=4, qlen=3):
        self.n, self.qlen = n, qlen
        self.image_processor = types.SimpleNamespace()
        self.tokenizer = _FakeTokenizer()

    def process_images(self, imgs):
        return _Batch(pixel_values=torch.zeros(1))

    def process_queries(self, qs):
        return _Batch(input_ids=torch.tensor([[10, 11, 12]][: 1]))

    def get_image_mask(self, b):
        return torch.ones(1, self.n).bool()

    def get_n_patches(self, size, **kw):
        return (2, 2)  # nx*ny == n == 4


class _FakeModel:
    def __init__(self, n, dim, qlen):
        self.n, self.dim, self.qlen = n, dim, qlen

    def __call__(self, **b):
        if "input_ids" in b:
            return torch.arange(self.qlen * self.dim, dtype=torch.float32).reshape(1, self.qlen, self.dim)
        return torch.arange(self.n * self.dim, dtype=torch.float32).reshape(1, self.n, self.dim)


def _fake_embedder(proc, model):
    from colpali_rag.embedder import ColpaliEmbedder

    e = object.__new__(ColpaliEmbedder)
    e.model_id, e.device, e.batch_size, e.torch = "fake/model", "cpu", 1, torch
    e.processor, e.model = proc, model
    e.spec = R.resolve("vidore/colqwen2-v1.0")
    e.family = e.spec.family
    return e


def test_similarity_maps_cross_model_no_attribute_error():
    """Pins the P0 heatmap bug: a processor WITHOUT get_similarity_maps_from_embeddings
    (colqwen2/colpali) must still produce maps via the vendored path."""
    from PIL import Image

    e = _fake_embedder(_FakeProcNoMaps(n=4, qlen=3), _FakeModel(4, 8, 3))
    tokens, maps = e.similarity_maps(Image.new("RGB", (40, 30)), "what colour")
    assert tokens and -1 in maps
    grid = maps[tokens[0]["index"]]
    assert len(grid) == 2 and len(grid[0]) == 2   # (ny, nx)


def test_heatmap_unsupported_raises_cleanly():
    from PIL import Image

    class _NoMask:
        tokenizer = _FakeTokenizer()

    e = _fake_embedder(_NoMask(), _FakeModel(4, 8, 3))
    assert e.heatmap_supported is False
    with pytest.raises(HeatmapUnsupported):
        e.similarity_maps(Image.new("RGB", (10, 10)), "x")


def test_embed_pages_strips_padding():
    """Pins the batch>1 pad-folding regression: attention_mask must drop pad tokens."""
    class _Proc:
        def process_images(self, imgs):
            n = len(imgs)
            # 2 images, seq=3, but image 1 has 1 pad token (attn=0)
            return _Batch(pixel_values=torch.zeros(n),
                          attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]])[:n])

    class _M:
        def __call__(self, **b):
            n = b["attention_mask"].shape[0]
            return torch.ones(n, 3, 4)

    e = _fake_embedder(_Proc(), _M())
    e.batch_size = 2
    embs = e.embed_pages(["a", "b"])
    assert embs[0].shape[0] == 3 and embs[1].shape[0] == 2   # pad token dropped from image 2


# --------------------------------------------------------------------------- store identity guard
class _FakeEmbedder:
    model_id = "model-A"
    dim = 4

    def score(self, query, embs):
        return list(embs)

    def page_to_list(self, e):
        return e if isinstance(e, list) else e.tolist()

    def embed_query_raw(self, q):
        return [[0.0] * 4]


def test_index_model_mismatch_guard(tmp_path):
    from PIL import Image

    from colpali_rag.store import MemoryStore
    from colpali_rag.pdf import Page

    recs = [Page("d.pdf", 1, "t")]
    imgs = [Image.new("RGB", (8, 8))]
    MemoryStore(_FakeEmbedder(), str(tmp_path)).build_from(recs, imgs, [[0.1] * 4])

    other = _FakeEmbedder(); other.model_id = "model-B"
    with pytest.raises(IndexModelMismatch):
        MemoryStore.load(other, str(tmp_path))
    # same model reopens fine
    assert len(MemoryStore.load(_FakeEmbedder(), str(tmp_path))) == 1


# --------------------------------------------------------------------------- generator labels
def test_generator_labels_images_for_verifiable_citations(monkeypatch):
    """Pins the citation bug: each image must be preceded by its page label."""
    import httpx

    from colpali_rag import generator as G
    from PIL import Image

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]},
                             request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    imgs = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    out = G.answer("q?", imgs, base_url="http://x/v1", labels=["Page 1 of a.pdf:", "Page 2 of a.pdf:"])
    assert out == "answer"
    content = captured["body"]["messages"][0]["content"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert "Page 1 of a.pdf:" in texts and "Page 2 of a.pdf:" in texts


# --------------------------------------------------------------------------- eval metrics
def test_eval_metrics_math():
    from colpali_rag.eval import ndcg_at_k, recall_at_k, reciprocal_rank, run_eval

    ranked = ["a", "b", "c", "d"]
    gold = {"b"}
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 1.0
    assert reciprocal_rank(ranked, gold) == pytest.approx(0.5)     # gold at rank 2
    assert ndcg_at_k(ranked, gold, 2) == pytest.approx(1 / 1.5849625, rel=1e-4)  # 1/log2(3)
    # run_eval end-to-end with a fake retriever
    cases = [{"query": "q", "gold_page_ids": ["b"]}]
    rep = run_eval(cases, lambda q, k: [("p", 1.0, x) for x in ranked[:k]], ks=(1, 2))
    assert rep["means"]["recall@2"] == 1.0 and rep["n"] == 1


def test_eval_loader_rejects_malformed(tmp_path):
    from colpali_rag.eval import load_eval

    p = tmp_path / "e.jsonl"
    p.write_text('{"query":"q","gold_page_ids":["a"]}\n\n{"query":"only"}\n')
    with pytest.raises(ValueError):
        load_eval(p)


# --------------------------------------------------------------------------- config
def test_config_new_fields(monkeypatch):
    from colpali_rag.config import Settings

    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("COLPALI_DPI", "220")
    monkeypatch.setenv("ANSWER_MIN_SCORE", "12.5")
    monkeypatch.setenv("COLPALI_FAMILY", "qwen2")
    s = Settings.from_env()
    assert s.rerank_enabled is True and s.dpi == 220
    assert s.answer_min_score == 12.5 and s.family == "qwen2"


def test_config_bad_int_raises(monkeypatch):
    from colpali_rag.config import Settings

    monkeypatch.setenv("COLPALI_PORT", "not-a-number")
    with pytest.raises(ValueError):
        Settings.from_env()


# --------------------------------------------------------------------------- rerank
def test_get_reranker_none_when_disabled():
    from colpali_rag.config import Settings
    from colpali_rag.rerank import NoopReranker, get_reranker

    assert get_reranker(Settings()) is None
    assert NoopReranker().rerank("q", [("p", 1.0, "id")], None, 5) == [("p", 1.0, "id")]


def test_app_heatmap_endpoint_returns_501_not_500():
    """Pins the flagship-UX bug: unsupported-model heatmap must be a clean 501."""
    from fastapi import HTTPException
    from PIL import Image

    import colpali_rag.app as A

    class _Emb:
        def similarity_maps(self, im, q):
            raise HeatmapUnsupported("no maps for this model")

    class _Store:
        def get_image(self, pid):
            return Image.new("RGB", (8, 8))

    A._STATE.update(store=_Store(), embedder=_Emb())
    with pytest.raises(HTTPException) as ei:
        A.heatmap("id", "q")
    assert ei.value.status_code == 501


def test_app_ask_503_without_vlm():
    from fastapi import HTTPException

    import colpali_rag.app as A

    A._STATE.update(store=object(), settings=types.SimpleNamespace(vlm_enabled=False))
    with pytest.raises(HTTPException) as ei:
        A.ask("q")
    assert ei.value.status_code == 503


def test_retrieve_applies_reranker():
    from colpali_rag.engine import retrieve

    class _Store:
        def search(self, q, top_k):
            return [("p", float(top_k - i), f"id{i}") for i in range(top_k)]

    class _Rev:
        def rerank(self, q, hits, store, top_k):
            return list(reversed(hits))[:top_k]

    out = retrieve(_Store(), "q", top_k=3, reranker=_Rev())
    assert [pid for _p, _s, pid in out][0] == "id29"  # reversed of first_stage_n=30
