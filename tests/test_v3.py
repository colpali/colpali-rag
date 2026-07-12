"""Tests for object storage, structured cited answers, and faithfulness. Model-free
(fakes/stubs). Each pins a correctness point the critique flagged."""

import types

import pytest
from PIL import Image


# --------------------------------------------------------------------------- object storage
def test_local_artifact_store_roundtrip(tmp_path):
    from colpali_rag.artifact_store import LocalArtifactStore
    from colpali_rag.errors import ArtifactStoreError

    s = LocalArtifactStore(tmp_path)
    s.put("images/a.png", b"data", "image/png")
    assert s.get("images/a.png") == b"data" and s.exists("images/a.png")
    assert s.get("images/missing.png") is None      # missing -> None
    assert s.url_for("images/a.png") is None         # local -> app proxies
    s.delete("images/a.png")
    assert not s.exists("images/a.png")
    with pytest.raises(ArtifactStoreError):          # path traversal blocked
        s.get("../secret")


def test_image_key_backcompat():
    """LocalArtifactStore(root=data_dir) + this key == the pre-adapter on-disk path."""
    from colpali_rag.store import _safe, image_key, page_id

    assert image_key(page_id("a.pdf", 3)) == "images/" + _safe("a.pdf::p3") + ".png"


def test_s3_get_semantics_missing_vs_error():
    """404/NoSuchKey -> None; auth/other errors -> raise (never mask as missing)."""
    pytest.importorskip("botocore")
    from botocore.exceptions import ClientError

    from colpali_rag.artifact_store import S3ArtifactStore
    from colpali_rag.errors import ArtifactStoreError

    s = object.__new__(S3ArtifactStore)
    s.bucket, s.prefix = "b", ""

    class _Client:
        def __init__(self, code):
            self.code = code

        def get_object(self, **kw):
            raise ClientError({"Error": {"Code": self.code}}, "GetObject")

    s._client = _Client("NoSuchKey")
    assert s.get("k") is None
    s._client = _Client("AccessDenied")
    with pytest.raises(ArtifactStoreError):
        s.get("k")


# --------------------------------------------------------------------------- structured answers
def test_validate_resolves_bracket_indices():
    from colpali_rag.schemas import validate_answer_obj

    attached = ["a.pdf::p3", "b.pdf::p1"]
    obj = {"answer": "A", "claims": [
        {"text": "c1", "cites": [1], "confidence": 0.9},
        {"text": "c2", "cites": [2, 5], "confidence": 1.5}]}   # 5 is out of range; 1.5 clamps
    res = validate_answer_obj(obj, attached, mode="json_schema")
    assert res.structured and res.claims[0].pages == ["a.pdf::p3"]
    assert res.claims[1].pages == ["b.pdf::p1"]                # index 2 -> b, not "page 2"
    assert 5 in res.hallucinated_citations
    assert res.claims[1].confidence == 1.0


def test_validate_rejects_empty():
    from colpali_rag.schemas import validate_answer_obj

    with pytest.raises(ValueError):
        validate_answer_obj({"answer": "a", "claims": []}, ["x::p1"])


def test_parse_json_from_prose():
    from colpali_rag.schemas import parse_json

    assert parse_json('here you go: ```json\n{"answer":"a","claims":[]}\n```')["answer"] == "a"
    assert parse_json("not json") is None


def _img():
    return Image.new("RGB", (8, 8))


def test_answer_structured_happy(monkeypatch):
    from colpali_rag import generator as G

    def fake_post(base_url, api_key, model, messages, *, response_format=None, **kw):
        return {"choices": [{"message": {"content":
                '{"answer":"A","claims":[{"text":"c","cites":[1],"confidence":0.8}]}'}}]}

    monkeypatch.setattr(G, "_post_chat", fake_post)
    G._CAP_CACHE.clear()
    res = G.answer_structured("q", [_img()], attached_page_ids=["d.pdf::p1"],
                              base_url="http://x/v1", mode="auto")
    assert res.structured and res.mode == "json_schema"
    assert res.claims[0].pages == ["d.pdf::p1"] and res.claims[0].confidence == 0.8


def test_answer_structured_demotes_on_400(monkeypatch):
    import httpx

    from colpali_rag import generator as G

    def fake_post(base_url, api_key, model, messages, *, response_format=None, **kw):
        if response_format and response_format.get("type") == "json_schema":
            req = httpx.Request("POST", "http://x/v1/chat/completions")
            raise httpx.HTTPStatusError("unsupported", request=req, response=httpx.Response(400, request=req))
        return {"choices": [{"message": {"content":
                '{"answer":"A","claims":[{"text":"c","cites":[1],"confidence":0.5}]}'}}]}

    monkeypatch.setattr(G, "_post_chat", fake_post)
    G._CAP_CACHE.clear()
    res = G.answer_structured("q", [_img()], attached_page_ids=["d::p1"],
                              base_url="http://x/v1", mode="auto")
    assert res.structured and res.mode == "json_object"        # demoted past the rejecting tier


def test_answer_structured_fallback_to_free_text(monkeypatch):
    from colpali_rag import generator as G

    monkeypatch.setattr(G, "_post_chat",
                        lambda *a, **k: {"choices": [{"message": {"content": "no json here"}}]})
    G._CAP_CACHE.clear()
    res = G.answer_structured("q", [_img()], attached_page_ids=["d::p1"],
                              base_url="http://x/v1", mode="auto", max_retries=0)
    assert not res.structured and res.claims[0].pages == ["d::p1"]   # never 500s


# --------------------------------------------------------------------------- faithfulness
def test_judge_answer_scores_and_gate(monkeypatch):
    from colpali_rag import faithfulness as F
    from colpali_rag.schemas import Claim, ClaimsResult

    monkeypatch.setattr(F, "_judge_claim",
                        lambda text, imgs, **kw: (("supported", "ok") if "good" in text else ("unsupported", "no")))
    settings = types.SimpleNamespace(judge_base_url="http://j/v1", judge_api_key=None,
                                     judge_model="j", vlm_base_url=None, vlm_api_key=None,
                                     vlm_model="v", judge_allow_same_endpoint=False)
    result = ClaimsResult(answer="A", structured=True,
                          claims=[Claim("good claim", ["d::p1"], 0.9), Claim("bad claim", ["d::p2"], 0.9)])
    rep = F.judge_answer(result, lambda pid: _img(), settings)
    assert rep.faithfulness == 0.5 and rep.citation_precision == 0.5 and rep.unsupported == [1]
    r2, withheld = F.apply_gate(result, rep, "withhold", 0.6)     # 0.5 < 0.6 -> withhold whole answer
    assert withheld and r2.claims == [] and "do not sufficiently support" in r2.answer


def test_faithfulness_off_without_judge_endpoint():
    from colpali_rag import faithfulness as F

    s = types.SimpleNamespace(judge_base_url=None, judge_allow_same_endpoint=False, vlm_base_url=None)
    assert F.judge_answer(object(), lambda p: None, s) is None    # no judge -> off, not faked


def test_apply_gate_flag_hides_nothing():
    from colpali_rag import faithfulness as F
    from colpali_rag.schemas import Claim, ClaimsResult

    res = ClaimsResult(answer="A", structured=True, claims=[Claim("c", ["d::p1"], 0.5)])
    rep = F.FaithfulnessReport([], 0.0, 0.0)
    out, withheld = F.apply_gate(res, rep, "flag", 0.5)
    assert not withheld and out.claims                            # flag never removes claims


# --------------------------------------------------------------------------- review regressions
def test_validate_coerces_scalar_cites():
    """Model returns "cites": 1 (or "2") instead of [1] -> coerced, never TypeError."""
    from colpali_rag.schemas import validate_answer_obj

    attached = ["a.pdf::p3", "b.pdf::p1"]
    obj = {"answer": "A", "claims": [
        {"text": "c1", "cites": 1, "confidence": 0.5},        # bare int
        {"text": "c2", "cites": "2", "confidence": 0.5}]}     # bare string
    res = validate_answer_obj(obj, attached, mode="json_object")
    assert res.claims[0].pages == ["a.pdf::p3"] and res.claims[1].pages == ["b.pdf::p1"]


def test_judge_answer_never_raises_on_image_error():
    """get_image raising (e.g. S3 auth error) must not escape judge_answer -> no 500."""
    from colpali_rag import faithfulness as F
    from colpali_rag.schemas import Claim, ClaimsResult

    def boom(pid):
        raise RuntimeError("storage down")

    settings = types.SimpleNamespace(judge_base_url="http://j/v1", judge_api_key=None,
                                     judge_model="j", vlm_base_url=None, vlm_api_key=None,
                                     vlm_model="v", judge_allow_same_endpoint=False)
    result = ClaimsResult(answer="A", structured=True, claims=[Claim("c", ["d::p1"], 0.9)])
    rep = F.judge_answer(result, boom, settings)                  # must not raise
    assert rep is not None and rep.verdicts[0].verdict == "unverified"


def test_local_artifact_store_rejects_absolute_key(tmp_path):
    """Absolute key would escape root under naive pathlib join -> must be rejected."""
    from colpali_rag.artifact_store import LocalArtifactStore
    from colpali_rag.errors import ArtifactStoreError

    s = LocalArtifactStore(tmp_path)
    with pytest.raises(ArtifactStoreError):
        s.get("/etc/passwd")
    with pytest.raises(ArtifactStoreError):
        s.get("images/../../etc/passwd")


def test_cascade_cache_hit_still_demotes(monkeypatch):
    """A cached tier that starts 400ing must demote through the rest, not fall to free text."""
    import httpx

    from colpali_rag import generator as G

    def fake_post(base_url, api_key, model, msgs, *, response_format=None, **kw):
        if response_format and response_format.get("type") == "json_schema":
            req = httpx.Request("POST", base_url)
            raise httpx.HTTPStatusError("bad", request=req,
                                        response=httpx.Response(400, request=req))
        return {"choices": [{"message": {"content":
                '{"answer":"A","claims":[{"text":"c","cites":[1],"confidence":0.7}]}'}}]}

    monkeypatch.setattr(G, "_post_chat", fake_post)
    G._CAP_CACHE.clear()
    G._CAP_CACHE[("http://x/v1", "vlm")] = "json_schema"          # pin the now-broken tier
    res = G.answer_structured("q", [_img()], attached_page_ids=["d::p1"],
                              base_url="http://x/v1", mode="auto", max_retries=0)
    assert res.structured and res.mode == "json_object"          # demoted, not free text
