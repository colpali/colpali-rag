"""OpenAI-compatible /v1 endpoint — model-free (fake store + patched retrieve)."""

import threading
import types


def _client(monkeypatch, vlm=False):
    from fastapi.testclient import TestClient

    import colpali_rag.app as appmod
    import colpali_rag.engine as engine
    from colpali_rag.pdf import Page

    hits = [(Page("mcu.pdf", 3, "vcc pinout"), 0.91, "mcu.pdf::p3"),
            (Page("sensor.pdf", 1, "i2c"), 0.44, "sensor.pdf::p1")]
    monkeypatch.setattr(engine, "retrieve", lambda *a, **k: hits)

    store = types.SimpleNamespace(get_image=lambda pid: None)
    appmod._STATE = {"store": store, "error": None, "reranker": None,
                     "settings": types.SimpleNamespace(answer_top_k=5, vlm_enabled=vlm)}
    appmod._LOCK = threading.Lock()
    return TestClient(appmod.app)


def test_v1_models_lists_both(monkeypatch):
    c = _client(monkeypatch)
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "colpali-rag" in ids and "colpali-diagram" in ids


def test_chat_completion_grounds_and_cites(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/chat/completions", json={
        "model": "colpali-rag",
        "messages": [{"role": "user", "content": "where is VCC?"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "colpali-rag"
    content = body["choices"][0]["message"]["content"]
    assert "mcu.pdf" in content and "p.3" in content            # retrieved page cited
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_sse(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/chat/completions", json={
        "model": "colpali-rag", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: " in r.text and "[DONE]" in r.text
    assert '"delta"' in r.text and "chat.completion.chunk" in r.text


def test_multimodal_content_array_is_read(monkeypatch):
    # OpenAI clients may send content as a list of parts; we extract the text
    c = _client(monkeypatch)
    r = c.post("/v1/chat/completions", json={
        "model": "colpali-rag",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "vcc?"}]}],
    })
    assert r.status_code == 200 and "mcu.pdf" in r.json()["choices"][0]["message"]["content"]
