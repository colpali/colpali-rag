"""An OpenAI-compatible chat endpoint over the ColPali index.

This is what lets a polished, off-the-shelf chat UI — Open WebUI, LibreChat, Continue, the OpenAI
SDK — drive this project: point the client at `http://<host>:8000/v1` and it sees ColPali as a
"model" it can chat with. Retrieval is ColPali visual search over your page images; the answer is
read from the top pages by the configured vision model (if any), with page citations appended.

Two model ids are exposed:
  * `colpali-rag`     — grounded Q&A over the indexed documents.
  * `colpali-diagram` — a structured diagram of what the sources describe, returned as Mermaid
                        (which Open WebUI renders inline in the chat).

Mounted on the `serve` app, so it shares the loaded index / vision model. No API key required
(any Bearer token is accepted), so the client's "OpenAI API key" field can be anything.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(prefix="/v1")

ANSWER_MODEL = "colpali-rag"
DIAGRAM_MODEL = "colpali-diagram"


def _state():
    """Serve-app state (store, settings, reranker). Imported lazily to avoid an import cycle."""
    from colpali_rag.app import _STATE

    return _STATE


def _lock():
    """The serve app's forward lock (CPU model forwards aren't re-entrant). Lazy import."""
    from colpali_rag.app import _LOCK

    return _LOCK


@router.get("/models")
def list_models():
    created = int(time.time())
    return {"object": "list", "data": [
        {"id": ANSWER_MODEL, "object": "model", "created": created, "owned_by": "colpali-rag"},
        {"id": DIAGRAM_MODEL, "object": "model", "created": created, "owned_by": "colpali-rag"},
    ]}


def _last_user_text(messages) -> str:
    """The latest user turn as plain text (handles OpenAI's multimodal content arrays)."""
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return (c or "").strip()
    return ""


def _answer(query: str) -> str:
    """Grounded answer for `colpali-rag`: retrieve top pages, let the vision model read them if
    configured (else return the ranked pages), and append verifiable citations."""
    st = _state()
    store, s = st.get("store"), st.get("settings")
    if store is None:
        return "No index is loaded. On the server run `colpali-rag index <folder-of-pdfs>` first."

    from colpali_rag.engine import retrieve

    top_k = (s.answer_top_k if s else 5) or 5
    with _lock():
        hits = retrieve(store, query, top_k=top_k, reranker=st.get("reranker"), settings=s)
    if not hits:
        return "No matching pages were found in the index for that question."

    sources = [(r.doc, r.page, pid, round(sc, 3)) for r, sc, pid in hits]
    if s and s.vlm_enabled:
        imgs, labels = [], []
        for r, _sc, pid in hits:
            im = store.get_image(pid)
            if im is not None:
                imgs.append(im)
                labels.append(f"Page {r.page} of {r.doc}:")
        if imgs:
            from colpali_rag.generator import answer as vlm_answer
            try:
                text = vlm_answer(query, imgs, base_url=s.vlm_base_url, api_key=s.vlm_api_key,
                                  model=s.vlm_model, labels=labels)
            except Exception as e:  # noqa: BLE001 - surface endpoint errors in the chat, don't 500
                text = f"_(answer model error: {type(e).__name__}: {e})_"
        else:
            text = "_(no page images available to read)_"
    else:
        text = ("**Top matching pages** — set `VLM_BASE_URL` on the server for a written, "
                "cited answer read from these pages:")
    cites = "\n".join(f"- {doc} · p.{pg}  ·  score {sc}" for doc, pg, _pid, sc in sources)
    return f"{text}\n\n**Sources**\n{cites}"


def _diagram(query: str) -> str:
    """Structured diagram for `colpali-diagram`, returned as a Mermaid block (Open WebUI renders
    it inline) plus the model's reasoning."""
    st = _state()
    store, s = st.get("store"), st.get("settings")
    try:
        from colpali_rag.studio.generate import generate_diagram
        from colpali_rag.studio.render import to_mermaid

        spec, _sources = generate_diagram(query, store=store, settings=s,
                                          reranker=st.get("reranker"), lock=_lock())
        mermaid = to_mermaid(spec)
    except Exception as e:  # noqa: BLE001
        return f"_(diagram generation failed: {type(e).__name__}: {e})_"
    note = spec.reasoning.strip() if getattr(spec, "reasoning", "") else ""
    return f"{note}\n\n```mermaid\n{mermaid}\n```".strip()


def _reply_text(model: str, query: str) -> str:
    if not query:
        return "Ask a question about your indexed documents."
    return _diagram(query) if model == DIAGRAM_MODEL else _answer(query)


@router.post("/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    model = body.get("model") or ANSWER_MODEL
    query = _last_user_text(body.get("messages"))
    stream = bool(body.get("stream", False))
    created = int(time.time())
    text = _reply_text(model, query)
    if stream:
        return StreamingResponse(_sse(model, text, created), media_type="text/event-stream")
    return JSONResponse(_completion(model, text, created))


# ---- OpenAI response shapes ---------------------------------------------------
def _completion(model: str, text: str, created: int) -> dict:
    return {
        "id": "chatcmpl-colpali", "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chunk(model: str, created: int, delta: dict, finish=None) -> str:
    payload = {"id": "chatcmpl-colpali", "object": "chat.completion.chunk", "created": created,
               "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(payload)}\n\n"


def _sse(model: str, text: str, created: int):
    yield _chunk(model, created, {"role": "assistant"})
    for i in range(0, len(text), 64):                    # stream in small slices
        yield _chunk(model, created, {"content": text[i:i + 64]})
    yield _chunk(model, created, {}, finish="stop")
    yield "data: [DONE]\n\n"
