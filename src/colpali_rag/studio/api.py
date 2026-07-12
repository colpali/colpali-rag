"""FastAPI routes for the studio. Mounted by studio.server on an app whose lifespan
loads the engine store into `CTX`. Stateless-ish: durable data is the engine index +
object storage; sessions are the ephemeral working set.

Routes (all under /api/studio):
  POST /session                 -> {session_id}
  GET  /sources                 -> selectable sources (doc -> page count)
  POST /upload                  -> ingest a CSV / Excel / text file into a session
  GET  /session/{sid}           -> session working set (selected docs, tables, notes, history)
  POST /select                  -> set which sources apply
  POST /generate                -> request -> cited structured output
  GET  /export                  -> last output as .mmd (mermaid) or .drawio
  GET  /image                   -> proxy a source page image (for citation preview)
"""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response

from colpali_rag.studio.generate import generate_diagram
from colpali_rag.studio.session import SessionStore

log = logging.getLogger(__name__)

# Populated by studio.server's lifespan. store/settings may be None if no index exists;
# the studio still runs in demo mode (diagrams from the request text alone).
CTX: dict = {"store": None, "settings": None, "lock": None, "reranker": None, "error": None}
SESSIONS = SessionStore()

router = APIRouter(prefix="/api/studio")


def _store():
    return CTX.get("store")


@router.get("/status")
def status():
    s = CTX.get("settings")
    store = _store()
    return {
        "ready": True,                                  # studio runs even with no index (demo mode)
        "index": store is not None and len(store) > 0,
        "pages": len(store) if store is not None else 0,
        "llm": bool(getattr(s, "vlm_enabled", False)),
        "mode": "llm" if getattr(s, "vlm_enabled", False) else "demo",
        "index_error": CTX.get("error"),
    }


@router.post("/session")
def new_session():
    return {"session_id": SESSIONS.create().id}


@router.get("/sources")
def sources():
    store = _store()
    if store is None:
        return {"docs": [], "pages": 0, "note": CTX.get("error") or "no index — run: colpali-rag index <dir>"}
    counts: dict[str, int] = {}
    for r in store.records:
        counts[r.doc] = counts.get(r.doc, 0) + 1
    return {"docs": [{"doc": d, "pages": n} for d, n in sorted(counts.items())],
            "pages": len(store)}


def _session_dict(sess):
    return {
        "session_id": sess.id,
        "selected_docs": sorted(sess.selected_docs),
        "tables": [{"name": t.name, "rows": t.total_rows, "columns": t.columns,
                    "sheet": t.sheet} for t in sess.tables],
        "notes": [{"name": n.name, "chars": len(n.text)} for n in sess.notes],
        "history": [{"request": t.request, "title": (t.spec or {}).get("title")}
                    for t in sess.history],
    }


@router.get("/session/{sid}")
def get_session(sid: str):
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "unknown session")
    return _session_dict(sess)


@router.post("/select")
def select(session_id: str = Form(...), docs: str = Form("")):
    """Set which datasheets apply (comma-separated doc ids; empty = all)."""
    sess = SESSIONS.get_or_create(session_id)
    sess.selected_docs = {d.strip() for d in docs.split(",") if d.strip()}
    return _session_dict(sess)


@router.post("/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    sess = SESSIONS.get_or_create(session_id)
    data = await file.read()
    if len(data) > 8_000_000:
        raise HTTPException(413, "file too large (8 MB limit)")
    try:
        note = sess.add_upload(file.filename or "upload", data)
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e
    return {"status": note, "session": _session_dict(sess)}


@router.post("/generate")
def generate(session_id: str = Form(...), message: str = Form(...),
             docs: str | None = Form(None), top_k: int = Form(6)):
    if not message.strip():
        raise HTTPException(422, "empty request")
    sess = SESSIONS.get_or_create(session_id)
    if docs is not None:
        sess.selected_docs = {d.strip() for d in docs.split(",") if d.strip()}
    s = CTX.get("settings")
    spec, srcs = generate_diagram(
        message, store=_store(), settings=s,
        selected_docs=sess.selected_docs or None,
        tables=sess.tables, notes=sess.notes,
        reranker=CTX.get("reranker"), lock=CTX.get("lock"), top_k=max(1, min(top_k, 12)))
    sess.last_spec = spec
    sess.last_sources = srcs
    from colpali_rag.studio.session import Turn
    payload = spec.to_dict(srcs)
    sess.history.append(Turn(request=message, spec=payload))
    return {"session_id": sess.id, "spec": payload, "sources": srcs}


@router.get("/export")
def export(session_id: str, fmt: str = Query("drawio", pattern="^(drawio|mermaid|mmd)$")):
    from colpali_rag.studio import render

    sess = SESSIONS.get(session_id)
    if sess is None or getattr(sess, "last_spec", None) is None:
        raise HTTPException(404, "no diagram to export yet")
    if fmt == "drawio":
        xml = render.to_drawio(sess.last_spec)
        return Response(xml, media_type="application/xml",
                        headers={"Content-Disposition": 'attachment; filename="diagram.drawio"'})
    return PlainTextResponse(render.to_mermaid(sess.last_spec),
                             headers={"Content-Disposition": 'attachment; filename="diagram.mmd"'})


@router.get("/image")
def image(page_id: str):
    """Proxy a datasheet page image so the UI can show what a citation points at."""
    store = _store()
    if store is None:
        raise HTTPException(503, "no index loaded")
    im = store.get_image(page_id)
    if im is None:
        raise HTTPException(404, "page image not found")
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")
