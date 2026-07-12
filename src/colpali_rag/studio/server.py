"""The studio web app: engine index + LLM wrapper behind the studio API, plus the
built React frontend served as static files.

Two ways to run:
  • Dev:  `colpali-rag studio` (backend :8000) + `cd web && npm run dev` (:5173).
          Vite proxies /api -> :8000; CORS below also allows it directly.
  • Prod: `cd web && npm run build` then `colpali-rag studio` — the built SPA in
          web/dist (or $STUDIO_WEB_DIST) is served at :8000.

If no index exists yet, the studio still boots in DEMO mode (structure inferred from the
request text alone) so you can see the UI before indexing anything.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from colpali_rag.config import get_settings
from colpali_rag.engine import open_index
from colpali_rag.studio.api import CTX, router

log = logging.getLogger(__name__)
_LOCK = threading.Lock()   # engine forwards aren't re-entrant on CPU


def _web_dist() -> Path | None:
    candidates = []
    if os.environ.get("STUDIO_WEB_DIST"):
        candidates.append(Path(os.environ["STUDIO_WEB_DIST"]))
    candidates += [
        Path(__file__).resolve().parent / "web_dist",       # packaged build
        Path.cwd() / "web" / "dist",                         # repo-root build
        Path(__file__).resolve().parents[3] / "web" / "dist",
    ]
    for c in candidates:
        if c.is_dir() and (c / "index.html").exists():
            return c
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    CTX["settings"] = s
    CTX["lock"] = _LOCK
    try:
        store, _emb = open_index(s)
        CTX["store"] = store
        try:
            from colpali_rag.rerank import get_reranker
            CTX["reranker"] = get_reranker(s)
        except Exception as e:  # noqa: BLE001 - reranker optional
            CTX["reranker"] = None
            log.info("reranker off: %s", e)
        log.info("studio index: %d page(s)", len(store))
    except Exception as e:  # noqa: BLE001 - no index -> demo mode, not a crash
        CTX["error"] = f"{type(e).__name__}: {e}"
        log.warning("no index (%s) — studio running in DEMO mode", CTX["error"])
    yield


app = FastAPI(title="colpali-studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(router)

_DIST = _web_dist()
if _DIST is not None:
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="web")
else:
    @app.get("/", response_class=HTMLResponse)
    def _placeholder():
        return _PLACEHOLDER_HTML


_PLACEHOLDER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>ColPali Studio</title>
<style>
 body{margin:0;background:#0b1020;color:#e5e7eb;font:15px/1.6 ui-sans-serif,system-ui;
      display:grid;place-items:center;min-height:100vh}
 .card{max-width:640px;padding:40px;background:#0f172a;border:1px solid #1e293b;border-radius:16px}
 h1{margin:0 0 4px;font-size:22px}.tag{color:#38bdf8;font-size:13px;letter-spacing:.08em;text-transform:uppercase}
 code{background:#020617;color:#7dd3fc;padding:2px 7px;border-radius:6px;font-size:13px}
 pre{background:#020617;padding:16px;border-radius:10px;overflow:auto;border:1px solid #1e293b}
 a{color:#38bdf8}
</style></head><body><div class="card">
 <div class="tag">ColPali · Studio</div>
 <h1>Frontend not built yet</h1>
 <p>The API is live at <code>/api/studio/status</code>. Build the UI once:</p>
 <pre>cd web
npm install
npm run dev      # dev UI at http://localhost:5173  (proxies /api here)
# or: npm run build   then reload this page</pre>
 <p>Nothing indexed? The studio still works in <b>demo mode</b> — index a folder
 with <code>colpali-rag index ./pdfs</code> to ground outputs in real pages.</p>
</div></body></html>"""
