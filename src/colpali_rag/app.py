"""FastAPI service + self-contained visual UI for colpali-rag.

The model + index are loaded once at startup (lifespan). Endpoints:
  GET /                 -> the single-page UI (no build step, no CDN)
  GET /api/status       -> model/index info
  GET /api/search       -> ranked pages for a query
  GET /api/image        -> a page PNG
  GET /api/heatmap      -> per-token similarity overlays for a page + query
"""

from __future__ import annotations

import io
import logging
import threading
from contextlib import asynccontextmanager

_log = logging.getLogger(__name__)

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from colpali_rag import heatmap as H
from colpali_rag.config import get_settings
from colpali_rag.engine import open_index, retrieve
from colpali_rag.errors import HeatmapUnsupported
from colpali_rag.rerank import get_reranker

_STATE: dict = {"store": None, "embedder": None, "info": None, "error": None}
_LOCK = threading.Lock()  # serialize model forwards (CPU model isn't re-entrant)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    _STATE["settings"] = s
    try:
        store, emb = open_index(s)
        _STATE["store"], _STATE["embedder"] = store, emb
        try:
            _STATE["reranker"] = get_reranker(s)
        except Exception as e:  # noqa: BLE001 - reranker is optional; never block startup
            _STATE["reranker"] = None
            _STATE["rerank_error"] = f"{type(e).__name__}: {e}"
        _STATE["info"] = {"model": s.model, "device": s.device, "store": s.store,
                          "collection": s.collection if s.store == "qdrant" else None,
                          "pages": len(store), "vlm": s.vlm_enabled,
                          "rerank": _STATE.get("reranker") is not None,
                          "heatmap": getattr(emb, "heatmap_supported", True)}
    except Exception as e:  # no index yet, or load failed — UI shows a friendly hint
        _STATE["error"] = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="colpali-rag", lifespan=lifespan)


def _need_index():
    if _STATE["store"] is None:
        raise HTTPException(status_code=503, detail=_STATE["error"] or "no index loaded")


def _snippet(text: str, q: str, width: int = 220) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    low = text.lower()
    for term in sorted(q.split(), key=len, reverse=True):
        i = low.find(term.lower())
        if i >= 0:
            a = max(0, i - width // 3)
            return ("…" if a else "") + text[a : a + width] + ("…" if a + width < len(text) else "")
    return text[:width] + ("…" if len(text) > width else "")


@app.get("/", response_class=HTMLResponse)
def home():
    return _INDEX_HTML


@app.get("/api/status")
def status():
    if _STATE["error"] and _STATE["store"] is None:
        return {"ready": False, "error": _STATE["error"]}
    return {"ready": True, **_STATE["info"]}


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), k: int = 12):
    _need_index()
    with _LOCK:
        hits = retrieve(_STATE["store"], q, top_k=k, reranker=_STATE.get("reranker"),
                        settings=_STATE.get("settings"))
    return {"query": q, "results": [
        {"page_id": pid, "doc": r.doc, "page": r.page, "score": round(sc, 3),
         "snippet": _snippet(r.text, q)}
        for r, sc, pid in hits]}


@app.get("/api/image")
def image(page_id: str):
    _need_index()
    s = _STATE.get("settings")
    # presigned redirect only if explicitly opted in (proxy is the safe default so
    # access control stays with the app, not a shareable URL)
    if s and getattr(s, "storage_serve_mode", "proxy") == "presigned":
        try:
            url = _STATE["store"].image_url(page_id, expires_in=s.storage_url_ttl)
        except Exception:  # noqa: BLE001 - presign failure -> fall through to proxy (never 500)
            url = None
        if url:
            return RedirectResponse(url, status_code=302)
    im = _STATE["store"].get_image(page_id)
    if im is None:
        raise HTTPException(status_code=404, detail="page image not found")
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/ask")
def ask(q: str = Query(..., min_length=1), k: int | None = None):
    """Optional RAG answer: retrieve top pages, let a (vendor-neutral) vision model
    read them (each labelled with its page, so citations are verifiable) and answer.
    Disabled unless VLM_BASE_URL is set. Gated by answer_min_score if configured."""
    _need_index()
    s = _STATE.get("settings")
    if not s or not s.vlm_enabled:
        raise HTTPException(status_code=503,
                            detail="No answer model configured — set VLM_BASE_URL. Search still works.")
    top_k = k or s.answer_top_k
    with _LOCK:
        hits = retrieve(_STATE["store"], q, top_k=top_k, reranker=_STATE.get("reranker"), settings=s)
    if s.answer_min_score is not None and (not hits or hits[0][1] < s.answer_min_score):
        return {"question": q, "answer": "No sufficiently relevant page was found for this question.",
                "sources": [], "gated": True}
    imgs, labels, sources = [], [], []
    for r, sc, pid in hits:
        im = _STATE["store"].get_image(pid)
        if im is not None:
            imgs.append(im)
            labels.append(f"Page {r.page} of {r.doc}:")
            sources.append({"doc": r.doc, "page": r.page, "page_id": pid, "score": round(sc, 3)})
    if not imgs:
        raise HTTPException(status_code=404, detail="no pages to read")

    if s.answer_structured:
        from colpali_rag.faithfulness import apply_gate, judge_answer
        from colpali_rag.generator import answer_structured

        page_ids = [src["page_id"] for src in sources]
        slabels = [f"[{i+1}] Page {src['page']} of {src['doc']}:" for i, src in enumerate(sources)]
        try:
            result = answer_structured(
                q, imgs, attached_page_ids=page_ids, base_url=s.vlm_base_url,
                api_key=s.vlm_api_key, model=s.vlm_model, labels=slabels,
                mode=s.answer_structured_mode, max_retries=s.answer_max_retries)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"answer model error: {type(e).__name__}: {e}") from e
        report, withheld = None, False
        if s.faithfulness_gate != "off":
            try:  # best-effort check: a judge/storage failure must not 500 a good answer
                report = judge_answer(result, lambda pid: _STATE["store"].get_image(pid), s)
                result, withheld = apply_gate(result, report, s.faithfulness_gate, s.faithfulness_min_score)
            except Exception as e:  # noqa: BLE001
                _log.warning("faithfulness check skipped: %s: %s", type(e).__name__, e)
                report, withheld = None, False
        resp = {"question": q, "answer": result.answer, "sources": sources,
                "structured": result.structured, "mode": result.mode,
                "claims": [{"text": c.text, "pages": c.pages, "confidence": c.confidence}
                           for c in result.claims],
                "hallucinated_citations": result.hallucinated_citations}
        if report is not None:
            resp["faithfulness"] = {
                "score": report.faithfulness, "citation_precision": report.citation_precision,
                "withheld": withheld,
                "verdicts": [{"claim": v.claim_index, "verdict": v.verdict, "pages": v.pages,
                              "why": v.why} for v in report.verdicts]}
        return resp

    from colpali_rag.generator import answer as vlm_answer

    try:
        text = vlm_answer(q, imgs, base_url=s.vlm_base_url, api_key=s.vlm_api_key,
                          model=s.vlm_model, labels=labels)
    except Exception as e:  # noqa: BLE001 - surface endpoint/model failures cleanly
        raise HTTPException(status_code=502, detail=f"answer model error: {type(e).__name__}: {e}") from e
    return {"question": q, "answer": text, "sources": sources}


@app.get("/api/heatmap")
def heatmap(page_id: str, q: str = Query(..., min_length=1)):
    _need_index()
    im = _STATE["store"].get_image(page_id)
    if im is None:
        raise HTTPException(status_code=404, detail="page image not found")
    try:
        with _LOCK:
            tokens, maps = _STATE["embedder"].similarity_maps(im, q)
    except HeatmapUnsupported as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    overlays = {"all": H.to_data_uri(H.overlay(im, np.array(maps[-1])))} if -1 in maps else {}
    for t in tokens:
        overlays[str(t["index"])] = H.to_data_uri(H.overlay(im, np.array(maps[t["index"]])))
    return {"tokens": tokens, "overlays": overlays}


# --------------------------------------------------------------------------- UI
_INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>colpali-rag — visual document search</title>
<style>
  :root{
    --bg:#eef1f2; --panel:#fff; --panel2:#f5f7f8; --ink:#0e1a1f; --soft:#43555b;
    --muted:#6a7c82; --line:#d6dee0; --accent:#0c6e7c; --accent2:#0a5a66;
    --shadow:0 1px 2px rgba(10,40,45,.06),0 8px 24px rgba(10,40,45,.08);
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#080f13; --panel:#0f1a20; --panel2:#13212a; --ink:#e7eeef; --soft:#aebdc1;
    --muted:#7f9197; --line:#21333b; --accent:#3cc9d6; --accent2:#84e2ea;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
  header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:16px;
    padding:12px 20px;background:color-mix(in srgb,var(--panel) 88%,transparent);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
  .brand{font-weight:800;letter-spacing:-.02em;font-size:1.05rem}
  .brand b{color:var(--accent2)}
  .chip{font-family:ui-monospace,monospace;font-size:.72rem;color:var(--soft);
    border:1px solid var(--line);border-radius:20px;padding:4px 10px;background:var(--panel2);white-space:nowrap}
  .chip .led{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);
    box-shadow:0 0 6px var(--accent);margin-right:6px;vertical-align:middle}
  .grow{flex:1}
  main{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
  .searchbar{display:flex;gap:10px;margin:8px 0 6px}
  .searchbar input{flex:1;font-size:1.05rem;padding:13px 16px;border-radius:11px;
    border:1px solid var(--line);background:var(--panel);color:var(--ink);box-shadow:var(--shadow)}
  .searchbar input:focus{outline:2px solid var(--accent);outline-offset:1px}
  .searchbar button{font:inherit;font-weight:600;padding:0 20px;border-radius:11px;border:0;cursor:pointer;
    background:var(--accent);color:#fff}
  @media (prefers-color-scheme:dark){.searchbar button{color:#04222a}}
  .hint{color:var(--muted);font-size:.9rem;margin:2px 2px 18px}
  .hint b{color:var(--soft)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(178px,1fr));gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;
    cursor:pointer;box-shadow:var(--shadow);transition:transform .12s,border-color .12s}
  .card:hover{transform:translateY(-2px);border-color:color-mix(in srgb,var(--accent) 55%,var(--line))}
  .thumb{width:100%;aspect-ratio:3/4;object-fit:cover;object-position:top;display:block;background:var(--panel2);border-bottom:1px solid var(--line)}
  .meta{padding:9px 11px}
  .meta .doc{font-size:.82rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .meta .row{display:flex;justify-content:space-between;align-items:center;margin-top:5px;gap:8px}
  .pg{font-family:ui-monospace,monospace;font-size:.7rem;color:var(--muted)}
  .scorebar{flex:1;height:6px;border-radius:4px;background:var(--panel2);overflow:hidden}
  .scorebar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
  .sc{font-family:ui-monospace,monospace;font-size:.72rem;color:var(--accent2);font-variant-numeric:tabular-nums}
  /* detail overlay */
  .scrim{position:fixed;inset:0;z-index:20;background:rgba(4,12,15,.55);backdrop-filter:blur(3px);
    display:none;align-items:stretch;justify-content:center;padding:24px}
  .scrim.on{display:flex}
  .detail{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);
    width:min(1120px,100%);display:grid;grid-template-columns:1.55fr 1fr;overflow:hidden}
  @media (max-width:820px){.detail{grid-template-columns:1fr;overflow:auto}}
  .stage{position:relative;background:var(--panel2);display:flex;align-items:center;justify-content:center;
    min-height:320px;max-height:88vh;overflow:auto;padding:10px}
  .frame{position:relative;display:inline-block;line-height:0}
  .frame img{display:block;max-width:100%;max-height:86vh}
  #heatimg{position:absolute;top:0;left:0;width:100%;height:100%;transition:opacity .18s}
  .side{padding:20px 22px;display:flex;flex-direction:column;gap:14px;min-width:0;max-height:88vh;overflow:auto}
  .side h3{margin:0;font-size:1.05rem;letter-spacing:-.01em;overflow-wrap:anywhere}
  .side .sub{color:var(--muted);font-family:ui-monospace,monospace;font-size:.76rem;margin-top:-8px}
  .label{font-family:ui-monospace,monospace;font-size:.68rem;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
  .tokens{display:flex;flex-wrap:wrap;gap:7px}
  .tok{font:inherit;font-size:.86rem;cursor:pointer;padding:5px 11px;border-radius:20px;
    border:1px solid var(--line);background:var(--panel2);color:var(--soft)}
  .tok.active{background:var(--accent);border-color:var(--accent);color:#fff}
  @media (prefers-color-scheme:dark){.tok.active{color:#04222a}}
  .toggle{display:flex;align-items:center;gap:9px;font-size:.9rem;color:var(--soft)}
  .snip{font-size:.9rem;color:var(--soft);background:var(--panel2);border:1px solid var(--line);
    border-radius:10px;padding:11px 13px;overflow-wrap:anywhere}
  .x{position:absolute;top:14px;right:16px;z-index:22;width:34px;height:34px;border-radius:50%;border:1px solid var(--line);
    background:var(--panel);color:var(--ink);cursor:pointer;font-size:1.1rem;line-height:1}
  .empty{text-align:center;color:var(--muted);padding:70px 20px}
  .empty .big{font-size:2.1rem;font-weight:800;letter-spacing:-.02em;color:var(--ink);margin-bottom:6px}
  .spin{width:20px;height:20px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;
    animation:spin .7s linear infinite;display:inline-block;vertical-align:-4px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .banner{background:color-mix(in srgb,var(--accent) 10%,var(--panel));border:1px solid color-mix(in srgb,var(--accent) 35%,var(--line));
    border-radius:11px;padding:13px 15px;color:var(--soft);font-size:.92rem;margin:14px 0}
  code{font-family:ui-monospace,monospace;font-size:.85em;background:var(--panel2);border:1px solid var(--line);padding:1px 6px;border-radius:6px}
</style></head>
<body>
<header>
  <div class="brand">colpali·<b>rag</b></div>
  <div class="grow"></div>
  <div class="chip" id="status"><span class="led"></span>loading…</div>
</header>
<main>
  <div class="searchbar">
    <input id="q" placeholder="Search your documents in plain language…" autofocus>
    <button id="go">Search</button>
  </div>
  <div class="hint">ColPali reads the page <b>pixels</b> — no OCR. Click a result to see <b>where on the page</b> the model matched, per query word.</div>
  <div id="askwrap" style="display:none">
    <div class="searchbar" style="margin-top:2px">
      <input id="aq" placeholder="…or ask a question and get an answer read from the top pages">
      <button id="askgo">Ask</button>
    </div>
    <div id="answer"></div>
  </div>
  <div id="results"></div>
</main>

<div class="scrim" id="scrim">
  <div class="detail">
    <div class="stage" id="stage">
      <button class="x" id="close">✕</button>
      <div class="frame">
        <img id="baseimg" alt="">
        <img id="heatimg" alt="">
      </div>
    </div>
    <div class="side">
      <div>
        <h3 id="dTitle">—</h3>
        <div class="sub" id="dSub"></div>
      </div>
      <label class="toggle"><input type="checkbox" id="heatOn" checked> show similarity heatmap</label>
      <div>
        <div class="label" style="margin-bottom:7px">Highlight by query term <span id="mapSpin"></span></div>
        <div class="tokens" id="tokens"></div>
      </div>
      <div>
        <div class="label" style="margin-bottom:6px">Page text (snippet)</div>
        <div class="snip" id="dSnip">—</div>
      </div>
    </div>
  </div>
</div>

<script>
const $=s=>document.querySelector(s), api=(p)=>fetch(p).then(r=>r.json());
let CUR=null; // {page_id, q, overlays, tokens, activeTok}

async function boot(){
  try{
    const s=await api('/api/status');
    if(!s.ready){ $('#status').innerHTML='<span class="led" style="background:#d9683a;box-shadow:none"></span>no index';
      $('#results').innerHTML='<div class="banner">No index loaded yet. On a terminal run '+
        '<code>colpali-rag index &lt;folder-of-pdfs&gt;</code> then reload this page.<br><span style="color:var(--muted)">'+
        (s.error||'')+'</span></div>'; return; }
    $('#status').innerHTML='<span class="led"></span>'+s.model.split('/').pop()+' · '+s.pages+' pages · '+s.store+(s.vlm?' · ask ✓':'');
    if(s.vlm) $('#askwrap').style.display='block';
    $('#results').innerHTML='<div class="empty"><div class="big">Search your documents</div>Type a question above. Every result shows a heatmap of what the model looked at.</div>';
  }catch(e){ $('#status').textContent='offline'; }
}

async function doSearch(){
  const q=$('#q').value.trim(); if(!q) return;
  $('#results').innerHTML='<div class="empty"><span class="spin"></span> searching…</div>';
  let d; try{ d=await api('/api/search?k=12&q='+encodeURIComponent(q)); }
  catch(e){ $('#results').innerHTML='<div class="banner">search failed</div>'; return; }
  if(!d.results.length){ $('#results').innerHTML='<div class="empty">No pages found.</div>'; return; }
  const max=Math.max(...d.results.map(r=>r.score))||1, min=Math.min(...d.results.map(r=>r.score));
  const g=document.createElement('div'); g.className='grid';
  d.results.forEach(r=>{
    const pct=Math.round(100*((r.score-min)/((max-min)||1))*0.85+15);
    const c=document.createElement('div'); c.className='card';
    c.innerHTML=`<img class="thumb" loading="lazy" src="/api/image?page_id=${encodeURIComponent(r.page_id)}">
      <div class="meta"><div class="doc" title="${r.doc}">${r.doc}</div>
      <div class="row"><span class="pg">p${r.page}</span><span class="scorebar"><i style="width:${pct}%"></i></span>
      <span class="sc">${r.score.toFixed(2)}</span></div></div>`;
    c.onclick=()=>openDetail(r,q);
    g.appendChild(c);
  });
  $('#results').innerHTML=''; $('#results').appendChild(g);
}

async function openDetail(r,q){
  CUR={page_id:r.page_id,q};
  $('#scrim').classList.add('on');
  $('#dTitle').textContent=r.doc;
  $('#dSub').textContent='page '+r.page+'  ·  score '+r.score.toFixed(3);
  $('#dSnip').textContent=r.snippet||'(no extractable text on this page)';
  $('#baseimg').src='/api/image?page_id='+encodeURIComponent(r.page_id);
  $('#heatimg').src=''; $('#tokens').innerHTML=''; $('#mapSpin').innerHTML='<span class="spin"></span>';
  let h; try{ h=await api('/api/heatmap?page_id='+encodeURIComponent(r.page_id)+'&q='+encodeURIComponent(q)); }
  catch(e){ $('#mapSpin').innerHTML=''; return; }
  $('#mapSpin').innerHTML='';
  if(!h||!h.overlays||!h.tokens){
    $('#tokens').innerHTML='<span style="color:var(--muted);font-size:.85rem">'+(h&&h.detail?h.detail:'heatmap unavailable for this model')+'</span>';
    $('#heatimg').src=''; return; }
  CUR.overlays=h.overlays; CUR.tokens=h.tokens;
  const chips=[{index:'all',text:'All'}].concat(h.tokens.map(t=>({index:String(t.index),text:t.text})));
  chips.forEach((t,i)=>{
    const b=document.createElement('button'); b.className='tok'+(i===0?' active':''); b.textContent=t.text;
    b.onclick=()=>{ document.querySelectorAll('.tok').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); setHeat(t.index); };
    $('#tokens').appendChild(b);
  });
  setHeat('all');
}
function setHeat(key){
  CUR.activeTok=key;
  const on=$('#heatOn').checked, u=CUR.overlays&&CUR.overlays[key];
  $('#heatimg').style.opacity=on?1:0;
  if(u) $('#heatimg').src=u;
}
$('#heatOn').onchange=()=>setHeat(CUR&&CUR.activeTok||'all');
$('#close').onclick=()=>$('#scrim').classList.remove('on');
$('#scrim').onclick=e=>{ if(e.target===$('#scrim')) $('#scrim').classList.remove('on'); };
document.addEventListener('keydown',e=>{ if(e.key==='Escape') $('#scrim').classList.remove('on'); });
async function doAsk(){
  const q=$('#aq').value.trim(); if(!q) return;
  $('#answer').innerHTML='<div class="banner"><span class="spin"></span> reading the top pages…</div>';
  let d; try{ const r=await fetch('/api/ask?k=3&q='+encodeURIComponent(q)); d=await r.json();
    if(!r.ok) throw new Error(d.detail||'failed'); }
  catch(e){ $('#answer').innerHTML='<div class="banner">'+(e.message||'answer failed')+'</div>'; return; }
  const src=(d.sources||[]).map(s=>s.doc+' p'+s.page).join(' · ');
  $('#answer').innerHTML='<div class="banner"><div style="white-space:pre-wrap">'+
    d.answer.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))+'</div>'+
    '<div class="mono" style="margin-top:8px;font-size:.72rem;color:var(--muted)">sources: '+src+'</div></div>';
}
$('#go').onclick=doSearch;
$('#q').addEventListener('keydown',e=>{ if(e.key==='Enter') doSearch(); });
$('#askgo').onclick=doAsk;
$('#aq').addEventListener('keydown',e=>{ if(e.key==='Enter') doAsk(); });
boot();
</script>
</body></html>"""
