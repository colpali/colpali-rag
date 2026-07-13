"""Index build + open + retrieve. Ties pdf -> embedder -> store together and hands
back a ready-to-query store for the CLI and the web app."""

from __future__ import annotations

import logging
from pathlib import Path

from colpali_rag.config import Settings
from colpali_rag.embedder import get_embedder
from colpali_rag.errors import IndexModelMismatch, PdfRenderError
from colpali_rag.pdf import (
    Page,
    doc_id,
    extract_page_texts,
    list_pdfs,
    render_page_images,
    text_coverage,
)
from colpali_rag.store import build_store, load_store

log = logging.getLogger(__name__)


def build_index(docs_dir, settings: Settings, progress=lambda m: None):
    """Rasterize + embed every PDF page under docs_dir and persist the index.
    A single unreadable PDF is skipped (logged), not fatal."""
    pdfs = list_pdfs(docs_dir)
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {docs_dir}")

    progress(f"loading model {settings.model} on {settings.device} …")
    embedder = get_embedder(settings.model, settings.device, settings.batch_size, settings.family)

    records: list[Page] = []
    images = []
    skipped = []
    for pdf in pdfs:
        try:
            texts = extract_page_texts(pdf)
            imgs = render_page_images(pdf, dpi=settings.dpi, max_dim=settings.max_dim)
        except PdfRenderError as e:
            log.warning("skipping unreadable PDF: %s", e)
            progress(f"  ⚠ skipped {pdf.name}: {e}")
            skipped.append(str(pdf))
            continue
        did = doc_id(pdf, docs_dir)
        for i, (t, im) in enumerate(zip(texts, imgs), start=1):
            records.append(Page(doc=did, page=i, text=t))
            images.append(im)
        progress(f"  · {did}: {len(imgs)} page(s)")

    if not records:
        raise FileNotFoundError(f"no readable PDF pages under {docs_dir}")

    cov = text_coverage([r.text for r in records])
    if cov < 0.5:
        progress(f"  ⚠ only {cov:.0%} of pages have extractable text — likely scanned; "
                 "ColPali reads pixels so retrieval still works, but keyword filtering won't.")

    progress(f"embedding {len(records)} page(s) …")
    embs = embedder.embed_pages(images)
    store = build_store(settings, embedder).build_from(records, images, embs)

    return store, embedder, {
        "docs": len({r.doc for r in records}),
        "pages": len(records),
        "skipped": len(skipped),
        "model": settings.model,
        "device": settings.device,
        "store": settings.store,
        "collection": settings.collection if settings.store == "qdrant" else None,
        "data_dir": str(Path(settings.data_dir).resolve()),
        "text_coverage": round(cov, 3),
    }


def open_index(settings: Settings):
    """Re-open a persisted index for querying/serving (no re-embedding). Raises
    IndexModelMismatch if the configured model differs from the one it was built with."""
    import json

    rec_path = Path(settings.data_dir) / "records.json"
    if not rec_path.exists():
        raise FileNotFoundError(
            f"no index at {settings.data_dir!r}. Run: colpali-rag index <pdf_dir>"
        )
    # cheap pre-check from metadata so we never download/load the WRONG model just to error
    meta = json.loads(rec_path.read_text())
    built = meta.get("model")
    if built and built != settings.model:
        raise IndexModelMismatch(
            f"index was built with model {built!r} but COLPALI_MODEL is {settings.model!r}. "
            f"Re-index, or set COLPALI_MODEL={built}."
        )
    embedder = get_embedder(settings.model, settings.device, settings.batch_size, settings.family)
    store = load_store(settings, embedder)   # defense-in-depth identity/schema check
    return store, embedder


def _rrf(rankings, kappa: int, top_k: int):
    """Reciprocal Rank Fusion. rankings: list of id-lists (best-first). Returns [(id, score)]
    best-first. Needs no score calibration between channels — it fuses ranks, sidestepping the
    fact that MaxSim and BM25 live on different scales."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (kappa + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def _lexical_index(store, settings):
    """Build (and cache on the store) a lexical index over Page.text. Cached because the
    corpus is fixed after indexing; a re-index makes a new store with no cache."""
    lo = int(getattr(settings, "hybrid_ngram_min", 3))
    hi = int(getattr(settings, "hybrid_ngram_max", 5))
    cached = getattr(store, "_lexical_index", None)
    if cached and cached[0] == (lo, hi):
        return cached[1]
    from colpali_rag.lexical import LexicalIndex

    idx = LexicalIndex(list(zip(store.ids, [r.text for r in store.records])), ngram=(lo, hi))
    try:
        store._lexical_index = ((lo, hi), idx)
    except Exception:  # noqa: BLE001 - a store that forbids attributes just won't cache
        pass
    return idx


def _hybrid_search(store, query: str, want: int, settings):
    """Fuse the visual MaxSim ranking with a lexical ranking over Page.text via RRF. Returns up
    to `want` [(Page, fused_score, page_id)] (caller passes the reranker's candidate depth here,
    not the final top_k, so a two-stage rerank still sees a wide pool), or None to fall back to
    visual-only (no text/records, or a scanned corpus where the lexical channel would be noise)."""
    records = getattr(store, "records", None)
    if not records:
        return None
    min_cov = float(getattr(settings, "hybrid_min_coverage", 0.5))
    if text_coverage([r.text for r in records]) < min_cov:
        return None                                     # scanned -> keyword channel is unreliable
    fetch = max(want, int(getattr(settings, "hybrid_fetch", 100)))
    visual = store.search(query, top_k=fetch)           # [(Page, score, id)]
    lexical = _lexical_index(store, settings).search(query, fetch)   # [(id, score)]
    if not lexical:
        return visual[:want]                            # nothing matched lexically -> visual
    kappa = int(getattr(settings, "hybrid_kappa", 60))
    fused = _rrf([[pid for _r, _s, pid in visual], [pid for pid, _s in lexical]], kappa, want)
    rec_by_id = dict(zip(store.ids, store.records))
    for rec, _s, pid in visual:
        rec_by_id.setdefault(pid, rec)
    return [(rec_by_id[pid], float(score), pid) for pid, score in fused if pid in rec_by_id]


def retrieve(store, query: str, top_k: int, *, reranker=None, first_stage_n: int | None = None,
             settings=None):
    """Retrieve top_k pages, optionally applying a second-stage reranker.

    Returns [(Page, score, page_id)]. First-stage MaxSim fetches first_stage_n (>= top_k)
    candidates; if a reranker is provided it re-orders them and we keep top_k. When hybrid
    retrieval is enabled (settings.hybrid_enabled), the first stage instead fuses the visual
    MaxSim ranking with a lexical ranking over Page.text (RRF) so exact identifiers aren't lost
    to a blurred page; it degrades to visual-only for a scanned corpus. The first-stage list is
    never mutated in place — reranked scores replace the returned ones.
    """
    if settings is not None and getattr(settings, "hybrid_enabled", False):
        n = first_stage_n or (max(top_k, 30) if reranker else top_k)   # same pool as visual path
        hybrid = _hybrid_search(store, query, n, settings)
        if hybrid is not None:
            if reranker is None or not hybrid:
                return hybrid[:top_k]
            return reranker.rerank(query, hybrid, store, top_k=top_k)[:top_k]

    n = first_stage_n or (max(top_k, 30) if reranker else top_k)
    hits = store.search(query, top_k=n)
    if reranker is None or not hits:
        return hits[:top_k]
    order = reranker.rerank(query, hits, store, top_k=top_k)  # -> [(Page, score, page_id)]
    return order[:top_k]
