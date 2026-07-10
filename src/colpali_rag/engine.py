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


def retrieve(store, query: str, top_k: int, *, reranker=None, first_stage_n: int | None = None):
    """Retrieve top_k pages, optionally applying a second-stage reranker.

    Returns [(Page, score, page_id)]. First-stage MaxSim fetches first_stage_n (>= top_k)
    candidates; if a reranker is provided it re-orders them and we keep top_k. The
    first-stage list is never mutated in place — reranked scores replace the returned ones.
    """
    n = first_stage_n or (max(top_k, 30) if reranker else top_k)
    hits = store.search(query, top_k=n)
    if reranker is None or not hits:
        return hits[:top_k]
    order = reranker.rerank(query, hits, store, top_k=top_k)  # -> [(Page, score, page_id)]
    return order[:top_k]
