"""Index build + open. Ties pdf -> embedder -> store together and hands back a
ready-to-query store for the CLI and the web app."""

from __future__ import annotations

from pathlib import Path

from colpali_rag.config import Settings
from colpali_rag.embedder import get_embedder
from colpali_rag.pdf import Page, extract_page_texts, list_pdfs, render_page_images, text_coverage
from colpali_rag.store import build_store, load_store


def build_index(docs_dir, settings: Settings, progress=lambda m: None):
    """Rasterize + embed every PDF page under docs_dir and persist the index."""
    pdfs = list_pdfs(docs_dir)
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {docs_dir}")

    progress(f"loading model {settings.model} on {settings.device} …")
    embedder = get_embedder(settings.model, settings.device, settings.batch_size)

    records: list[Page] = []
    images = []
    for pdf in pdfs:
        texts = extract_page_texts(pdf)
        imgs = render_page_images(pdf)
        for i, (t, im) in enumerate(zip(texts, imgs), start=1):
            records.append(Page(doc=pdf.name, page=i, text=t))
            images.append(im)
        progress(f"  · {pdf.name}: {len(imgs)} page(s)")

    cov = text_coverage([r.text for r in records])
    if cov < 0.5:
        progress(f"  ⚠ only {cov:.0%} of pages have extractable text — likely scanned; "
                 "ColPali reads pixels so retrieval still works, but keyword filtering won't.")

    progress(f"embedding {len(records)} page(s) …")
    embs = embedder.embed_pages(images)
    store = build_store(settings, embedder).build_from(records, images, embs)

    return store, embedder, {
        "docs": len(pdfs),
        "pages": len(records),
        "model": settings.model,
        "device": settings.device,
        "store": settings.store,
        "collection": settings.collection if settings.store == "qdrant" else None,
        "data_dir": str(Path(settings.data_dir).resolve()),
        "text_coverage": round(cov, 3),
    }


def open_index(settings: Settings):
    """Re-open a persisted index for querying/serving (no re-embedding)."""
    if not (Path(settings.data_dir) / "records.json").exists():
        raise FileNotFoundError(
            f"no index at {settings.data_dir!r}. Run: colpali-rag index <pdf_dir>"
        )
    embedder = get_embedder(settings.model, settings.device, settings.batch_size)
    store = load_store(settings, embedder)
    return store, embedder
