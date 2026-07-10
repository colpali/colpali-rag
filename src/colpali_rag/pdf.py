"""PDF -> page images + page text, via pypdfium2 (Apache-2.0 / BSD, bundles PDFium).
No OCR here — ColPali reads the page pixels directly; text is kept only for optional
keyword filtering and snippets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    doc: str          # corpus-relative path (unique across subfolders)
    page: int         # 1-based
    text: str         # extracted text (may be empty for scanned pages)


def list_pdfs(docs_dir) -> list[Path]:
    """All PDFs under docs_dir, recursively, case-insensitive (.pdf / .PDF)."""
    return sorted(p for p in Path(docs_dir).rglob("*") if p.suffix.lower() == ".pdf")


def doc_id(pdf_path, docs_dir) -> str:
    """Corpus-relative id so two same-named PDFs in different subfolders don't collide."""
    try:
        return str(Path(pdf_path).relative_to(docs_dir))
    except ValueError:
        return Path(pdf_path).name


def render_page_images(pdf_path, dpi: int = 150, max_dim: int = 1600) -> list:
    """Rasterize each page to a PIL RGB image (longest side <= max_dim)."""
    import pypdfium2 as pdfium
    from PIL import Image

    from colpali_rag.errors import PdfRenderError

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:  # noqa: BLE001
        raise PdfRenderError(f"cannot open PDF {pdf_path}: {type(e).__name__}: {e}") from e
    try:
        scale = dpi / 72.0
        images = []
        for i in range(len(pdf)):
            img = pdf[i].render(scale=scale).to_pil()
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > max_dim:
                r = max_dim / max(img.size)
                img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))),
                                 Image.Resampling.LANCZOS)
            images.append(img)
        return images
    finally:
        pdf.close()


def extract_page_texts(pdf_path) -> list[str]:
    import pypdfium2 as pdfium

    from colpali_rag.errors import PdfRenderError

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:  # noqa: BLE001
        raise PdfRenderError(f"cannot open PDF {pdf_path}: {type(e).__name__}: {e}") from e
    try:
        return [pdf[i].get_textpage().get_text_range() or "" for i in range(len(pdf))]
    finally:
        pdf.close()


def text_coverage(texts: list[str], min_chars: int = 20) -> float:
    """Fraction of pages with real extractable text; near 0 => scanned/image-only."""
    if not texts:
        return 1.0
    return sum(1 for t in texts if len((t or "").strip()) >= min_chars) / len(texts)
