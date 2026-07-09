"""PDF -> page images + page text, via pypdfium2 (Apache-2.0 / BSD, bundles PDFium).
No OCR here — ColPali reads the page pixels directly; text is kept only for optional
keyword filtering and snippets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    doc: str          # file name
    page: int         # 1-based
    text: str         # extracted text (may be empty for scanned pages)


def list_pdfs(docs_dir) -> list[Path]:
    return sorted(Path(docs_dir).rglob("*.pdf"))


def render_page_images(pdf_path, dpi: int = 150, max_dim: int = 1600) -> list:
    """Rasterize each page to a PIL RGB image (longest side <= max_dim)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        scale = dpi / 72.0
        images = []
        for i in range(len(pdf)):
            img = pdf[i].render(scale=scale).to_pil()
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > max_dim:
                r = max_dim / max(img.size)
                img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
            images.append(img)
        return images
    finally:
        pdf.close()


def extract_page_texts(pdf_path) -> list[str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return [pdf[i].get_textpage().get_text_range() or "" for i in range(len(pdf))]
    finally:
        pdf.close()


def text_coverage(texts: list[str], min_chars: int = 20) -> float:
    """Fraction of pages with real extractable text; near 0 => scanned/image-only."""
    if not texts:
        return 1.0
    return sum(1 for t in texts if len((t or "").strip()) >= min_chars) / len(texts)
