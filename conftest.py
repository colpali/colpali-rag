"""Generate a tiny, neutral sample PDF corpus for tests (git-ignored).

The tests need *some* text PDFs to exercise the PDF layer and store; these are
generic placeholder documents with no particular subject. Needs reportlab (dev extra).
"""

from pathlib import Path

import pytest

SAMPLE = Path(__file__).parent / "tests" / "_sample_docs"


@pytest.fixture(scope="session", autouse=True)
def _sample_docs():
    SAMPLE.mkdir(parents=True, exist_ok=True)
    if not list(SAMPLE.glob("*.pdf")):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

            st = getSampleStyleSheet()

            def make(name, title, rows):
                SimpleDocTemplate(str(SAMPLE / name), pagesize=A4, title=title).build(
                    [Paragraph(title, st["Heading1"]), Spacer(1, 12), Table(rows)]
                )

            make("sample_a.pdf", "Sample Document A",
                 [["Property", "Value"], ["Height", "120 mm"], ["Weight", "300 g"], ["Colour", "black"]])
            make("sample_b.pdf", "Sample Document B",
                 [["Item", "Count"], ["Alpha", "4"], ["Beta", "2"], ["Gamma", "7"]])
        except Exception:  # noqa: BLE001 - corpus-dependent tests report clearly if this fails
            pass
    yield
