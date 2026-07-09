"""colpali-rag — a generic, visual multimodal document RAG.

Point it at any folder of PDFs. It embeds each *page image* with a ColPali
(late-interaction / ColVision) model — no OCR — stores the multivectors, and lets
you search in natural language and **see where on the page the model looked** via
a similarity heatmap, with clickable per-token highlighting.

Domain-agnostic: it only knows "documents", "pages", and "queries".
"""

__version__ = "0.1.0"
