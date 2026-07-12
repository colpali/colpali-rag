"""Studio — a structured-output product layer on top of the colpali_rag engine.

The engine (colpali_rag) does grounded visual retrieval: PDF pages -> embeddings ->
store (the "document DB") -> an OpenAI-compatible LLM wrapper for cited answers.

Studio uses that engine to turn a natural-language request plus a chosen set of sources
and uploaded tabular data (CSV / Excel) into a STRUCTURED, CITED output — typed nodes and
labeled connections where every element can point back to the source page or spreadsheet
row it came from, and the same faithfulness machinery can verify it against those sources.

Nothing here is domain-specific. Swap in any corpus.
"""

from colpali_rag.studio.spec import DiagramSpec, validate_diagram_obj  # noqa: F401
