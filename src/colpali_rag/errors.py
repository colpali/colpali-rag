"""Typed domain errors so failures carry context (which file, which model, which
version) instead of a raw stack trace deep in a request."""

from __future__ import annotations


class ColpaliRagError(Exception):
    """Base class for all colpali-rag errors."""


class UnsupportedModel(ColpaliRagError):
    """The configured model id matches no known ColVision family."""


class ModelLoadError(ColpaliRagError):
    """A model/processor failed to load (bad id, wrong class, engine mismatch)."""


class EngineCapabilityError(ColpaliRagError):
    """The installed colpali-engine is missing an API this code depends on."""


class HeatmapUnsupported(ColpaliRagError):
    """This model's processor cannot produce similarity maps for the heatmap."""


class IndexModelMismatch(ColpaliRagError):
    """The index was built with a different model/dim than the one now configured."""


class PdfRenderError(ColpaliRagError):
    """A PDF failed to rasterize or extract text (carries the file path)."""


class StoreError(ColpaliRagError):
    """A page store (memory or Qdrant) operation failed."""


class AnswerModelError(ColpaliRagError):
    """The optional answer endpoint failed or returned an unusable response."""
