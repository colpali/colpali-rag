"""Declarative registry of supported ColVision model families — the single source
of truth for (a) which model/processor class to load, (b) whether the heatmap is
supported, and (c) the *base-model* license (baked at curation time, never fetched
from the network, so it works offline).

Adding or fixing a family is a data change here, not new branching logic. Verified
against colpali-engine 0.3.17 class names (ColQwen3, ColGemma3, ColModernVBert,
ColQwen2_5_Processor, …). If the engine renames a class, `resolve()` fails loudly
with the version it saw rather than silently mis-loading.

License codes: 'mit' / 'apache-2.0' are commercially clean; 'gemma' has a use
policy; 'research-nc' is non-commercial only; 'varies' means it depends on the
exact checkpoint (e.g. colnomic 7B is Apache but 3B rides a non-commercial base).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CLEAN_LICENSES = {"mit", "apache-2.0"}


@dataclass(frozen=True)
class ModelSpec:
    pattern: str            # regex tried against the lowercased model id
    model_cls: str          # class name in colpali_engine.models
    proc_cls: str           # processor class name
    family: str
    heatmap: str            # 'full' (processor sim-maps) | 'vendored' (einsum) | 'none'
    license: str            # base-model license code (see module docstring)


# Order matters: more specific patterns first.
REGISTRY: list[ModelSpec] = [
    ModelSpec(r"modernvbert",           "ColModernVBert", "ColModernVBertProcessor", "modernvbert", "full",     "mit"),
    ModelSpec(r"smol|idefics",          "ColIdefics3",    "ColIdefics3Processor",    "idefics3",    "full",     "apache-2.0"),
    ModelSpec(r"qwen3[._]5|qwen3\.5",   "ColQwen3_5",     "ColQwen3_5Processor",     "qwen3",       "vendored", "apache-2.0"),
    ModelSpec(r"qwen3",                 "ColQwen3",       "ColQwen3Processor",       "qwen3",       "vendored", "apache-2.0"),
    ModelSpec(r"colnomic|nomic",        "ColQwen2_5",     "ColQwen2_5_Processor",    "qwen2.5",     "vendored", "varies"),
    ModelSpec(r"qwen2[._]5|qwen2\.5",   "ColQwen2_5",     "ColQwen2_5_Processor",    "qwen2.5",     "vendored", "research-nc"),
    ModelSpec(r"qwen2",                 "ColQwen2",       "ColQwen2Processor",       "qwen2",       "vendored", "apache-2.0"),
    ModelSpec(r"gemma",                 "ColGemma3",      "ColGemmaProcessor3",      "gemma",       "vendored", "gemma"),
    ModelSpec(r"colpali|paligemma",     "ColPali",        "ColPaliProcessor",        "paligemma",   "vendored", "gemma"),
]

SUPPORTED_HINT = (
    "supported families: modernvbert, smol/idefics, qwen3(.5), qwen2(.5), colnomic, "
    "gemma, colpali. Set COLPALI_FAMILY to force one for a new checkpoint."
)


def resolve(model_id: str, family_override: str | None = None) -> ModelSpec:
    """Return the ModelSpec for a model id (or an explicit family override)."""
    from colpali_rag.errors import UnsupportedModel

    mid = model_id.lower()
    if family_override:
        for spec in REGISTRY:
            if spec.family == family_override:
                return spec
        raise UnsupportedModel(f"unknown family override {family_override!r}. {SUPPORTED_HINT}")
    for spec in REGISTRY:
        if re.search(spec.pattern, mid):
            return spec
    raise UnsupportedModel(f"no ColVision family matches model id {model_id!r}. {SUPPORTED_HINT}")


def load_classes(spec: ModelSpec):
    """getattr the actual (model_cls, proc_cls) from the installed engine, failing
    loudly (with the version) if the engine renamed them."""
    import importlib.metadata as _md

    from colpali_engine import models as M

    from colpali_rag.errors import EngineCapabilityError

    model_cls = getattr(M, spec.model_cls, None)
    proc_cls = getattr(M, spec.proc_cls, None)
    if model_cls is None or proc_cls is None:
        try:
            ver = _md.version("colpali-engine")
        except Exception:  # noqa: BLE001
            ver = "unknown"
        missing = [n for n, c in [(spec.model_cls, model_cls), (spec.proc_cls, proc_cls)] if c is None]
        raise EngineCapabilityError(
            f"colpali-engine {ver} is missing {missing} for family {spec.family!r}. "
            "Pin colpali-engine==0.3.17 (see pyproject) or update models_registry.py."
        )
    return model_cls, proc_cls
