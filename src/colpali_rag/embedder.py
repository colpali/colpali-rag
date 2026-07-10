"""The ColPali (ColVision late-interaction) model wrapper.

Two jobs:
  1. Retrieval — embed page images into patch multivectors and score a query with
     ColBERT-style MaxSim.
  2. Interpretability — for a single page + query, return per-content-token
     similarity grids aligned to the page (the heatmap of *where the model looked*),
     computed on a non-split single-image pass.

The model/processor classes come from a declarative registry (models_registry.py)
so a new family is a data change, not new branching — and an unknown id fails loudly
instead of silently loading the wrong architecture. Heavy deps are late-bound.
"""

from __future__ import annotations

import logging
import re

from colpali_rag import models_registry as registry
from colpali_rag.errors import HeatmapUnsupported, ModelLoadError

log = logging.getLogger(__name__)

# query tokens that are structural, not content — excluded from per-token heatmaps
_SPECIAL_TOK = re.compile(r"^<.*>$|^\s*$")
_STRIP = "Ġ▁ \t"


class ColpaliEmbedder:
    name = "colpali"

    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 1,
                 family: str | None = None):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.spec = registry.resolve(model_id, family)   # raises UnsupportedModel on no match
        self.family = self.spec.family
        self._load()

    # ---- loading ---------------------------------------------------------
    def _load(self):
        import torch

        model_cls, proc_cls = registry.load_classes(self.spec)  # raises EngineCapabilityError if renamed
        self.torch = torch
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        try:
            self.model = model_cls.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
            self.processor = proc_cls.from_pretrained(self.model_id)
        except Exception as e:  # noqa: BLE001 - add context (which model, which class)
            raise ModelLoadError(
                f"failed to load {self.model_id!r} as {self.spec.model_cls}/{self.spec.proc_cls} "
                f"(family {self.family}): {type(e).__name__}: {e}"
            ) from e
        if self.spec.license not in registry.CLEAN_LICENSES:
            log.warning("model %s base license is %r (not Apache/MIT) — check terms before shipping",
                        self.model_id, self.spec.license)

    # ---- retrieval -------------------------------------------------------
    def embed_pages(self, images: list) -> list:
        """One multivector tensor per page image. Strips padding per page so
        COLPALI_BATCH_SIZE>1 doesn't fold pad tokens into the stored multivector."""
        embs = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            batch = self.processor.process_images(chunk).to(self.device)
            with self.torch.no_grad():
                out = self.model(**batch).to("cpu").float()
            attn = batch.get("attention_mask")
            attn = attn.to("cpu").bool() if attn is not None else None
            for j in range(len(chunk)):
                v = out[j]
                embs.append(v[attn[j]] if attn is not None else v)
        return embs

    def _query_multivector(self, query: str):
        batch = self.processor.process_queries([query]).to(self.device)
        with self.torch.no_grad():
            q = self.model(**batch)
        return q.to("cpu").float()[0]

    def score(self, query: str, page_embs: list) -> list[float]:
        qs = [self._query_multivector(query)]
        scores = self.processor.score_multi_vector(qs, page_embs)
        return [float(x) for x in scores[0]]

    # raw multivectors for an external store (e.g. Qdrant)
    def embed_query_raw(self, query: str) -> list:
        return self._query_multivector(query).tolist()

    @staticmethod
    def page_to_list(page_emb) -> list:
        return page_emb.tolist() if hasattr(page_emb, "tolist") else page_emb

    @property
    def dim(self) -> int:
        return int(self._query_multivector("x").shape[-1])

    @property
    def heatmap_supported(self) -> bool:
        proc = self.processor
        return hasattr(proc, "get_image_mask") or hasattr(proc, "get_local_image_mask")

    # ---- interpretability (the heatmap) ----------------------------------
    def similarity_maps(self, page_image, query: str):
        """Return (tokens, maps) for a single page + query, cross-model.

        Uses the processor's own similarity maps when available (Idefics3 / ModernVBert),
        else a matplotlib-free einsum path over masked patch embeddings (ColQwen2/3,
        ColPali). Raises HeatmapUnsupported for processors without an image mask.

        maps: {token_index -> 2D list (rows=height, cols=width)}; key -1 = mean over
        content tokens.
        """
        proc = self.processor
        if not self.heatmap_supported:
            raise HeatmapUnsupported(
                f"model {self.model_id!r} (family {self.family}) has no image-mask API; "
                "the heatmap is unavailable for this model."
            )
        ip = getattr(proc, "image_processor", None)
        prev_split = getattr(ip, "do_image_splitting", None) if ip is not None else None
        if ip is not None and hasattr(ip, "do_image_splitting"):
            ip.do_image_splitting = False
        try:
            bimg = proc.process_images([page_image]).to(self.device)
            bq = proc.process_queries([query]).to(self.device)
            with self.torch.no_grad():
                img_emb = self.model(**bimg)
                q_emb = self.model(**bq)
            mask = (proc.get_local_image_mask(bimg) if hasattr(proc, "get_local_image_mask")
                    else proc.get_image_mask(bimg))
            n = int(mask.sum())
            nx, ny = self._resolve_grid(proc, page_image.size, n)
            token_grids = self._compute_maps(proc, img_emb, q_emb, mask, (nx, ny), n)
        finally:
            if ip is not None and prev_split is not None:
                ip.do_image_splitting = prev_split

        ids = bq["input_ids"][0].tolist()
        raw_tokens = proc.tokenizer.convert_ids_to_tokens(ids)
        content, out_maps, acc = [], {}, None
        for i, tok in enumerate(raw_tokens):
            if _SPECIAL_TOK.match(tok):
                continue
            text = tok.strip(_STRIP)
            if not text:
                continue
            grid = token_grids[i]  # (ny, nx) numpy, rows=height
            content.append({"index": i, "text": text})
            out_maps[i] = grid.tolist()
            acc = grid if acc is None else acc + grid
        if acc is not None:
            out_maps[-1] = (acc / max(len(content), 1)).tolist()
        return content, out_maps

    def _resolve_grid(self, proc, image_size, n_patches):
        """(nx, ny) patch grid. Prefer the processor's get_n_patches when it agrees
        with the real token count; else factor n_patches by nearest aspect ratio."""
        if hasattr(proc, "get_n_patches"):
            for extra in ({}, {"patch_size": getattr(proc, "patch_size", 14)},
                          {"spatial_merge_size": getattr(proc, "spatial_merge_size", 2)}):
                try:
                    nx, ny = proc.get_n_patches(image_size, **extra)
                    if nx * ny == n_patches:
                        return nx, ny
                except (TypeError, ValueError, AttributeError):
                    continue
        return self._grid_shape(n_patches, image_size)

    def _compute_maps(self, proc, img_emb, q_emb, mask, n_patches, n):
        """Per-token grids of shape (ny, nx). Uses the processor sim-map method when
        present, else a vendored einsum over row-major masked patch embeddings."""
        nx, ny = n_patches
        if hasattr(proc, "get_similarity_maps_from_embeddings"):
            maps = proc.get_similarity_maps_from_embeddings(
                image_embeddings=img_emb, query_embeddings=q_emb,
                n_patches=(nx, ny), image_mask=mask,
            )[0].float()  # (qlen, nx, ny)
            return [maps[i].numpy().T for i in range(maps.shape[0])]  # -> (ny, nx)
        # vendored path (matplotlib-free): masked patch embeddings, row-major grid
        patches = img_emb[0].float()[mask[0].bool()]           # (n, dim)
        if patches.shape[0] != nx * ny:                        # safety: refactor to exact n
            nx, ny = self._grid_shape(patches.shape[0], (nx, ny))
        grid = patches.reshape(ny, nx, -1)                     # (ny, nx, dim) row-major
        sim = self.torch.einsum("qd,ijd->qij", q_emb[0].float(), grid)  # (qlen, ny, nx)
        return [sim[i].numpy() for i in range(sim.shape[0])]

    @staticmethod
    def _grid_shape(n_patches: int, image_size) -> tuple[int, int]:
        """Factor the patch count into (n_x, n_y) with the aspect ratio closest to
        the image's W/H (robust when get_n_patches disagrees with the token count)."""
        W, H = image_size
        target = (W / H) if H else 1.0
        best = None
        for ny in range(1, n_patches + 1):
            if n_patches % ny:
                continue
            nx = n_patches // ny
            d = abs((nx / ny) - target)
            if best is None or d < best[0]:
                best = (d, nx, ny)
        return best[1], best[2]


def get_embedder(model_id: str, device: str = "cpu", batch_size: int = 1,
                 family: str | None = None) -> ColpaliEmbedder:
    return ColpaliEmbedder(model_id, device=device, batch_size=batch_size, family=family)
