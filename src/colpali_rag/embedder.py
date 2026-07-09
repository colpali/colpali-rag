"""The ColPali (ColVision late-interaction) model wrapper.

Two jobs:
  1. Retrieval — embed page images into patch multivectors and score a query with
     ColBERT-style MaxSim.
  2. Interpretability — for a *single* page + query, return per-query-token
     similarity grids (rows x cols) aligned to the page, so the UI can show a
     heatmap of *where the model looked*. This uses the model's own
     `get_similarity_maps_from_embeddings`, computed on a non-split single-image
     pass so the patch grid is clean and geometrically correct for any model.

The model class is auto-selected from the model id (colSmol/idefics, colqwen2,
colqwen2.5, colpali). Everything is late-bound so importing this module is cheap
and doesn't require torch until you actually load a model.
"""

from __future__ import annotations

import math
import re

# query tokens that are structural, not content — excluded from per-token heatmaps
_SPECIAL_TOK = re.compile(r"^<.*>$|^\s*$")
_STRIP = "Ġ▁ \t"


class ColpaliEmbedder:
    name = "colpali"

    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 1):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._load()

    # ---- loading ---------------------------------------------------------
    def _load(self):
        import torch
        from colpali_engine import models as M

        mid = self.model_id.lower()
        if "smol" in mid or "idefics" in mid:
            model_cls, proc_cls = M.ColIdefics3, M.ColIdefics3Processor
        elif "qwen2.5" in mid or "qwen2_5" in mid:
            model_cls = getattr(M, "ColQwen2_5")
            proc_cls = getattr(M, "ColQwen2_5_Processor", None) or getattr(M, "ColQwen2_5Processor")
        elif "qwen2" in mid:
            model_cls, proc_cls = M.ColQwen2, M.ColQwen2Processor
        else:
            model_cls, proc_cls = M.ColPali, M.ColPaliProcessor

        self.torch = torch
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model = model_cls.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
        self.processor = proc_cls.from_pretrained(self.model_id)

    # ---- retrieval -------------------------------------------------------
    def embed_pages(self, images: list) -> list:
        """One multivector tensor per page image (CPU float)."""
        embs = []
        for i in range(0, len(images), self.batch_size):
            batch = self.processor.process_images(images[i : i + self.batch_size]).to(self.device)
            with self.torch.no_grad():
                out = self.model(**batch)
            embs.extend(list(self.torch.unbind(out.to("cpu").float())))
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
        # infer from a tiny query embedding
        return int(self._query_multivector("x").shape[-1])

    # ---- interpretability (the heatmap) ----------------------------------
    def similarity_maps(self, page_image, query: str):
        """Return (tokens, maps) for a single page + query.

        tokens: list of {"text": str, "index": int} for the *content* query tokens
                (special/padding tokens filtered out).
        maps:   dict token_index -> 2D list (rows=height, cols=width) of scores.
        Also returns an "aggregate" map under key -1 (mean over content tokens).

        Computed on a non-split single-image pass so the patch grid is a clean,
        correctly-ordered rectangle for any ColVision model.
        """
        proc = self.processor
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
            # local image mask excludes the global patch (required by the sanity check)
            mask = proc.get_local_image_mask(bimg)
            n_local = int(mask.sum())
            nx, ny = self._grid_shape(n_local, page_image.size)
            maps = proc.get_similarity_maps_from_embeddings(
                image_embeddings=img_emb,
                query_embeddings=q_emb,
                n_patches=(nx, ny),
                image_mask=mask,
            )[0].float()  # (query_tokens, nx, ny)
        finally:
            if ip is not None and prev_split is not None:
                ip.do_image_splitting = prev_split

        ids = bq["input_ids"][0].tolist()
        raw_tokens = proc.tokenizer.convert_ids_to_tokens(ids)

        content, out_maps = [], {}
        acc = None
        for i, tok in enumerate(raw_tokens):
            if _SPECIAL_TOK.match(tok):
                continue
            text = tok.strip(_STRIP)
            if not text:
                continue
            grid = maps[i].numpy().T  # (nx,ny) -> (ny=rows/height, nx=cols/width)
            content.append({"index": i, "text": text})
            out_maps[i] = grid.tolist()
            acc = grid if acc is None else acc + grid
        if acc is not None:
            out_maps[-1] = (acc / max(len(content), 1)).tolist()
        return content, out_maps

    @staticmethod
    def _grid_shape(n_patches: int, image_size) -> tuple[int, int]:
        """Factor the patch count into (n_x, n_y) with the aspect ratio closest to
        the image's W/H. Robust across models whose get_n_patches doesn't account
        for pixel-shuffle (e.g. colSmol resizes to a fixed square grid)."""
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


def get_embedder(model_id: str, device: str = "cpu", batch_size: int = 1) -> ColpaliEmbedder:
    return ColpaliEmbedder(model_id, device=device, batch_size=batch_size)
