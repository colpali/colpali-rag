"""Optional second-stage reranking (the single biggest accuracy lever after the
base model).

OFF by default and NOT on the CPU path: reranking a shortlist of page images with a
2B VLM is seconds/pair, which would wreck the sub-second CPU retrieval story. Enable
with RERANK_ENABLED=true (needs the `[rerank]` extra and, realistically, a GPU).

Pattern: first-stage MaxSim keeps its top-N; the reranker re-scores only the top_k
candidate page images pointwise and re-orders them. Default backend
`lightonai/MonoQwen2-VL-v0.1` — Apache-2.0 (LoRA on Qwen2-VL-2B). AVOID
`jina-reranker-m0` (CC-BY-NC) in a shipped product.

The MonoQwen backend is GPU-oriented and not exercised in the fast test suite; any
load/inference failure degrades to the first-stage order (logged), never a 500.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class NoopReranker:
    """Identity — returns the first-stage order unchanged."""

    def rerank(self, query, hits, store, top_k):
        return hits[:top_k]


class MonoQwenReranker:
    """Pointwise VLM reranker: score = P('True') that a page is relevant to the query."""

    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype).to(device).eval()
        tok = self.processor.tokenizer
        self._true = tok.convert_tokens_to_ids("True")
        self._false = tok.convert_tokens_to_ids("False")

    def _score(self, query: str, image) -> float:
        prompt = ("Assert the relevance of the previous image document to the following "
                  f"query, answer True or False. The query is: {query}")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]
        pair = self.torch.tensor([logits[self._true], logits[self._false]])
        return float(self.torch.softmax(pair, dim=0)[0])

    def rerank(self, query, hits, store, top_k):
        shortlist = hits[:top_k]
        try:
            scored = []
            for page, _first, pid in shortlist:
                img = store.get_image(pid)
                s = self._score(query, img) if img is not None else 0.0
                scored.append((page, s, pid))
            scored.sort(key=lambda t: -t[1])
            log.info("reranked %d candidates for query=%r", len(scored), query[:40])
            return scored
        except Exception as e:  # noqa: BLE001 - never fail the request; fall back
            log.warning("rerank failed (%s: %s); keeping first-stage order", type(e).__name__, e)
            return shortlist


def get_reranker(settings):
    """Build a reranker from settings, or None if disabled."""
    if not getattr(settings, "rerank_enabled", False):
        return None
    backend = getattr(settings, "rerank_backend", "none")
    if backend in ("none", "", None):
        return None
    if backend == "monoqwen":
        return MonoQwenReranker(settings.rerank_model, settings.rerank_device)
    from colpali_rag.errors import ColpaliRagError

    raise ColpaliRagError(f"unknown rerank backend {backend!r} (expected: none | monoqwen)")
