# ColPali — visual page retrieval

The retrieval half of colpali-rag. A ColPali / ColVision model reads **page images as
pixels** and ranks them for a query — no OCR.

## Late interaction (MaxSim)

Classic embedding search encodes a whole page into **one** vector. ColPali instead runs
a vision-language model over the page image and keeps a **multivector**: one ~128-dim
vector per image *patch* (plus a per-token multivector for the query). Scoring is
**MaxSim** (ColBERT-style late interaction): for each query token, take its max
similarity over all page patches, then sum across query tokens.

```
score(q, page) = Σ_over_query_tokens   max_over_page_patches  ⟨q_tok, patch⟩
```

Why this beats OCR-then-text on real documents:

- **No OCR.** Tables, pinouts, charts, dense/rotated layouts, and scans are captured as
  pixels. OCR accuracy collapses on exactly this content.
- **Fine-grained.** A query can light up the specific patch region that holds the answer,
  even on a busy page — which is what makes the **heatmap** possible.
- **Layout-robust.** Multi-column pages and callouts need no parsing.

Trade-off: multivectors are larger than single vectors and MaxSim is heavier than a dot
product — fine at document scale (hundreds–thousands of pages); see [SCALING.md](SCALING.md).

## Model family & licensing

colpali-rag is model-agnostic within the ColVision family; the class is auto-selected
from the model id at load time (`embedder.py::_load`). For a shipped product the *base*
checkpoint's license matters, not just the adapter.

| Model id | Base | License | Tier |
|---|---|---|---|
| `vidore/colSmol-500M` | SmolVLM (Idefics3) | Apache-2.0 | dev / CPU (default) |
| `vidore/colqwen2-v1.0` | Qwen2-VL | Apache-2.0 | production (GPU); sharp heatmaps |
| `nomic-ai/colnomic-embed-multimodal-7b` | — | Apache-2.0 | top quality tier |
| `vidore/colqwen2.5-*` | Qwen2.5-VL | ⚠️ Qwen **Research** (non-commercial) | avoid in a product |
| `vidore/colpali-v1.x` | PaliGemma | ⚠️ Gemma (use-restricted) | avoid in a product |

Rule of thumb: **colSmol for dev, colqwen2-v1.0 for production.** Swap via `COLPALI_MODEL`.

## How colpali-rag uses it

`embedder.py` wraps a `colpali_engine` model behind two jobs:

1. **Retrieval** — `embed_pages(images)` → one multivector per page; `score(query, embs)`
   returns per-page MaxSim scores. Raw-multivector helpers (`embed_query_raw`,
   `page_to_list`) feed the Qdrant store.
2. **Interpretability** — `similarity_maps(page_image, query)` returns per-content-token
   similarity grids aligned to the page, which `heatmap.overlay` renders as an inferno
   overlay.

Two correctness details worth knowing:

- **Non-split single-image pass.** ColPali models tile large images into sub-images plus
  a global image, which scrambles a naive patch grid. For interpretability, the embedder
  temporarily sets `do_image_splitting=False` so the page is one clean grid, then derives
  `n_patches` by the factorization closest to the page aspect ratio.
- **Content tokens only.** ColPali pads queries with special/`<end_of_utterance>` tokens;
  those are filtered out so per-token heatmaps reflect real query words.

Heatmap resolution follows the model: `colSmol-500M` gives a coarse grid (fast, CPU);
`colqwen2-v1.0` gives high-resolution maps.

## Device notes

`COLPALI_DEVICE` is `cpu | cuda | mps`. `cpu` is safe everywhere. `mps` needs
`torch==2.5.1`. `cuda` is the production path for colqwen2-v1.0 / colnomic.
