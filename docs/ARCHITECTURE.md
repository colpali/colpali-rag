# Architecture

colpali-rag is a small, modular visual document RAG. Every stage is one file with a
clear contract, so each is an independent extension seam (see [EXTENDING.md](EXTENDING.md)).

## Pipeline

```
 PDFs ─▶ rasterize ─▶ ColPali embed ─▶ store ─────────────┐
        (pdf.py)      (embedder.py)   (store.py)           │
                                                           ▼
 query ─▶ embed ─▶ MaxSim rank ──────────────────────▶ top pages
                   (store.search)                          │
                                                           ├─▶ heatmap overlay (heatmap.py)
                                                           │      similarity_maps(page, query)
                                                           └─▶ (optional) answer (generator.py)
                                                                  retrieve-then-read over page images
```

- **Index (offline):** `engine.build_index` walks a folder, rasterizes each page
  (`pdf.render_page_images`), embeds every page image into ColPali patch multivectors
  (`embedder.embed_pages`), and stores them (`store.build_from`). The index persists to
  disk so `serve`/`query` reopen it without re-embedding (`engine.open_index`).
- **Retrieve:** `store.search(query, top_k)` embeds the query and scores it against
  every page with **MaxSim** (late interaction). Backend-agnostic: `MemoryStore` does it
  brute-force in Python; `QdrantStore` uses Qdrant's native `MAX_SIM` comparator.
- **Explain:** `embedder.similarity_maps(page_image, query)` returns per-content-token
  similarity grids; `heatmap.overlay` upsamples one onto the page as an aligned overlay.
- **Answer (optional):** `generator.answer(question, page_images, ...)` sends the top
  page images to any OpenAI-compatible vision endpoint and returns a grounded answer.

## Module map & contracts

| Module | Role | Key contract |
|---|---|---|
| `config.py` | env-driven `Settings` | `get_settings()`; `Settings.vlm_enabled` |
| `pdf.py` | PDF → images + text | `render_page_images(path) -> [PIL]`, `extract_page_texts`, `Page(doc,page,text)` |
| `embedder.py` | ColPali model wrapper | `embed_pages(images)`, `score(query, embs)`, `similarity_maps(img, query) -> (tokens, maps)` |
| `heatmap.py` | grid → overlay | `overlay(page, grid) -> PIL`, `to_data_uri` |
| `store.py` | page store | `build_from(records, images, embs)`, `search(q,k) -> [(Page,score,page_id)]`, `get_image(page_id)` |
| `generator.py` | optional answer model | `answer(question, images, *, base_url, api_key, model) -> str` |
| `engine.py` | orchestration | `build_index(dir, settings)`, `open_index(settings)` |
| `app.py` | HTTP + UI | `/api/search`, `/api/image`, `/api/heatmap`, `/api/ask`, `/api/status` |
| `cli.py` | CLI | `index`, `query`, `serve`, `info` |

## Two backends behind one interface

`store.py` exposes the same shape for both backends — `build_from` / `search` /
`get_image` / `__len__` — so everything downstream is backend-blind. Page **images** are
always persisted to `<data_dir>/images/`; the vector index lives in memory (`MemoryStore`)
or Qdrant (`QdrantStore`). Pick with `COLPALI_STORE=memory|qdrant`. See [SCALING.md](SCALING.md).

## Design choices

- **No OCR, ever.** Retrieval and (optional) answering both read page *pixels*.
- **Interpretability is first-class.** The retriever can always explain a match as a
  heatmap, because it keeps patch-level multivectors (not a single pooled vector).
- **Vendor-neutral generation.** The answer step is plain HTTP to an OpenAI-compatible
  endpoint — no SDK, no provider assumption. Off unless `VLM_BASE_URL` is set.
- **Config over code.** Nothing hard-codes a model id, endpoint, or key.
