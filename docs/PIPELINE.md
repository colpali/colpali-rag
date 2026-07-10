# The colpali-rag pipeline — end to end, and how to connect every piece

This is the single reference for how colpali-rag works and how to wire each part
(models, store, Qdrant, reranker, the answer endpoint). For a quick tour see
[HANDOFF.md](HANDOFF.md); for the upgrade roadmap see [EXTENDING.md](EXTENDING.md).

---

## 1. Mental model

Documents are **images, not text.** Each PDF page is rasterized and embedded by a
ColPali (ColVision) model into ~1,000 patch vectors; a query is scored against every
page with **MaxSim** late interaction (ColBERT-style). No OCR — tables, diagrams,
scans, and dense layouts are read as pixels. Three user surfaces sit on this:

- **Search** — ranked pages for a query.
- **Heatmap** — *where on the page* the model matched, per query word.
- **Answer** (optional) — a vendor-neutral vision model reads the top pages and
  answers, each page labelled so citations are verifiable.

---

## 2. End-to-end data flow

```
 PDFs ─▶ pdf.render_page_images / extract_page_texts     (Stage 1: ingestion, no OCR)
      ─▶ embedder.embed_pages   (Stage 2: ColPali patch multivectors)
      ─▶ store.build_from       (Stage 3: MemoryStore | QdrantStore, persisted)

 query ─▶ engine.retrieve(store.search  ─[opt]▶ rerank)   ─▶ top pages
                                                            ├─▶ embedder.similarity_maps ─▶ heatmap.overlay   (Stage 4)
                                                            └─▶ generator.answer (labelled page images)       (Stage 5, optional)
```

Modules: `pdf.py · embedder.py · store.py · engine.py · rerank.py · heatmap.py ·
generator.py`, driven by `config.py`, exposed by `app.py` (HTTP + UI) and `cli.py`.

---

## 3. Stage 1 — ingestion (`pdf.py`)

- `render_page_images(pdf, dpi, max_dim)` → PIL RGB per page (pypdfium2, Apache/BSD;
  no OCR, no Poppler). Downscales with LANCZOS so small text/diagram patches stay sharp.
- `extract_page_texts(pdf)` → text per page (empty on scanned pages), used for UI
  snippets and `text_coverage()` (a scanned-corpus warning; retrieval still works on
  pixels).
- `doc_id(pdf, docs_dir)` = **corpus-relative path** so two same-named PDFs in
  different subfolders don't collide on `page_id`.
- Unreadable PDFs raise `PdfRenderError`; `engine.build_index` skips them (logged),
  never aborting the whole run.

Knobs: `COLPALI_DPI` (150), `COLPALI_MAX_DIM` (1600).

---

## 4. Stage 2 — embedding (`embedder.py` + `models_registry.py`)

The **declarative registry** (`models_registry.py`) is the single source of truth
mapping a model id → `(model_cls, proc_cls, family, heatmap, base_license)`. An
unknown id raises `UnsupportedModel` (no silent mis-load); a renamed engine class
raises `EngineCapabilityError` naming the version it saw. Set `COLPALI_FAMILY` to
force a family for a brand-new checkpoint.

- `embed_pages(images)` → one multivector per page. Padding is stripped per page via
  the attention mask, so `COLPALI_BATCH_SIZE>1` doesn't fold pad tokens into the
  stored vectors.
- `score(query, page_embs)` → MaxSim scores. `embed_query_raw` / `page_to_list` feed
  the Qdrant store. `dim` is inferred from a query embedding.
- The base license is a **baked** field (resolved at curation time, never fetched
  from the network — works offline); a non-Apache/MIT base logs a warning.

> ⚠️ colpali-engine is **pinned** (`==0.3.17`) because dispatch + the heatmap depend
> on its internal API surface. Bump deliberately and re-run the model-family contract
> test (`test_registry_contract_against_real_engine`).

---

## 5. Stage 3 — storage & retrieval (`store.py`, `engine.py`)

Both stores share one shape (`build_from` / `search` / `get_image` / `__len__`) and
persist page images to `<data_dir>/images/` + metadata to `<data_dir>/records.json`.

| Store | How | When |
|---|---|---|
| `memory` | brute-force MaxSim in Python, embeddings on disk | default; a few thousand pages |
| `qdrant` (embedded) | on-disk `QdrantClient(path=…)` | persistence, no server |
| `qdrant` (server) | `QDRANT_URL` | millions of pages |

**Identity guard.** `records.json` stores `{model, dim, schema_version}`. `open_index`
compares the configured model against it *from metadata first* — so changing
`COLPALI_MODEL` after indexing raises `IndexModelMismatch` immediately (with a
re-index hint) instead of silently scoring one model's query vectors against
another's page vectors, and without downloading the wrong model.

`engine.retrieve(store, query, top_k, reranker=…)` is the retrieval entry point:
first-stage MaxSim fetches a shortlist, an optional reranker re-orders it, top_k
returned.

---

## 6. Stage 3b — reranking (`rerank.py`, config-gated, OFF by default)

The single biggest accuracy lever after the base model. First stage keeps top-N;
the reranker re-scores the top_k page images pointwise and re-orders them.

- Backend `monoqwen` → `lightonai/MonoQwen2-VL-v0.1` (**Apache-2.0**, LoRA on
  Qwen2-VL-2B). GPU-oriented (seconds/pair) → **off by default** to protect the
  sub-second CPU path. A load/inference failure degrades to the first-stage order.
- Enable: `RERANK_ENABLED=true` + `pip install '.[rerank]'` (transformers/peft) +
  `RERANK_DEVICE=cuda`. A/B it with `colpali-rag eval --rerank`.
- **Avoid** CC-BY-NC rerankers (e.g. `jina-reranker-m0`) in a shipped product.

---

## 7. Stage 4 — the heatmap (`embedder.similarity_maps`, `heatmap.py`)

Per-content-token similarity grids aligned to the page, on a non-split single-image
pass. **Cross-model:** uses the processor's own sim-maps when present (Idefics3 /
ModernVBert) or a matplotlib-free einsum over masked patch embeddings (ColQwen2/3,
ColPali). Grid geometry from `get_n_patches` when it matches the token count, else a
nearest-aspect factorization. Models with no image-mask API raise `HeatmapUnsupported`
→ the API returns a clean **501**, never a 500. `heatmap.overlay` renders the grid as
an inferno overlay upsampled to the page.

Resolution follows the model: `colSmol` is a coarse 8×8; `colqwen2-v1.0` on GPU is
high-resolution.

---

## 8. Stage 5 — answer (`generator.py`, optional, vendor-neutral)

Off unless `VLM_BASE_URL` is set. `answer(question, images, base_url, model, labels)`
POSTs an OpenAI-compatible `/chat/completions` request with each page image
**preceded by its `Page N of doc:` label**, so the model can cite pages verifiably
(without labels, any page citation is fabricated). Point it at *any* endpoint —
vLLM / Ollama / LM Studio / TGI / hosted — no provider is named. An optional
`ANSWER_MIN_SCORE` gate skips answering when the top retrieval score is too low, so
irrelevant queries don't get confident answers.

---

## 9. Configuration reference (all env / `.env`, no code)

| Setting | Env | Default | Notes |
|---|---|---|---|
| model | `COLPALI_MODEL` | `vidore/colSmol-500M` | any ColVision id |
| family override | `COLPALI_FAMILY` | *(auto)* | force a family for a new checkpoint |
| device | `COLPALI_DEVICE` | `cpu` | cpu \| cuda \| mps (mps → torch==2.5.1) |
| batch size | `COLPALI_BATCH_SIZE` | 1 | pad-safe at >1 |
| dpi / max_dim | `COLPALI_DPI` / `COLPALI_MAX_DIM` | 150 / 1600 | rasterization |
| store | `COLPALI_STORE` | `memory` | memory \| qdrant |
| data dir | `COLPALI_DATA_DIR` | `colpali_data` | index + images |
| qdrant | `QDRANT_URL` / `QDRANT_API_KEY` | *(embedded)* | server vs on-disk |
| collection | `COLPALI_COLLECTION` | `documents` | |
| rerank | `RERANK_ENABLED` / `RERANK_BACKEND` / `RERANK_DEVICE` | off / monoqwen / cuda | |
| answer model | `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` | *(off)* | OpenAI-compatible |
| answer gate | `ANSWER_MIN_SCORE` / `ANSWER_TOP_K` | off / 3 | relevance gate |
| server | `COLPALI_HOST` / `COLPALI_PORT` | 127.0.0.1 / 8000 | |

---

## 10. Connecting the pieces — recipes

**A. CPU zero-infra (default).**
```bash
pip install -e '.[rag,api]'
colpali-rag index ./pdfs && colpali-rag serve
```

**B. GPU + Qdrant server.**
```bash
docker compose up -d qdrant
export COLPALI_MODEL=vidore/colqwen2-v1.0 COLPALI_DEVICE=cuda \
       COLPALI_STORE=qdrant QDRANT_URL=http://localhost:6333
colpali-rag index ./pdfs && colpali-rag serve
```

**C. Enable answers (any OpenAI-compatible vision endpoint).**
```bash
export VLM_BASE_URL=http://localhost:8000/v1 VLM_MODEL=<your-model>
# export VLM_API_KEY=...   # if required
colpali-rag serve      # an "Ask" box appears
```

**D. Enable reranking (GPU).**
```bash
pip install -e '.[rerank]'
export RERANK_ENABLED=true RERANK_DEVICE=cuda
colpali-rag query "…" --rerank        # or add --rerank to `eval`
```

---

## 11. Models & licensing

Clean CPU→GPU ladder (Apache/MIT base *and* adapter, verified):

| Tier | Model | License | Note |
|---|---|---|---|
| CPU / dev | `vidore/colSmol-500M` | Apache-2.0 | default; coarse heatmap |
| CPU (newer) | `ModernVBERT/colmodernvbert` | MIT end-to-end | near-ColPali accuracy, designed for CPU |
| GPU / prod | `OpenSearch-AI/Ops-Colqwen3-4B` | Apache-2.0 | strong clean pick; Matryoshka dims |
| GPU (mature) | `nomic-ai/colnomic-embed-multimodal-7b` | Apache-2.0 | well-established |

**Hard-exclude in a shipped product** (the registry flags these): `colqwen2.5-*` and
`colnomic-3b` (Qwen *Research*, non-commercial **base** despite an MIT adapter tag),
`nvidia/nemotron-colembed-*` (CC-BY-NC — even though they top the raw leaderboard),
`jina-reranker-m0` (CC-BY-NC), PaliGemma-based `colpali-v1.x` (Gemma use policy).
Benchmark numbers move monthly — re-check before pinning a default.

---

## 12. Evaluation (`eval.py`)

Measure retrieval, don't guess. `eval.jsonl` lines: `{"query": "...",
"gold_page_ids": ["doc::p3", ...]}` (gold ids reuse `store.page_id()`).

```bash
colpali-rag eval eval.jsonl --k 1,5,10 [--rerank] [--report report.json]
```

Reports recall@k, nDCG@k, MRR. Honest scope: numbers are only as good as your labeled
set; bootstrap ~50–100 pairs (e.g. VLM-generate queries per page, LLM-judge, then a
human confirm) before nDCG deltas are meaningful — and treat any single-config number
as directional until A/B'd on your corpus. A model-free unit test checks the metric
math; real retrieval numbers are an offline job, not the fast CI suite.

---

## 13. API reference (`app.py`)

| Route | Returns / errors |
|---|---|
| `GET /api/status` | model / store / pages / rerank / vlm / heatmap flags |
| `GET /api/search?q=&k=` | ranked pages (reranked if enabled) |
| `GET /api/image?page_id=` | page PNG; 404 if missing |
| `GET /api/heatmap?page_id=&q=` | per-token overlays; **501** if model has no heatmap |
| `GET /api/ask?q=&k=` | grounded answer + cited sources; 503 no VLM, 502 model error, gated if below `ANSWER_MIN_SCORE` |

---

## 14. Extending & operations

- **Protocols** (`protocols.py`): implement `Embedder` / `PageStore` / `Reranker` to
  plug in a backend; tests use fakes against these contracts.
- **Testing seams:** fake embedder, embedded Qdrant (`QdrantClient(path=…)`), httpx
  stub for the answer path, FastAPI `TestClient`. See `tests/test_v2.py`.
- **Threading:** the model forward isn't re-entrant; the app serializes it behind a
  lock. **MPS:** pin `torch==2.5.1`. **Security:** `torch.load(weights_only=False)` on
  the memory store trusts `data_dir` — treat it as trusted; be wary of feeding
  untrusted PDF text into the answer model (prompt-injection surface).

---

## 15. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Heatmap "not available" (501) | model has no image-mask API — use colSmol / colqwen2 / colmodernvbert. |
| `IndexModelMismatch` on serve | `COLPALI_MODEL` differs from the index — re-index or match it. |
| Low `text_coverage` warning | scanned/image-only PDFs — retrieval still works (pixels), snippets won't. |
| `EngineCapabilityError` | colpali-engine version drift — pin `==0.3.17` or update the registry. |
| Slow / OOM on CPU with rerank | reranking is GPU-oriented — keep `RERANK_ENABLED=false` on CPU. |
