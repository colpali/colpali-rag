# Use it with your own documents / your own RAG

A practical, copy-paste path from "a folder of PDFs" to "a working visual RAG you can query, embed
in your own pipeline, ground answers with, and measure." Everything is generic — point it at any
PDFs and any OpenAI-compatible model endpoint.

## 0. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[rag,api]'            # core engine + web service
# add ',studio' for the structured-output studio (CSV/Excel upload, canvas UI)
```

First run downloads the model (~1 GB for the default CPU model `colSmol-500M`).

## 1. Index your PDFs

```bash
colpali-rag index ./my_pdfs            # searched recursively; no OCR — it reads page pixels
```

This rasterizes every page, embeds it, and persists the index under `./colpali_data`
(`COLPALI_DATA_DIR`). Re-run after adding PDFs.

## 2. Run it

```bash
colpali-rag serve                      # visual web UI at http://127.0.0.1:8000  (search + heatmaps)
# or the terminal:
colpali-rag query "thermal derating curve" --k 5
# or the structured-output studio:
colpali-rag studio
```

## 3. (Optional) plug in your own answer model

Answering is off until you point it at **any OpenAI-compatible `/chat/completions` vision
endpoint** (self-hosted vLLM / Ollama / LM Studio / TGI, or a hosted one). No provider is
hard-coded:

```bash
export VLM_BASE_URL=https://your-endpoint/v1
export VLM_MODEL=<your-model-name>
export VLM_API_KEY=<key-if-needed>
colpali-rag serve                      # an "Ask" box appears; answers cite the pages they used
```

## 4. Use the engine inside your OWN RAG pipeline

The retriever is a library — drop it into whatever you're building:

```python
from colpali_rag.config import Settings
from colpali_rag.engine import build_index, open_index, retrieve

settings = Settings.from_env()                 # or Settings(model="vidore/colqwen2-v1.0", device="cuda")

# build once…
store, embedder, info = build_index("./my_pdfs", settings, progress=print)
# …or reopen a persisted index without re-embedding:
# store, embedder = open_index(settings)

for page, score, page_id in retrieve(store, "thermal derating curve", top_k=5, settings=settings):
    print(f"{score:.3f}  {page.doc}  p.{page.page}")
    image = store.get_image(page_id)           # the PIL page image (feed to your own VLM)
    text  = page.text                          # extracted page text (may be empty for scans)
```

`retrieve(...)` returns `[(Page, score, page_id)]`, best first. Pass `settings=` to pick up hybrid
retrieval; pass a `reranker=` for a second stage. That's the whole integration surface — build/open
an index, call `retrieve`, use the returned page images/text in your own generation.

## 5. (Optional) turn on the advanced features

| Want | Set | Doc |
|---|---|---|
| Exact IDs/codes found even on blurry pages | `COLPALI_HYBRID_ENABLED=true` | [RETRIEVAL.md](RETRIEVAL.md) |
| Structured outputs limited to a controlled vocabulary from an uploaded table | `CATALOG_ID_COL=<col>` + `COLPALI_CATALOG_GATE=flag\|withhold` | [CONSTRAINTS.md](CONSTRAINTS.md) |
| Follow-up questions retrieve well (rewrite "and the current?" into a standalone query) | `COLPALI_QUERY_REWRITE=true` | uses session history + your model |
| Scale past the in-process store | `COLPALI_STORE=qdrant` + `QDRANT_URL=...` | [SCALING.md](SCALING.md) |
| Page images on S3-compatible object storage | `STORAGE_BACKEND=s3` + `STORAGE_*` | [GROUNDING.md](GROUNDING.md) |

## 6. Measure your accuracy

Label a few queries with their relevant pages (`eval.jsonl`), then:

```bash
colpali-rag eval eval.jsonl --k 1,5,10     # coverage@k / recall@100 / MAP / nDCG / MRR
```

A/B two configurations (e.g. hybrid on vs off) with a significance test using `compare_runs` — see
[EVAL.md](EVAL.md). Ship a change only when the delta is significant, not on a hunch.

## 6b. See what happened (step trace + saved run summaries)

Every studio generation logs a step-by-step trace to the console (retrieve → sources studied →
catalog constraint → model call(s) → repair passes → result) and finishes with a summary of what
it studied and produced. Set the level and, optionally, a folder to persist a `.json` + `.txt`
summary of every run:

```bash
export COLPALI_LOG_LEVEL=INFO          # INFO shows the per-generation step trace (default)
export COLPALI_RUN_LOG_DIR=./runs      # write a JSON + text summary of each generation here
```

Each run file records the request, the exact pages/tables it read (with scores), the nodes and
connections it drew, the model's reasoning/assumptions, and every constraint/repair check
(dropped, remapped, missing-required, infeasible, withheld). The same summary is returned in the
`/generate` API response, so the UI or your own code can show it. Great for debugging and audit.

## 7. Health-check the index

```bash
colpali-rag doctor        # model + adapter + backend + schema, and an embedding unit-norm check
```

The unit-norm check matters if you move from the in-memory store (dot product) to Qdrant (cosine)
— they agree only when embeddings are unit-norm, and `doctor` tells you.

## 8. (Later) fine-tune to your domain

When you eventually train a domain adapter, point the engine at it — no code change:

```bash
export COLPALI_ADAPTER_PATH=/path/to/adapter      # a PEFT/LoRA adapter dir or hub id
colpali-rag index ./my_pdfs                       # re-index; the adapter is baked into the index identity
```

A fine-tuned index can only be queried with the same adapter (the identity guard enforces it, so
you can't accidentally mix base and fine-tuned vectors). The training itself is external — see
[EXTENDING.md](EXTENDING.md) for the roadmap and the false-negative pitfalls to avoid.

---

**That's the whole loop:** index → query (or integrate) → optionally ground with your model →
measure → health-check → (later) fine-tune. Start on the in-process store with the default CPU
model; scale up only when a measurement says you need to.
