# colpali-rag — visual document search you can *see*

![License: MIT](https://img.shields.io/badge/License-MIT-0c6e7c.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)
![Retrieval: ColPali](https://img.shields.io/badge/retrieval-ColPali%20late--interaction-6C8EBF.svg)
![OCR: none](https://img.shields.io/badge/OCR-none-2C7A4B.svg)

An open-source, **document-agnostic** visual RAG built on [ColPali](https://github.com/illuin-tech/colpali).
Point it at a folder of PDFs; it embeds each **page image** — **no OCR** — retrieves with
ColBERT-style late interaction (MaxSim), and shows a **heatmap of where on the page the
model looked**, per query word. Optionally, ask a question and a vision model reads the
top pages and answers with citations.

- **Reads pixels, not text.** Tables, diagrams, scans, dense multi-column layouts —
  captured directly. No OCR pipeline to mangle them.
- **Explainable retrieval.** Every result shows a heatmap overlay of *where* it matched.
- **Zero-infra to start.** In-process store — no database required. Add Qdrant for scale.
- **Self-contained.** Pure-Python web UI — no Node build, no CDN.
- **Vendor-neutral answers.** The optional answer model is *any* OpenAI-compatible
  endpoint you point it at — no provider lock-in.
- **Hybrid retrieval (optional).** Fuse the visual ranking with a keyword ranking over
  extracted page text (RRF) so exact identifiers aren't lost to a blurry page — auto-off
  on scanned corpora. See [docs/RETRIEVAL.md](docs/RETRIEVAL.md).
- **Constrained structured outputs (optional).** Constrain the studio's node/connection
  outputs to a closed vocabulary compiled from an uploaded table: every node projected
  onto it, connections and required items verified, the model re-prompted to fix
  violations, or the output withheld. See [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md).
- **Measured, not vibed.** A retrieval + structured-output eval harness with coverage@k /
  recall@100 / MAP / graded nDCG and a paired-bootstrap A/B. See [docs/EVAL.md](docs/EVAL.md).

---

## Quickstart (no database needed)

```bash
git clone <this-repo> && cd <this-repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[rag,api]'

colpali-rag index ./pdfs        # any folder of PDFs (searched recursively)
colpali-rag serve               # open http://127.0.0.1:8000
```

First run downloads the model (~1 GB for the default CPU model, `colSmol-500M`). Drop
your own PDFs into a folder and re-run `index`.

Terminal-only:

```bash
colpali-rag query "thermal derating curve" --k 5
colpali-rag info                # what model / store / index is configured
```

---

## What you get in the UI

1. **Search** in natural language across all indexed pages.
2. **Ranked page results** as thumbnails with a relevance score.
3. Click a result → the page opens with a **similarity heatmap overlay**.
4. **Highlight by query term** — click *All* or any word to see that token's heatmap;
   toggle the heat on/off to read the page.
5. **Ask** (optional) — if an answer model is configured, ask a question and get an
   answer read from the top pages, with the source pages cited.

The heatmap uses the model's own late-interaction similarity maps
(`get_similarity_maps_from_embeddings`) on a clean, non-split single-image pass, so the
overlay is geometrically aligned to the page for any ColVision model.

> Heatmap resolution follows the model: `colSmol-500M` gives a coarse grid (fast,
> CPU-friendly); `vidore/colqwen2-v1.0` on a GPU gives sharp, high-resolution maps.

---

## Optional answer model (vendor-neutral)

Answering is off by default (search + heatmaps need no model beyond the retriever).
To enable it, point `colpali-rag` at **any OpenAI-compatible `/chat/completions` vision
endpoint** you run — self-hosted (vLLM / Ollama / LM Studio / TGI) or hosted:

```bash
export VLM_BASE_URL=http://localhost:8000/v1
export VLM_MODEL=<your-model-name>
# export VLM_API_KEY=...        # if your endpoint needs one
colpali-rag serve
```

An "Ask" box appears in the UI. The model only ever sees the **retrieved page images**,
so answers are grounded in your documents. No provider is named or assumed — swap the
endpoint freely.

---

## Configuration (all via `.env` or flags — never code)

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Model | `COLPALI_MODEL` | `vidore/colSmol-500M` | Any ColVision id. GPU: `vidore/colqwen2-v1.0`. |
| Device | `COLPALI_DEVICE` | `cpu` | `cpu` \| `cuda` \| `mps` (mps needs `torch==2.5.1`). |
| Store | `COLPALI_STORE` | `memory` | `memory` (zero infra) \| `qdrant` (scale). |
| Data dir | `COLPALI_DATA_DIR` | `colpali_data` | Where page images + embeddings persist. |
| Qdrant URL | `QDRANT_URL` | *(embedded on-disk)* | Set to `http://localhost:6333` for a server. |
| Collection | `COLPALI_COLLECTION` | `documents` | Qdrant collection name. |
| Answer model | `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` | *(off)* | Any OpenAI-compatible vision endpoint. |
| Hybrid retrieval | `COLPALI_HYBRID_ENABLED` | `false` | Fuse lexical + visual (RRF). [docs/RETRIEVAL.md](docs/RETRIEVAL.md). |
| Vocabulary constraint | `CATALOG_ID_COL` / `COLPALI_CATALOG_GATE` | *(off)* | Constrain studio outputs to an uploaded table. [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md). |
| Host / Port | `COLPALI_HOST` / `COLPALI_PORT` | `127.0.0.1` / `8000` | Web UI bind. |

Every flag mirrors an env var, e.g. `colpali-rag index ./pdfs --model vidore/colqwen2-v1.0 --device cuda`.

---

## Scaling with Qdrant (optional)

The in-process store does exact MaxSim in Python — fine for hundreds–thousands of pages.
For more, use **Qdrant** (native multivector MAX_SIM):

```bash
docker compose up -d qdrant                       # ships in docker-compose.yml
export COLPALI_STORE=qdrant QDRANT_URL=http://localhost:6333
colpali-rag index ./pdfs
colpali-rag serve
```

Without `QDRANT_URL`, `COLPALI_STORE=qdrant` uses an **embedded on-disk** Qdrant under
`COLPALI_DATA_DIR/qdrant` — persistent, still no server to run.

---

## How it works

```
 PDFs ─▶ rasterize (pypdfium2) ─▶ ColPali embed (per-page patch multivectors)
                                         │
                                         ▼
                        store (memory MaxSim  |  Qdrant MAX_SIM)
                                         │
        query ──▶ embed ──▶ MaxSim rank ─┴─▶ top pages
                                         │
        selected page + query ──▶ similarity maps ──▶ heatmap overlay (per token)
        top pages + question   ──▶ (optional) vision model ──▶ grounded answer + citations
```

- **Retrieval:** ColBERT-style late interaction (MaxSim) over page-image patch
  multivectors — state of the art on visually rich documents.
- **Interpretability:** per-query-token similarity grids aligned to the page pixels,
  rendered as an inferno heatmap.
- **License-clean by default:** `colSmol-500M` and `colqwen2-v1.0` are both Apache-2.0.

## Module map

```
src/colpali_rag/
  config.py     env-driven settings (model, device, store, qdrant, answer model)
  pdf.py        PDF → page images + text (pypdfium2; no OCR)
  embedder.py   ColPali wrapper: retrieval + similarity-map interpretability
  heatmap.py    similarity grid → aligned inferno overlay (NumPy + Pillow)
  store.py      MemoryStore (brute-force) + QdrantStore (native MAX_SIM)
  generator.py  optional, vendor-neutral answer model (any OpenAI-compatible endpoint)
  engine.py     build_index / open_index / retrieve (+ optional hybrid RRF fusion)
  lexical.py    char-n-gram BM25 over page text (the lexical channel for hybrid retrieval)
  eval.py       retrieval metrics (coverage@k, recall@100, MAP, nDCG) + paired-bootstrap A/B
  graph_eval.py structured-output metrics (vocabulary adherence, required coverage)
  app.py        FastAPI service + self-contained web UI (search · heatmap · ask)
  cli.py        colpali-rag index | query | serve | studio | info | eval
  studio/       structured, cited output layer over the index (spec, generate, api, render)
    catalog.py  closed-vocabulary compiler + matcher + projection/verify/repair
```

## Studio (optional)

An application layer on top of the engine: **select which indexed sources apply**, **upload
your own CSV / Excel / notes**, and **describe what you want**. A model reads the retrieved
pages and returns a **structured, cited output — typed nodes and labeled connections, each
citing the page or row it came from** — as an interactive React canvas you can explore and
export.

```bash
pip install -e '.[rag,api,studio]'
colpali-rag studio                     # backend + built UI at http://127.0.0.1:8000
cd web && npm install && npm run dev    # modern dev UI at http://localhost:5173
```

Runs in **demo mode** with zero infrastructure (structure inferred from the request text);
index a corpus and point `VLM_BASE_URL` at any OpenAI-compatible vision endpoint to ground
it in real pages. See **[docs/STUDIO.md](docs/STUDIO.md)**.

## Documentation

- **[docs/QUICKSTART.md](docs/QUICKSTART.md) — start here: index your own PDFs, query them,
  integrate the retriever into your own RAG pipeline, enable the optional features, and measure.**
- **[docs/PIPELINE.md](docs/PIPELINE.md) — the single end-to-end reference: every stage
  and how to connect every piece (models, Qdrant, reranker, answer endpoint).**
- [docs/HANDOFF.md](docs/HANDOFF.md) — start here / repo map
- [docs/COLPALI.md](docs/COLPALI.md) — late interaction / MaxSim, models + licensing, the heatmap
- [docs/GROUNDING.md](docs/GROUNDING.md) — **structured cited answers + faithfulness checks + the stateless cloud pipeline** (object storage / generic LLM), and how to make answers provably grounded
- [docs/STUDIO.md](docs/STUDIO.md) — **Studio**: chat + source selection + CSV/Excel upload → interactive, cited structured outputs (React frontend), and how to make it god-tier
- [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md) — **closed-vocabulary constraints**: force studio outputs to use only entities from an uploaded table (project, verify, repair, or abstain)
- [docs/RETRIEVAL.md](docs/RETRIEVAL.md) — **hybrid visual + lexical retrieval** (RRF over page text) for exact identifiers
- [docs/EVAL.md](docs/EVAL.md) — **measuring accuracy**: retrieval + structured-output metrics and a paired-bootstrap A/B
- [docs/SCALING.md](docs/SCALING.md) — in-memory vs Qdrant + the multivector scaling recipe
- [docs/EXTENDING.md](docs/EXTENDING.md) — roadmap to a top-tier system, mapped to files & prompts

Measure retrieval accuracy on a labeled set: `colpali-rag eval eval.jsonl --k 1,5,10 [--rerank]`
(reports coverage@k / recall@100 / MAP / graded nDCG; A/B two runs with `compare_runs` — see
[docs/EVAL.md](docs/EVAL.md)).

## Requirements

- Python 3.11+
- `pip install -e '.[rag,api]'` (torch, colpali-engine, pypdfium2, einops, fastapi)
- Optional: Docker for Qdrant; a CUDA GPU for `colqwen2-v1.0` (sharper heatmaps); any
  OpenAI-compatible vision endpoint for the answer feature.
