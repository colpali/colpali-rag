# Studio — structured, cited outputs from your documents

The studio is an application layer on top of the `colpali_rag` engine. You **select which
sources apply**, **upload your own CSV / Excel / notes**, and **describe what you want** in
plain language. A model reads the retrieved pages and returns a **structured, cited output**
— a validated object of typed **nodes** and labeled **connections** where every element can
cite the page or spreadsheet row it came from — rendered as an interactive, explorable
canvas and exportable to portable formats.

Nothing here is domain-specific. Point it at any PDF corpus and any set of tabular files.

---

## Why this shape (the important part)

Anything a model composes from memory is a **confident guess**. The moment that output is
something a person builds on, an ungrounded guess is a liability. So the studio rests on
three deliberate constraints:

1. **The output is a cited structure, not prose.** The model must emit an enforced object
   — typed `nodes`, `connections`, and `groups` — under a strict JSON schema. No free-text
   "here you go." A structure you can validate, diff, render, and **attach evidence to**.

2. **Citations are resolved, not trusted.** The model cites sources by **bracket index**
   (`[1]`, `[2]`) into the pages/tables it was actually shown, in the order shown. We
   resolve those indices back to real source ids server-side and **flag any out-of-range
   index as a hallucinated citation**. The model never hands us an id to mangle. (Same
   discipline as the engine's grounded answers — see [GROUNDING.md](GROUNDING.md).)

3. **It degrades honestly, never crashes.** Malformed JSON → one corrective retry → next
   schema tier → a deterministic demo result. A missing model → demo mode. A connection to
   a node that doesn't exist → dropped and counted, not rendered as a phantom. The UI shows
   `demo` vs `grounded`, and surfaces dropped links / bad citations as badges rather than
   hiding them.

The point isn't "a model produced something." It's **an output you can trace back to the
documents, whose gaps are visible.**

---

## The pipeline

```
 sources (PDF) ─▶ colpali_rag engine: rasterize ─▶ ColPali embed ─▶ store  (the document DB:
                                                                      memory | Qdrant)
                                                            │
 request ("…") ──────▶ ColPali retrieval ───────────────────┴─▶ top pages (images)
 selected sources ────▶ (scopes retrieval)                                 │
 uploaded CSV / Excel / notes ─▶ compact citable text ─────────────────────┤
                                                                            ▼
                      LLM wrapper (any OpenAI-compatible vision endpoint)
                      enforced JSON: json_schema → json_object → prompt cascade
                                                                            │
                      validate + resolve [n] cites → real source ids ───────┤
                                                                            ▼
       interactive canvas (nodes + connections)   ·   portable exports
        (each element cites its source)                (Mermaid, draw.io)
```

- **The document DB** is the engine's store: in-process for hundreds of pages, **Qdrant**
  (native multivector MAX_SIM) for scale — set `COLPALI_STORE=qdrant` + `QDRANT_URL`. Page
  images live on local disk or S3-compatible **object storage** (`STORAGE_BACKEND`).
- **The LLM wrapper** is vendor-neutral: any OpenAI-compatible `/chat/completions` vision
  endpoint (self-hosted vLLM / Ollama / LM Studio / TGI, or hosted). Set `VLM_BASE_URL`,
  `VLM_MODEL`, `VLM_API_KEY`. No provider is named or assumed.
- **No model configured?** The studio still runs in **demo mode**: it infers a plausible
  structure from your request text so you can see the whole UI end-to-end with zero
  infrastructure.

---

## Constrain outputs to a vocabulary (optional)

By default a node can be labeled anything. When the output must only use entities from a
source-of-truth table you upload (a product list, a component registry, an inventory — anything
with an id column), turn on the **closed-vocabulary constraint**: every node is projected onto the
compiled vocabulary (confident match → canonical id, otherwise dropped and flagged), connections
and required items are verified, the model is re-prompted with the specific violations, and the
output is emitted-with-flags or **withheld** if it still can't be grounded. Off by default; set
`CATALOG_ID_COL` + `COLPALI_CATALOG_GATE` to enable. Full details, config, and metrics in
**[CONSTRAINTS.md](CONSTRAINTS.md)**.

## Run it

```bash
pip install -e '.[rag,api,studio]'      # engine + web service + studio (uploads, Excel)

# (optional) index a corpus so outputs are grounded in real pages
colpali-rag index ./pdfs

# (optional) point at any OpenAI-compatible vision endpoint so a model does the work
export VLM_BASE_URL=http://localhost:8000/v1
export VLM_MODEL=<your-model>

# backend (API + serves the built UI at :8000)
colpali-rag studio

# frontend — dev server with hot reload (proxies /api to :8000)
cd web && npm install && npm run dev      # → http://localhost:5173
```

For a single-port production serve: `cd web && npm run build`, then `colpali-rag studio`
serves the built SPA from `web/dist` at `:8000` (override with `STUDIO_WEB_DIST`).

---

## Structure

```
src/colpali_rag/studio/
  spec.py       the enforced output schema + citation-resolving validator
  generate.py   request → retrieval + tabular context → structured output (+ demo generator)
  tabular.py    CSV (stdlib) / Excel (openpyxl) → compact, citable table summaries
  session.py    in-memory working set: selected sources, uploads, history
  render.py     structured output → portable export formats
  api.py        FastAPI router: /session /sources /upload /select /generate /export /image
  server.py     the studio app: loads the engine index, mounts the API, serves the SPA

web/            React + Vite + Tailwind + interactive canvas frontend
  src/App.tsx              three-pane orchestrator
  src/components/          SourcesPanel · ChatPanel · canvas · node
  src/lib/flow.ts          structured output → canvas nodes/edges (layered layout, kind colors)
  src/api.ts, types.ts     typed client + shared shapes
```

The output schema is the **renderer-agnostic** contract between backend and frontend — the
canvas and every export consume the same object; none of them leaks into the schema. Add a
renderer without touching generation; change generation without touching renderers.

---

## How to make it god-tier

Ordered by value. Each builds on the cited-structure foundation:

1. **Faithfulness on the output.** Reuse the engine's separate-endpoint vision judge
   (`faithfulness.py`) to check each *connection* against its cited pages: does the source
   actually support A → B? Gate `off | flag | withhold` per element. An output whose every
   link is judge-verified is a different class of artifact.
2. **Deterministic layout.** Swap the longest-path grid for ELK/Dagre so large outputs read
   cleanly, and persist user drags so a hand-tuned layout survives regeneration.
3. **Iterative refine chat.** "Merge these two nodes", "drop that link" — send the current
   structure back as context so requests refine the output instead of redrawing it.
4. **Structured tabular grounding.** Today tables are summarized to text; parse them into
   typed rows so a node can cite `sheet.xlsx row 12` exactly and relationships come straight
   from the data.
5. **Output eval set.** Golden (request → expected nodes/connections) pairs; score node
   recall and connection precision the way `eval.py` scores retrieval, so prompt/model
   changes are measured, not vibed.
6. **Multi-output projects & versioning.** Save structures to the object store, diff two
   versions, and export a whole set — the schema is JSON, so this is storage, not rework.

---

## What NOT to claim

- A resolved citation means the element is **consistent with** a page the retriever
  surfaced — not that the source formally specifies it, nor that retrieval found every
  relevant page. It's grounding, not proof.
- `confidence`-style signals and the demo generator's structure are **heuristics**. Demo
  mode infers structure from your sentence; it has read no document.
- The studio does not verify correctness of the result. It turns documents + intent into a
  **traceable draft** a human reviews — faster and more honest than a blank start, not a
  replacement for judgment.
```
