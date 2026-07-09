# Extending — the road to a top-tier system

colpali-rag is deliberately small and modular so each upgrade is one file and one clear
seam. This is the roadmap from a solid base to a best-in-class system — each item names
**where it plugs in** and a **prompt** you can hand an agent to build it.

## The base you're starting from

Already working: PDF → ColPali multivectors → MaxSim retrieval (memory + Qdrant) →
per-token heatmaps → optional vendor-neutral answering → web UI + CLI. So every item
below is *additive* — nothing here requires a rewrite.

## Retrieval quality

**1. Second-stage reranking.** Retrieve top-k with ColPali, then rerank with a
cross-encoder / MonoVLM reranker before answering.
- *Where:* new `src/colpali_rag/rerank.py`, called in `engine`/`store.search`.
- *Prompt:* “Add a config-gated reranking stage: after `store.search` returns top-k,
  rerank with an Apache-licensed visual reranker and re-order the results.”

**2. Agentic / multi-query retrieval.** Expand or decompose the query, retrieve for each,
and fuse — with a grade-and-retry loop for weak retrievals.
- *Where:* wrap `store.search` in a new `retrieve()` in `engine.py`.
- *Prompt:* “Add query expansion + reciprocal-rank fusion over multiple sub-queries, and
  a CRAG-style grade→rewrite loop when the top score is below a threshold.”

**3. Hybrid retrieval.** Combine ColPali visual retrieval with sparse/BM25 over the
extracted page text, fused with RRF (Qdrant's Query API can do this server-side).
- *Where:* `store.py` (add a text channel + fusion).

## Answer quality

**4. Structured, cited answers.** Make the answer model return JSON with per-claim page
citations and a confidence, instead of free text.
- *Where:* `generator.py` (request a JSON schema; parse + validate).
- *Prompt:* “Have `generator.answer` return `{answer, claims:[{text, page, confidence}]}`
  and surface the citations in the UI, each linking to that page's heatmap.”

**5. Self-check / critic pass.** Re-present the cited pages and ask the model to verify
each claim is supported; drop or flag unsupported ones.
- *Where:* `generator.py` (a second bounded call).

## Scale & ops

**6. Pooling + quantization + two-stage Qdrant.** The recipe in [SCALING.md](SCALING.md).
- *Where:* `store.py::QdrantStore` (add pooled prefetch vectors + `query_points` prefetch).

**7. Incremental indexing.** Add/remove documents without re-embedding the corpus
(content-hash ids, `recreate=False`).
- *Where:* `store.py`, `engine.py`.

## Trust & measurement

**8. Evaluation harness.** A labeled `query → gold page` set scored with recall@k / nDCG@k
on every change to retrieval, plus citation-precision for answers.
- *Where:* new `src/colpali_rag/eval.py` + `tests/`.
- *Prompt:* “Add an eval harness that loads `eval.jsonl` (query, expected page) and reports
  recall@k / nDCG@k for `store.search`; wire it into CI.”

**9. Hallucination gate on answers.** Score answer faithfulness against the cited pages;
block/flag answers below a threshold.

## Suggested order

Highest leverage first: **4 (cited answers) → 8 (eval) → 1 (rerank) → 2 (agentic) →
6 (scale)**. Cited answers + an eval harness make everything after measurable, so quality
gains stop being guesswork.

## Ground rules to keep

- **No OCR.** Retrieval and answering read pixels; keep it that way.
- **Config over code.** New knobs (reranker, thresholds, pooling) go in `config.py` +
  env, never hard-coded.
- **Commercial-clean models.** Apache/MIT base for any model you add (see [COLPALI.md](COLPALI.md)).
- **Vendor-neutral generation.** Keep `generator.py` talking plain OpenAI-compatible HTTP.
