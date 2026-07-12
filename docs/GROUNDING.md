# Grounded answers — structure, faithfulness, and a stateless cloud pipeline

Retrieval is only half of RAG. The other half — the *answer* — is where systems quietly
lie. This document is about making answers **provably grounded** in the pages they cite,
and running the whole thing as a stateless cloud pipeline. It's opinionated on purpose.

All of it is **off by default** — the CPU, zero-infra path is untouched. You turn each
piece on when you need trust or scale.

---

## 1. Why free-text answers are dangerous

A vanilla RAG answer is a paragraph of prose. Two failure modes hide inside it:

- **Fabricated citations.** If the model is handed page images with no labels, it *cannot*
  know which image is page 3 — so "see page 3" is a guess dressed as a fact. (This was a
  real bug here until images were page-labelled.)
- **Unsupported claims.** The model can write a fluent sentence the cited page never
  says. Prose gives you no seam to check it.

You cannot audit, gate, or measure a paragraph. You *can* audit a structure. So the first
move is to stop answering in prose.

---

## 2. Enforced structure — the shape of a trustworthy answer

The answer model is required to return:

```json
{ "answer": "…",
  "claims": [ { "text": "…", "cites": [1, 2], "confidence": 0.0 } ] }
```

Two deliberate design choices:

- **Cite by bracket index, not page number.** Each attached image is labelled `[1]`,
  `[2]`, … and the model cites `[1]`. Indices are resolved *here* to the real
  `(doc, page, page_id)`. Models reliably echo `[1]` but mangle `report.pdf::p3` — and a
  bare "page 3" is ambiguous across interleaved documents. Indexing removes the ambiguity;
  an out-of-range index is flagged as a **citation hallucination**, not silently coerced.
- **`confidence` is informational, never load-bearing.** Small models emit `0.9` for
  everything. We keep the field (strict schemas require it) but *never* gate on it — the
  gate uses an independent judge (below), not the model's opinion of itself.

**Enforcement is portable.** There is no single way to force JSON across every
OpenAI-compatible server, so `answer_structured` runs a cascade and **caches the winning
tier in memory** (no stale on-disk pin):

1. `response_format={"type":"json_schema", …, "strict":true}` — grammar-constrained on
   servers that support it (hosted, recent vLLM, Ollama ≥0.5, LM Studio).
2. `response_format={"type":"json_object"}` + the schema in the prompt.
3. Plain "return only JSON" prompt.

A tier is **demoted only on an explicit 400/422** (the server rejecting the format). Every
tier is parsed defensively — `json.loads` → balanced-brace extract → validate → one
corrective retry → and, if all else fails, the whole answer becomes a single free-text
claim so the endpoint **never 500s**. Pin a tier with `ANSWER_STRUCTURED_MODE` if you know
your endpoint. Enable with `ANSWER_STRUCTURED=true`.

> Honest limit: a server can *accept* `json_schema` and ignore it (return unconstrained
> text that happens to parse). There's no portable signal that distinguishes "enforced" from
> "the model volunteered JSON." So structure buys you a checkable shape — not a guarantee the
> decoder was constrained. The faithfulness check is what turns shape into trust.

---

## 3. Faithfulness — don't trust the model's own citations

A structured citation says *which* page the model claims to have used. It does **not** say
that page actually supports the claim. Faithfulness closes that gap: a **separate vision
judge re-reads the cited page image(s)** for each claim and rules `supported` / `partial` /
`unsupported`, using only the pixels.

- **The judge must be a different endpoint** (`JUDGE_BASE_URL`). A model grading its own
  homework is not a check; reusing the generator requires `JUDGE_ALLOW_SAME_ENDPOINT=true`
  and logs a warning.
- **Multi-page claims** are judged against *all* their cited pages together (a claim
  synthesized from pages 2+3 isn't "unsupported" just because page 2 alone doesn't say it).
- **The gate** (`FAITHFULNESS_GATE`): `off` (compute nothing), `flag` (attach verdicts, hide
  nothing), or `withhold` (if overall faithfulness < threshold, replace the *whole* answer
  with "could not verify"). We do **not** try to mask sentences out of a prose answer —
  that leaves the hallucinated text visible; withholding is all-or-nothing.
- Outputs: a `faithfulness` score (mean support), `citation_precision` (supported ÷ cited),
  the per-claim verdicts, and two hallucination signals — **citation hallucination** (cited
  an image that wasn't attached) and **attribution hallucination** (cited a page that
  doesn't support the claim).

> Honest limits (say these to stakeholders): the judge proves a claim is *consistent with*
> the cited page — not that the model causally used it. Judges can be wrong or biased.
> There is **no cheap lexical fallback that works** on the visual/scanned documents ColPali
> exists for (extracted text is empty or OCR-noisy there), which is exactly why faithfulness
> is off until you stand up a judge — turning it on costs a judge call per claim. Don't ship
> "verified"; ship "checked against the cited pages, N% supported."

---

## 4. The stateless cloud pipeline

The three storage concerns are cleanly separated so the app holds no state:

```
 vectors      → vector store (Qdrant server)           COLPALI_STORE=qdrant, QDRANT_URL
 page images  → object storage (S3-compatible)         STORAGE_BACKEND=s3, STORAGE_*
 answers      → an LLM/vision endpoint (OpenAI-compat)  VLM_BASE_URL (+ JUDGE_BASE_URL)
```

- **Object storage** is a pluggable adapter (`ArtifactStore` protocol): `local` (default,
  byte-identical to the old on-disk path — existing indexes keep working) or an
  **S3-compatible** backend (AWS S3 / MinIO / R2 / … via one client, endpoint + path
  addressing). Config names are generic (`STORAGE_*`); no vendor is assumed. `get` returns
  `None` only for a genuinely missing object — auth/network errors raise, so
  misconfiguration never masquerades as "image not found."
- **Serving** defaults to **proxy** (the app streams the bytes, keeping access control at
  the app). Presigned redirects are opt-in (`STORAGE_SERVE_MODE=presigned`) — faster, but a
  presigned URL is a bearer token for the object, so only use it on non-gated corpora.
- **The answer/judge endpoints** are any OpenAI-compatible service — self-hosted or hosted.
  Nothing is named; you point the URLs at whatever you run, and those secrets live in your
  private `.env`, never the repo.

Result: web tier + vector DB + object storage + LLM endpoint, each independently scalable;
the app process is disposable.

---

## 5. How to make it god-tier

1. **Turn on structure and a real separate judge.** `ANSWER_STRUCTURED=true`,
   `FAITHFULNESS_GATE=flag` first (observe), then `withhold` once you trust it. Use a
   *strong* judge model on `JUDGE_BASE_URL`, different family from the generator.
2. **Measure, don't vibe.** Build a labelled eval set and run `colpali-rag eval`; track
   `citation_precision` from the faithfulness report over time. Calibrate
   `FAITHFULNESS_MIN_SCORE` on your corpus (thresholds don't transfer across models).
3. **Feed the answer better pages.** Grounding quality is capped by retrieval — use a
   strong retriever (`colqwen2-v1.0`/`Ops-Colqwen3-4B` on GPU), enable the reranker, and
   set `ANSWER_MIN_SCORE` so off-topic questions never reach the model.
4. **Go stateless.** Qdrant + object storage + a horizontally-scaled web tier; the answer
   and judge endpoints scale independently of retrieval.
5. **Watch the two hallucination signals**, not the model's confidence. A rising
   citation-hallucination rate means the retrieval/label wiring drifted; a rising
   attribution-hallucination rate means the generator or the pages degraded.

## 6. What NOT to claim

- Not "the model is faithful" — "N% of claims were supported by their cited pages."
- Not "guaranteed grounded" — a judge can be wrong; structure can be volunteered not enforced.
- Confidence numbers are the model's uncalibrated self-report; don't put them in an SLA.
- Presigned URLs are access grants; don't default them on for sensitive corpora.

Config for every knob above is in [PIPELINE.md §9](PIPELINE.md) and `.env.example`.
