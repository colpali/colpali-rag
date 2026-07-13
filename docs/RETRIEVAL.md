# Retrieval: visual, and optional hybrid visual + lexical

The core retriever ranks pages by **visual late interaction** (ColBERT-style MaxSim over
page-image patch multivectors) — see [COLPALI.md](COLPALI.md). It reads pixels, so it captures
tables, diagrams, and dense layouts that OCR mangles.

One thing pixels are weak at: an **exact identifier or code** (`AX-1234`, `RS-232/A`, an order
number) printed small or on a blurry scan. The visual channel may know a page is "about that kind
of thing" without locking onto the precise string. The pages already carry extracted text
(`Page.text`), so there is a cheap fix.

## Hybrid retrieval (Reciprocal Rank Fusion)

Turn it on and retrieval fuses two rankings:

- **visual** — the usual MaxSim ranking;
- **lexical** — a character-n-gram BM25 ranking over the extracted page text. Char n-grams (not
  whole words) are used because identifiers are full of punctuation that word tokenizers shatter;
  overlapping char grams keep the run contiguous and score a near-exact hit highly.

The two are combined with **Reciprocal Rank Fusion**: `RRF(page) = Σ_channel 1/(κ + rank)`. RRF
fuses *ranks*, not scores, so it needs no calibration between the two very different score scales
(MaxSim sums vs BM25). A page that either channel ranks highly rises; a page both agree on rises
most.

```bash
export COLPALI_HYBRID_ENABLED=true
colpali-rag serve            # or `studio`, or `query`, or `eval`
```

That's it — every retrieval path (search UI, ask, studio, CLI `query`/`eval`) picks it up.

### It degrades honestly

- **Scanned corpora**: if too few pages have extractable text (below `COLPALI_HYBRID_MIN_COVERAGE`),
  the lexical channel would be noise, so it's skipped and retrieval is pure-visual.
- **No lexical match**: a query that matches no page text falls back to the visual ranking.
- **With a reranker**: the fused pool handed to the reranker is as wide as the pure-visual path,
  so the two-stage rerank still sees deep candidates.

### Tuning

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Enable | `COLPALI_HYBRID_ENABLED` | `false` | fuse lexical + visual |
| RRF constant | `COLPALI_HYBRID_KAPPA` | `60` | larger ⇒ flatter rank weighting |
| Candidate depth | `COLPALI_HYBRID_FETCH` | `100` | pulled from each channel before fusion |
| Min text coverage | `COLPALI_HYBRID_MIN_COVERAGE` | `0.5` | skip lexical below this fraction of text-bearing pages |
| N-gram range | `COLPALI_HYBRID_NGRAM_MIN` / `_MAX` | `3` / `5` | char-gram sizes; 3–5 is robust for codes |

The lexical index is built once per corpus and cached on the store.

### Measure before you trust it

Hybrid helps most on exact-identifier queries and can be neutral elsewhere. Don't ship it on a
hunch — A/B it with the eval harness ([EVAL.md](EVAL.md)):

```python
from colpali_rag.eval import compare_runs
from colpali_rag.engine import retrieve
cmp = compare_runs(
    cases,
    lambda q, k: retrieve(store, q, k, settings=visual_only),
    lambda q, k: retrieve(store, q, k, settings=hybrid_on),
    metric="coverage@1", n_boot=10000, seed=0)
# -> {'mean_a':.., 'mean_b':.., 'mean_delta':.., 'ci_low':.., 'ci_high':.., 'p_value':.., 'significant':..}
```

`coverage@k` (did *every* gold page make the top-k?) is the metric to watch for "find all of it";
`significant` gates on the paired-bootstrap CI excluding 0.

## Where to take it next

- **Two-stage prefetch → exact rerank** for scale: cheap candidate stage (row/col mean-pooled
  reps or IVF), exact MaxSim only on the top-N. Wire the existing `rerank_first_stage_n`.
- **Layout-aware lexical**: pdfium already exposes char boxes (geometry, not OCR) — usable to
  weight matches near captions/table cells.
- **Score calibration / cross-query normalization**: length-normalize MaxSim and fit a
  probability map so a single threshold transfers across queries and backends (needed for a
  trustworthy retrieval-side abstain). Note this needs a labeled set and refits per model/backend.
- **Fine-tuning the encoder** to the corpus is the highest-effort lever and belongs *last* — only
  once the above make its effect measurable. See [EXTENDING.md](EXTENDING.md).
