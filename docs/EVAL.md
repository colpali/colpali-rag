# Measuring accuracy (so changes ship on evidence, not vibes)

Every retrieval or generation change should be *measured*. A model, a prompt, a resolution bump, a
reranker, hybrid fusion — none of them are improvements until a number says so, on a labeled set,
with a significance test. `colpali_rag.eval` and `colpali_rag.graph_eval` provide that, pure-Python
and dependency-free.

## Retrieval metrics

Label a set of queries with their relevant page ids — `eval.jsonl`, one object per line:

```json
{"query": "thermal derating curve", "gold_page_ids": ["report.pdf::p3", "report.pdf::p4"]}
```

Optionally give graded relevance instead of a flat gold set: `{"query": "...", "gold_gains":
{"report.pdf::p3": 3, "report.pdf::p4": 1}}` (gain 0 marks a judged-nonrelevant page).

```bash
colpali-rag eval eval.jsonl --k 1,5,10 [--rerank] [--report report.json]
```

Per cutoff `k` and overall, `run_eval` reports:

| Metric | Question it answers |
|---|---|
| **recall@k** | what fraction of relevant pages made the top-k |
| **coverage@k** | did **every** relevant page make the top-k (all-or-nothing) — the "find all of it" signal |
| **recall@100** | recall at a deep cutoff (the honest ceiling when a query has many relevant pages) |
| **nDCG@k** | rank quality (graded when `gold_gains` is given, binary otherwise) |
| **MAP** | mean average precision across queries |
| **MRR** | reciprocal rank of the first relevant page |

> Note: `recall@k` divides by the number of relevant pages, so for a multi-page query it can't
> reach 1.0 at a small `k`. Read `coverage@k` and `recall@100` alongside it — those are the
> unbiased "did we get everything" signals.

The numbers are only as good as your labels. A real retrieval eval needs a real index + a real
labeled set (an offline job, not the fast test suite).

## A/B with significance (paired bootstrap)

A +0.03 on 30 queries is usually noise. `compare_runs` runs two retrievers on the **same** queries
and paired-bootstraps the per-query delta — pairing removes query-difficulty variance:

```python
from colpali_rag.eval import compare_runs
cmp = compare_runs(cases, retrieve_a, retrieve_b, metric="coverage@5", n_boot=10000, seed=0)
# {'metric':'coverage@5', 'mean_a':.., 'mean_b':.., 'mean_delta':..,
#  'ci_low':.., 'ci_high':.., 'p_value':.., 'significant': True/False}
```

`significant` is the ship/no-ship flag: the 95% CI of the delta excludes 0 (and there are ≥ 2
queries). It's deterministic given `seed`, so results are reproducible. Ship a change when the
delta is significant on the metric you care about — not before.

## Structured-output metrics

For the closed-vocabulary constraint (see [CONSTRAINTS.md](CONSTRAINTS.md)), `graph_eval` scores
the model's raw output against the vocabulary:

- **P_adh** — fraction of raw nodes in the vocabulary; **HPR** = 1 − P_adh (drive to 0);
- **C_req** — required-item coverage in the emitted output (report jointly so P_adh isn't gamed
  by dropping).

```python
from colpali_rag.graph_eval import graph_report, format_graph_report
print(format_graph_report(graph_report(raw_labels, emitted_ids,
                                       accept=catalog.accept, required=catalog.required)))
```

## Where to take it next

- **Score calibration (ECE / Brier)** once a probability map is fit — for a trustworthy
  retrieval-side abstain and for comparing base vs fine-tuned models across a score-scale shift.
- **Judge-vs-human agreement (Cohen's κ)** for the faithfulness judge, as its trust floor.
- **Graded gold + node/edge-F1** for structured outputs via bipartite matching against a gold
  graph (needs labeled examples).
- **A held-out generic eval set** checked into your own harness so every change has a baseline to
  beat — keep any domain-specific labels external.
