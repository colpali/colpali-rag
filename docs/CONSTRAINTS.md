# Closed-vocabulary constraints for structured outputs

The studio turns a request + retrieved pages + an uploaded table into a **structured, cited
output** — typed `nodes` and labeled `connections` (see [STUDIO.md](STUDIO.md)). By default a
model is free to name a node anything. When the output is something people build on, an invented
entity is a liability.

This layer makes the output **provably drawn from a controlled vocabulary you supply**: a table
that is your source of truth for the entities allowed in the output (a product/SKU list, a
component registry, an account chart, an asset inventory — anything with an id column). It is
**generic and off by default**; nothing here hard-codes a schema. You point it at your table's
column names at runtime and it does the rest.

The guarantee is not "the model tried to stay in the vocabulary." It is **generate-then-verify-
and-repair**: whatever the model returns is projected onto the vocabulary server-side, so every
emitted node is in it by construction — and every change is recorded, never silent.

---

## Turn it on

Set the header names of your table's columns (only the id column is required). Any uploaded table
carrying the id column is compiled into the vocabulary.

```bash
export CATALOG_ID_COL="id"                # the column whose values are the allowed entity ids
export COLPALI_CATALOG_GATE=withhold      # off | flag | withhold   (default off)
# optional:
export CATALOG_NAME_COLS="name,alias"     # extra columns to match against (resolve aliases -> id)
export CATALOG_REQUIRED_COL="required"    # a truthy column marks ids that MUST appear
export CATALOG_IFACE_COLS="interface"     # interface tokens -> which entities may connect
export CATALOG_REPAIR_MAX=1               # times to re-prompt the model to fix violations
```

Leave `CATALOG_ID_COL` blank (the default) and everything below is inert — the studio behaves
exactly as it did before. If the id column is set but no uploaded table carries it, the constraint
logs that it is **inactive** rather than silently doing nothing.

The **whole** table is compiled — there is no row cap on the constraint channel. (The separate
`TABULAR_MAX_*` knobs only bound how much of a table is rendered into the model's prompt; they can
never shrink the vocabulary.)

---

## What it enforces

Given a proposed output, projection (Π) runs four checks and records the result on the spec:

1. **Node membership.** Each node label is matched to the vocabulary:
   - **exact** id (case/punctuation-normalized) → kept as the canonical id;
   - a confident **fuzzy** match (id-aware char similarity + token overlap, above a threshold and
     margin) → **remapped** to the canonical id, recorded in `remapped_parts` and an assumption;
   - otherwise → **dropped** and recorded in `hallucinated_parts`. Edges to a dropped node go too.
   Every surviving node is therefore in the vocabulary. Ids are preserved, not slugged, so
   `AX-12.34` and `AX 12 34` stay distinct.
2. **Connection feasibility** (if `CATALOG_IFACE_COLS` is set). Two entities may connect only if
   they share an interface token (a common bus / connector / type). Infeasible edges are
   **flagged** (`infeasible_connections`), never silently deleted — because incomplete interface
   data must not remove real edges. When either endpoint has no interface data, the edge isn't
   judged.
3. **Required completeness** (if `CATALOG_REQUIRED_COL` is set). Required ids absent from the
   output are reported in `missing_required` — so you cannot "win" by dropping everything.
4. **Repair, then gate.** When a model is configured and violations remain, the model is
   re-prompted with the *specific* violations ("these labels aren't in the vocabulary", "these
   connections aren't permitted", "these required items are missing") up to `CATALOG_REPAIR_MAX`
   times. Then the terminal gate decides:
   - `off` — no constraint (default);
   - `flag` — emit the repaired output with all violations annotated (visible in the UI);
   - `withhold` — **abstain** (emit nothing) when the result is still invalid: nothing grounded,
     too much dropped, a required item missing, or an infeasible edge remaining.

The demo generator (no model configured) is projected through the same Π, so it's constrained too.

New fields on the output object (all additive, safe to ignore): `hallucinated_parts`,
`remapped_parts`, `dropped_blocks`, `infeasible_connections`, `missing_required`,
`repair_attempts`, `withheld`.

---

## Measuring it

`colpali_rag.graph_eval` scores the *model's raw output* against the vocabulary (dependency-pure;
wire in a compiled `Catalog.accept` and `required` set):

- **P_adh** (adherence) — fraction of raw nodes that resolve to the vocabulary.
- **HPR** = 1 − P_adh — the hallucinated-part rate to drive to zero.
- **C_req** — fraction of the required set present in the *emitted* output, reported alongside
  P_adh so adherence can't be gamed by dropping.

```python
from colpali_rag.graph_eval import graph_report, format_graph_report
r = graph_report(raw_labels, emitted_ids, accept=catalog.accept, required=catalog.required)
print(format_graph_report(r))   # nodes=3  P_adh=0.6667  HPR=0.3333  C_req=1.0000  dropped=1
```

Post-projection every emitted node is in the vocabulary by construction, so the honest signal is
**HPR on the raw output falling toward 0** while **C_req stays high**.

---

## Tuning

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Gate | `COLPALI_CATALOG_GATE` | `off` | `off` \| `flag` \| `withhold` |
| Id column | `CATALOG_ID_COL` | *(off)* | header of the id column; empty ⇒ disabled |
| Alias columns | `CATALOG_NAME_COLS` | — | comma-separated; match-only, resolve to the id |
| Required column | `CATALOG_REQUIRED_COL` | — | truthy cell ⇒ id must appear |
| Interface columns | `CATALOG_IFACE_COLS` | — | shared token ⇒ connection permitted |
| Match threshold | `CATALOG_MATCH_THRESHOLD` | `0.84` | accept a fuzzy match at/above this |
| Match margin | `CATALOG_MATCH_MARGIN` | `0.08` | …only if it beats the runner-up by this |
| Repair passes | `CATALOG_REPAIR_MAX` | `1` | re-prompts before the terminal gate |
| Withhold drop cap | `CATALOG_WITHHOLD_MAX_DROP` | `0.5` | abstain if this fraction of nodes drop |

Raise `CATALOG_MATCH_THRESHOLD` for stricter matching (more drops, fewer wrong remaps); lower it
if legitimate near-misses are being dropped. Watch `remapped_parts` after a run to calibrate.

---

## Where to take it next

- **Enum-constrained decoding** for the strict `json_schema` tier: inject a shortlist of allowed
  ids (top-N by request similarity ∪ required) as a JSON-schema enum, so the model biases toward
  the vocabulary *before* projection. A complement to Π, never the sole guarantee.
- **Richer graph rules**: port/cardinality limits, mandatory sub-structures, directional
  interfaces (separate in/out columns) — layered onto the same verify-and-repair loop.
- **Embedding match channel**: an optional semantic similarity term in the matcher for
  description-style labels (kept off by default; needs a vocabulary embedding matrix).
- **Active learning**: surface low-margin remaps and near-threshold drops for review; feed the
  decisions back to tune the threshold per corpus.
