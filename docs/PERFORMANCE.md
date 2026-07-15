# Indexing performance — making embedding fast on a big corpus

Indexing embeds every page through a vision transformer. That forward pass is the whole cost, and
it is **~100× faster on a GPU than on a CPU**. If a run is crawling (e.g. a fraction of a page per
second), you are almost certainly embedding on a CPU with no GPU — that is the bottleneck, not the
code.

Two facts shape the right approach:

1. **Indexing is a one-time batch job; querying is cheap.** A query embeds *one* multivector
   (~100 ms even on CPU) and scores it against the stored pages. So you only need a GPU to
   **index** — you can happily **serve on a CPU** afterward.
2. **The index is portable.** For the in-memory store it is just files
   (`colpali_data/embeddings.pt`, `records.json`, `images/`); for Qdrant it is points in the DB.
   So: index on a GPU, copy the index to wherever you serve.

## The fix: index on a GPU (then serve anywhere)

Rent a GPU box (or use one you have), index there, ship the index back.

```bash
# on a GPU machine (a single T4/A10/A100 is plenty)
pip install -e '.[rag,api]'
export COLPALI_DEVICE=cuda            # <- the whole point
export COLPALI_MODEL=vidore/colqwen2-v1.0   # a sharper GPU model (or keep colSmol)
colpali-rag index ./corpus            # thousands of pages in minutes, not hours

# then serve on the GPU box, OR copy the index to a CPU box and serve there:
rsync -a gpu-box:/path/colpali_data ./          # in-memory store = just files
colpali-rag serve                                # querying is fast on CPU
```

For scale or to avoid copying files, index straight into **Qdrant** (`COLPALI_STORE=qdrant
QDRANT_URL=…`) on the GPU box; then any CPU box pointed at the same Qdrant serves immediately.

Rough order of magnitude: a corpus that takes ~10+ hours on a laptop CPU indexes in **tens of
minutes** on one modern GPU. The GPU time is cheap and one-time.

## If you must stay on CPU — squeeze levers (measure each)

None of these approach a GPU, but stacked they can be a few× faster. **Validate quality** after
any resolution change with the eval harness ([EVAL.md](EVAL.md)) — lower resolution trades accuracy
for speed.

| Lever | How | Effect |
|---|---|---|
| **Smallest model** | `COLPALI_MODEL=vidore/colSmol-500M` (the CPU default) | biggest single factor; don't run a 2B+ model on CPU |
| **Lower resolution** | `COLPALI_MAX_DIM=1024` (from 1600), `COLPALI_DPI=110` | fewer patches/tiles → ~2–4× faster (the default model tiles, so this compounds) |
| **Batch pages** | `COLPALI_BATCH_SIZE=4` (or 8) | better core utilization; ~1.5–3× if you have the RAM |
| **Use all cores** | `OMP_NUM_THREADS=<physical cores>` | ensure Torch isn't under-threaded |
| **Intel acceleration** | export the model to **OpenVINO** / use **IPEX** (optimum-intel) | on an Intel Core Ultra (iGPU + NPU) this is the largest CPU-side win, but it requires exporting the model and is an advanced setup |

## Resumable + incremental indexing (built in)

`colpali-rag index` is **resumable and incremental**, so a long run is survivable:

- It **checkpoints every `COLPALI_INDEX_CHECKPOINT_PAGES` pages** (default 250) — an interrupted
  run resumes from the last checkpoint instead of restarting.
- Re-running only embeds **documents that aren't already indexed** — add new PDFs and just run
  `index` again; it skips the ones already done.
- It prints a live **pages/sec and ETA** so you can see where you stand.
- Use **`--fresh`** to rebuild from scratch (do this after changing DPI / `max_dim`, since those
  aren't tracked by the index-identity guard).

```bash
colpali-rag index ./corpus            # start (or resume) — prints "· doc: N page(s) … ETA ~M min"
# Ctrl-C, come back later:
colpali-rag index ./corpus            # resumes; "resuming — K document(s) already indexed"
colpali-rag index ./corpus --fresh    # force a full rebuild
```

## The recommended workflow

1. **Index on a GPU** (cloud or otherwise) — one-time, minutes. Use Qdrant if you want the index
   to live in a DB rather than files.
2. **Serve on whatever you like**, including a CPU laptop — querying is cheap.
3. Keep re-indexing **incremental** (just run `index` when documents change).
4. Only tune CPU levers if a GPU truly isn't an option — and measure the accuracy cost.
