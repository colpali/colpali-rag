# Scaling

colpali-rag has two stores behind one interface (`store.py`). Start with `memory`; move
to `qdrant` when the corpus grows.

## Choosing a store

| Store | How | When |
|---|---|---|
| `memory` | brute-force MaxSim in Python; persists to `<data_dir>` | zero infra; correct to a few thousand pages |
| `qdrant` (embedded) | `QdrantClient(path=…)` on disk | persistence, no server |
| `qdrant` (server) | `QDRANT_URL` | scale-out to millions of pages |

```bash
docker compose up -d qdrant                       # optional server on :6333
export COLPALI_STORE=qdrant QDRANT_URL=http://localhost:6333
colpali-rag index ./pdfs
```

Without `QDRANT_URL`, `COLPALI_STORE=qdrant` uses an **embedded on-disk** Qdrant under
`COLPALI_DATA_DIR/qdrant` — persistent, still no server.

## Why a multivector store is required

ColPali emits **>1,000 vectors per page** (ColQwen ~700, image-dependent). An ordinary
single-vector database can't store or score that. You need native multivector support:
Qdrant (used here) with the `MAX_SIM` comparator, or Milvus/Vespa.

## The scaling recipe (enable when the corpus is large)

Brute-force exact MaxSim is correct but heavy at scale. The standard evolution for
`QdrantStore`:

1. **HNSW off on the heavy originals.** MaxSim doesn't work with proximity graphs:
   `hnsw_config=HnswConfigDiff(m=0)` on the full multivector.
2. **Mean-pooled prefetch vectors.** Pool the patch grid by rows and columns into
   ~32–64 vectors per page and HNSW-index *those*. This is the order-of-magnitude
   speed win. (`colpali_engine`'s `HierarchicalTokenPooler` is an alternative: ~3× fewer
   vectors for a small quality loss.)
3. **Two-stage query.** `query_points(prefetch=[pooled_rows, pooled_cols], using="original")`:
   cheap pooled prefetch (limit ~100) → exact MaxSim rerank on the survivors (limit ~10).
4. **Quantization.** Binary (~32×) or scalar (~4×) quantization reduces **memory**, not
   indexing time. Apply after pooling if RAM-bound; validate recall on your own corpus.
5. **MUVERA (fixed-dimensional encodings).** Collapse a multivector to a single vector so
   an off-the-shelf ANN index can serve it — ~8× faster, for billion-scale. Overkill for
   most corpora; the escape hatch if it explodes.

## GPU throughput (indexing)

Retrieval quality and heatmap sharpness improve markedly on GPU with
`vidore/colqwen2-v1.0`. A few hundred pages index in well under a minute on any modern
GPU; CPU `colSmol-500M` handles small corpora for local use.
