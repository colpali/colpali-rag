# Get a demo running in ~10 minutes (Windows, no Docker, no GPU)

Three paths, fastest first. You do **not** need Docker, a GPU, or Qdrant to show something real.

## 0. Install — the #1 thing that trips people up

Install the **right extras**. Missing them is why uploads or the test suite fail:

```powershell
python -m venv .venv
.venv\Scripts\activate                     # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[rag,api,studio,dev]"
cd web; npm install; npm run build; cd ..
```

- `studio` pulls in **python-multipart** (file uploads) + **openpyxl** (Excel). Without it the upload endpoint errors.
- `dev` pulls in **reportlab**, which the test suite needs to generate its sample PDFs.

Sanity check: `python -m pytest -q` → all green. If tests fail about a missing sample PDF or
`python-multipart`, you skipped an extra above.

## Path A — instant demo: no model, no index, no GPU (~30 seconds)

```powershell
colpali-rag studio            # open http://127.0.0.1:8000
```

With no index built, the studio runs in **demo mode**: type a request and it returns a
node/connection diagram inferred from your text, rendered on the canvas. It shows the whole UI
immediately and doesn't even download the model. Great for "here's the product" in a meeting.

## Path B — grounded on a handful of real pages (in-memory, no Docker) (~2–3 min)

```powershell
mkdir pdfs                                  # drop 2-3 real PDFs in here
colpali-rag index .\pdfs --limit 16         # embeds only 16 pages; prints pages/sec + ETA, checkpoints
colpali-rag serve                           # http://127.0.0.1:8000 — search + heatmaps over real pages
```

The first `index` downloads the model once (~1 GB). 16 pages on CPU ≈ a couple minutes. `--limit`
is exactly for this: a fast demo index off a big corpus without hand-copying files. Re-run without
`--limit` later to index everything (it resumes and only embeds what's new).

## Path C — real model-read answers (LM Studio, local)

Start LM Studio, load a **vision** model (e.g. Qwen2.5-VL), **Start Server**, then:

```powershell
set VLM_BASE_URL=http://localhost:1234/v1
set VLM_MODEL=<model-id-as-lm-studio-shows-it>
set VLM_API_KEY=lm-studio
colpali-rag studio
```

Now the studio actually reads the retrieved page images and returns cited structured output.

## Optional — Qdrant without Docker (native Windows binary)

You don't need Qdrant for a demo (Path B's in-memory store is simpler). But if you want the Qdrant
dashboard and can't use Docker:

1. Download the **Qdrant Windows x64 binary** from the official GitHub releases; **verify the
   published SHA-256** before running it.
2. Run it — it listens on `127.0.0.1:6333` (REST) and `:6334` (gRPC). Point its storage at a
   gitignored folder via `QDRANT__STORAGE__STORAGE_PATH`.
3. For the dashboard UI, download the **qdrant-web-ui** release and set
   `QDRANT__SERVICE__STATIC_CONTENT_DIR` to its `dist/` folder, then open
   `http://localhost:6333/dashboard`.
4. Point the app at it and index:
   ```powershell
   set COLPALI_STORE=qdrant
   set QDRANT_URL=http://localhost:6333
   colpali-rag index .\pdfs --limit 16
   ```

## See exactly what happened (great for a demo + debugging)

```powershell
set COLPALI_LOG_LEVEL=INFO
set COLPALI_RUN_LOG_DIR=.\runs
```

Now every generation prints a step trace and writes a `.json` + `.txt` summary to `.\runs\` — what
it studied, what it produced, and every constraint/repair check.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Upload / `python-multipart` error | you missed the studio extra: `pip install -e ".[studio]"` |
| Tests fail on a missing sample PDF | install dev: `pip install -e ".[dev]"` (reportlab) |
| Indexing crawls (fraction of a page/sec) | that's CPU with no GPU — use `--limit` for demos; index the full corpus on a GPU ([PERFORMANCE.md](PERFORMANCE.md)) |
| Want zero setup on screen | Path A (demo mode) — no model, no index |
