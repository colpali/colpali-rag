"""colpali-rag command line.

  colpali-rag index <pdf_dir>   build the index from a folder of PDFs
  colpali-rag query "<text>"    search from the terminal
  colpali-rag serve             launch the visual web UI
  colpali-rag info              show the current index / settings
  colpali-rag doctor            index health check (identity + embedding unit-norm)
  colpali-rag qdrant            run a local Qdrant server + web dashboard (no Docker/npm)
  colpali-rag migrate           push an existing index into Qdrant (no re-embedding)
  colpali-rag eval <file>       measure retrieval accuracy on a labeled set
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from colpali_rag.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Generic visual document RAG powered by ColPali (late-interaction vision retrieval).")


@app.callback()
def _main():
    """Configure logging (COLPALI_LOG_LEVEL, default INFO) before any command runs."""
    from colpali_rag.config import setup_logging

    setup_logging()


def _apply_overrides(settings, model, device, store, data_dir, collection, qdrant_url):
    if model: settings.model = model
    if device: settings.device = device
    if store: settings.store = store
    if data_dir: settings.data_dir = data_dir
    if collection: settings.collection = collection
    if qdrant_url: settings.qdrant_url = qdrant_url
    return settings


@app.command()
def index(
    pdf_dir: Path = typer.Argument(..., help="Folder of PDFs to index (searched recursively)"),
    model: Optional[str] = typer.Option(None, help="ColVision model id (default from env / colSmol-500M)"),
    device: Optional[str] = typer.Option(None, help="cpu | cuda | mps"),
    store: Optional[str] = typer.Option(None, help="memory | qdrant"),
    data_dir: Optional[str] = typer.Option(None, help="Where to persist the index"),
    collection: Optional[str] = typer.Option(None, help="Qdrant collection name"),
    qdrant_url: Optional[str] = typer.Option(None, help="Qdrant server URL (else embedded on-disk)"),
    fresh: bool = typer.Option(False, "--fresh", help="rebuild from scratch (else resume/add only new docs)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="embed only the first N pages (fast demo index off a big corpus)"),
):
    """Rasterize + embed every page under PDF_DIR and persist a searchable index.

    Incremental + resumable: re-running only embeds documents not already indexed, and an
    interrupted run picks up where it left off. Use --fresh to rebuild (e.g. after changing DPI),
    or --limit N to index just the first N pages for a quick demo.
    """
    from colpali_rag.engine import build_index

    s = _apply_overrides(get_settings(), model, device, store, data_dir, collection, qdrant_url)
    _store, _emb, info = build_index(pdf_dir, s, progress=lambda m: typer.echo(m), fresh=fresh, limit=limit)
    typer.secho(f"\n✓ indexed {info['pages']} page(s) from {info['docs']} doc(s) "
                f"({info.get('new_pages', 0)} newly embedded) → {info['store']} store at {info['data_dir']}",
                fg="green")
    typer.echo(f"  model={info['model']} device={info['device']} text_coverage={info['text_coverage']}")
    typer.echo("  next: colpali-rag serve   (then open http://127.0.0.1:8000)")


@app.command()
def migrate(
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    data_dir: Optional[str] = typer.Option(None), collection: Optional[str] = typer.Option(None),
):
    """Push an EXISTING on-disk index into Qdrant WITHOUT re-embedding.

    Use this if you already ran `index` with the default store and now want the Qdrant dashboard:
    it moves the vectors you already computed (the slow part is never repeated).
    """
    from colpali_rag.engine import migrate_index

    s = _apply_overrides(get_settings(), None, None, "qdrant", data_dir, collection, qdrant_url)
    migrate_index(s, progress=lambda m: typer.echo(m))
    typer.secho(f"\n✓ migrated. Serve it with:  set COLPALI_STORE=qdrant  (QDRANT_URL={s.qdrant_url})  "
                "then colpali-rag serve.\n  Dashboard: http://localhost:6333/dashboard", fg="green")


@app.command()
def qdrant(data_dir: Optional[str] = typer.Option(None, help="Where to keep the server + its storage")):
    """Download (once) and run a native Qdrant server + web dashboard — no Docker, no npm.

    Leaves the server running in the foreground; open http://localhost:6333/dashboard.
    In another terminal:  colpali-rag migrate   then   colpali-rag serve.
    """
    from colpali_rag.qdrant_server import run_server

    s = get_settings()
    if data_dir:
        s.data_dir = data_dir
    try:
        run_server(s, progress=lambda m: typer.echo(m))
    except KeyboardInterrupt:
        typer.echo("\nqdrant stopped.")


@app.command()
def query(
    text: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(8, help="How many pages to show"),
    rerank: bool = typer.Option(False, "--rerank", help="apply the configured reranker (needs [rerank] + RERANK_ENABLED)"),
    model: Optional[str] = typer.Option(None), device: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None), data_dir: Optional[str] = typer.Option(None),
    collection: Optional[str] = typer.Option(None), qdrant_url: Optional[str] = typer.Option(None),
):
    """Search an existing index from the terminal."""
    from colpali_rag.engine import open_index, retrieve
    from colpali_rag.rerank import get_reranker

    s = _apply_overrides(get_settings(), model, device, store, data_dir, collection, qdrant_url)
    store_obj, _emb = open_index(s)
    reranker = get_reranker(s) if rerank else None
    typer.echo(f"Top {k} pages for {text!r}:")
    for rec, score, pid in retrieve(store_obj, text, k, reranker=reranker, settings=s):
        typer.echo(f"  {score:9.4f}  {rec.doc}  p{rec.page}")


@app.command("eval")
def eval_cmd(
    eval_file: Path = typer.Argument(..., help='eval.jsonl lines: {"query": ..., "gold_page_ids": ["doc::p3", ...]}'),
    k: str = typer.Option("1,5,10", help="cutoffs, comma-separated"),
    rerank: bool = typer.Option(False, "--rerank", help="A/B: measure with the configured reranker on"),
    report: Optional[str] = typer.Option(None, help="write the full JSON report here"),
    model: Optional[str] = typer.Option(None), device: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None), data_dir: Optional[str] = typer.Option(None),
):
    """Measure retrieval accuracy (recall@k / nDCG@k / MRR) on a labeled query set."""
    import json as _json

    from colpali_rag.engine import open_index, retrieve
    from colpali_rag.eval import format_report, load_eval, run_eval
    from colpali_rag.rerank import get_reranker

    s = _apply_overrides(get_settings(), model, device, store, data_dir, None, None)
    store_obj, _emb = open_index(s)
    reranker = get_reranker(s) if rerank else None
    cases = load_eval(eval_file)
    ks = tuple(int(x) for x in k.split(","))
    rep = run_eval(cases, lambda query, tk: retrieve(store_obj, query, tk, reranker=reranker, settings=s), ks=ks)
    typer.echo(format_report(rep))
    if report:
        Path(report).write_text(_json.dumps(rep, indent=2))
        typer.secho(f"\nwrote {report}", fg="green")


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host (default 127.0.0.1)"),
    port: Optional[int] = typer.Option(None, help="Bind port (default 8000)"),
    model: Optional[str] = typer.Option(None), device: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None), data_dir: Optional[str] = typer.Option(None),
    collection: Optional[str] = typer.Option(None), qdrant_url: Optional[str] = typer.Option(None),
):
    """Launch the visual search UI (loads the index once at startup)."""
    import os

    import uvicorn

    s = _apply_overrides(get_settings(), model, device, store, data_dir, collection, qdrant_url)
    # pass overrides to the app process via env (app reads settings from env at startup)
    os.environ["COLPALI_MODEL"] = s.model
    os.environ["COLPALI_DEVICE"] = s.device
    os.environ["COLPALI_STORE"] = s.store
    os.environ["COLPALI_DATA_DIR"] = s.data_dir
    os.environ["COLPALI_COLLECTION"] = s.collection
    if s.qdrant_url:
        os.environ["QDRANT_URL"] = s.qdrant_url
    h, p = host or s.host, port or s.port
    typer.secho(f"colpali-rag serving on http://{h}:{p}  (model={s.model}, store={s.store})", fg="cyan")
    uvicorn.run("colpali_rag.app:app", host=h, port=p)


@app.command()
def studio(
    host: Optional[str] = typer.Option(None, help="Bind host (default 127.0.0.1)"),
    port: Optional[int] = typer.Option(None, help="Bind port (default 8000)"),
    model: Optional[str] = typer.Option(None), device: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None), data_dir: Optional[str] = typer.Option(None),
    collection: Optional[str] = typer.Option(None), qdrant_url: Optional[str] = typer.Option(None),
):
    """Launch Studio: chat + source selection + CSV/Excel upload → interactive, cited
    structured outputs over your documents. Demo mode if no index/LLM is configured."""
    import os

    import uvicorn

    s = _apply_overrides(get_settings(), model, device, store, data_dir, collection, qdrant_url)
    os.environ["COLPALI_MODEL"] = s.model
    os.environ["COLPALI_DEVICE"] = s.device
    os.environ["COLPALI_STORE"] = s.store
    os.environ["COLPALI_DATA_DIR"] = s.data_dir
    os.environ["COLPALI_COLLECTION"] = s.collection
    if s.qdrant_url:
        os.environ["QDRANT_URL"] = s.qdrant_url
    h, p = host or s.host, port or s.port
    mode = "llm" if s.vlm_enabled else "demo"
    typer.secho(f"colpali-rag studio on http://{h}:{p}  (mode={mode}, store={s.store})", fg="magenta")
    typer.echo("  dev UI: cd web && npm install && npm run dev   → http://localhost:5173")
    uvicorn.run("colpali_rag.studio.server:app", host=h, port=p)


@app.command()
def info(
    data_dir: Optional[str] = typer.Option(None), store: Optional[str] = typer.Option(None),
):
    """Show the resolved settings and whether an index exists."""
    s = _apply_overrides(get_settings(), None, None, store, data_dir, None, None)
    rec = Path(s.data_dir) / "records.json"
    typer.echo(f"model      : {s.model}")
    typer.echo(f"device     : {s.device}")
    typer.echo(f"store      : {s.store}")
    typer.echo(f"data_dir   : {Path(s.data_dir).resolve()}")
    if s.store == "qdrant":
        typer.echo(f"qdrant_url : {s.qdrant_url or '(embedded on-disk)'}")
        typer.echo(f"collection : {s.collection}")
    typer.echo(f"index      : {'present' if rec.exists() else 'MISSING — run colpali-rag index <dir>'}")


@app.command("doctor")
def doctor(
    data_dir: Optional[str] = typer.Option(None), store: Optional[str] = typer.Option(None),
    sample: int = typer.Option(4000, help="patch vectors to sample for the unit-norm check"),
):
    """Health check: index identity (model/adapter/schema) + embedding unit-norm.

    dot-product (in-memory) and cosine (Qdrant) scoring agree only when the model's page
    vectors are unit-norm; this flags a checkpoint/fine-tune whose head dropped that.
    """
    import json as _json

    s = _apply_overrides(get_settings(), None, None, store, data_dir, None, None)
    rec = Path(s.data_dir) / "records.json"
    if not rec.exists():
        typer.secho("no index — run: colpali-rag index <pdf_dir>", fg="red")
        raise typer.Exit(1)
    meta = _json.loads(rec.read_text())
    typer.echo(f"model      : {meta.get('model')}")
    typer.echo(f"adapter    : {meta.get('adapter') or '(none)'}")
    typer.echo(f"backend    : {meta.get('backend')}")
    typer.echo(f"schema     : v{meta.get('schema_version')}")
    typer.echo(f"pages      : {len(meta.get('ids', []))}")

    emb_path = Path(s.data_dir) / "embeddings.pt"
    if meta.get("backend") == "memory" and emb_path.exists():
        import torch

        from colpali_rag.diagnostics import check_unit_norm

        embs = torch.load(emb_path, weights_only=False)
        ok, stats = check_unit_norm(embs, tol=s.norm_tol, sample=sample)
        metric = "cosine" if s.store == "qdrant" else "dot-product"
        typer.echo(f"norm dev   : mean {stats['mean_dev']:.5f}  max {stats['max_dev']:.5f}  "
                   f"(n={stats['n']}, tol={s.norm_tol})")
        if ok:
            typer.secho(f"unit-norm  : OK — {metric} scoring is safe; in-memory and Qdrant agree",
                        fg="green")
        else:
            typer.secho("unit-norm  : WARN — page embeddings aren't unit-norm; the in-memory (dot) "
                        "and Qdrant (cosine) backends can rank differently. Pick one backend, or "
                        "re-embed with a normalizing checkpoint.", fg="yellow")
    else:
        typer.echo("norm dev   : (embeddings not stored locally for this backend — run doctor "
                   "against a memory index, or use diagnostics.probe_backend_agreement)")


if __name__ == "__main__":
    app()
