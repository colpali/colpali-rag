"""colpali-rag command line.

  colpali-rag index <pdf_dir>   build the index from a folder of PDFs
  colpali-rag query "<text>"    search from the terminal
  colpali-rag serve             launch the visual web UI
  colpali-rag info              show the current index / settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from colpali_rag.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Generic visual document RAG powered by ColPali (late-interaction vision retrieval).")


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
):
    """Rasterize + embed every page under PDF_DIR and persist a searchable index."""
    from colpali_rag.engine import build_index

    s = _apply_overrides(get_settings(), model, device, store, data_dir, collection, qdrant_url)
    _store, _emb, info = build_index(pdf_dir, s, progress=lambda m: typer.echo(m))
    typer.secho(f"\n✓ indexed {info['pages']} page(s) from {info['docs']} doc(s) "
                f"→ {info['store']} store at {info['data_dir']}", fg="green")
    typer.echo(f"  model={info['model']} device={info['device']} text_coverage={info['text_coverage']}")
    typer.echo("  next: colpali-rag serve   (then open http://127.0.0.1:8000)")


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
    for rec, score, pid in retrieve(store_obj, text, k, reranker=reranker):
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
    rep = run_eval(cases, lambda query, tk: retrieve(store_obj, query, tk, reranker=reranker), ks=ks)
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


if __name__ == "__main__":
    app()
