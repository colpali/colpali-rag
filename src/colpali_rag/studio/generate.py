"""Turn a request into a cited DiagramSpec.

Flow:
  1. Retrieve the most relevant datasheet pages for the request (ColPali late
     interaction), restricted to the datasheets the user selected.
  2. Attach any uploaded tables / notes as additional citable sources.
  3. Ask the configured LLM (any OpenAI-compatible vision endpoint) to emit a
     DiagramSpec under the enforced JSON schema — json_schema → json_object → prompt
     cascade, defensive parse, one corrective retry — so a block or connection can
     cite the exact page or row it came from.
  4. If no LLM is configured (or the call fails), fall back to a deterministic DEMO
     generator that builds a plausible block diagram from the request text + selected
     sources. This is what makes the whole studio runnable with zero infrastructure.

Sources are numbered [1..n] in the order shown; the model cites those indices and we
resolve them back to real page/table ids in spec.validate_diagram_obj.
"""

from __future__ import annotations

import logging
import re

from colpali_rag.studio.spec import (
    DIAGRAM_JSON_SCHEMA,
    Block,
    Connection,
    DiagramSpec,
    parse_diagram,
    validate_diagram_obj,
)

log = logging.getLogger(__name__)

_TIERS = ["json_schema", "json_object", "prompt"]


# --------------------------------------------------------------------------- sources
def collect_sources(request, *, store, selected_docs=None, tables=None, notes=None,
                    top_k=6, reranker=None, lock=None, settings=None):
    """Build the ordered source list (+ the page images to send). Returns
    (sources, page_images) where sources[i] has a stable 1-based cite index i+1."""
    from colpali_rag.engine import retrieve   # lazy: keeps torch out of the demo/import path

    caps = {}
    if settings is not None:
        caps = {"max_rows": getattr(settings, "tabular_max_preview_rows", 40),
                "max_cols": getattr(settings, "tabular_max_cols", 24),
                "max_cell": getattr(settings, "tabular_max_cell", 80)}

    sources: list[dict] = []
    page_images: list = []
    selected = set(selected_docs or [])

    if store is not None and len(store) > 0 and request.strip():
        fetch = top_k * 4 if selected else top_k
        _acquire = lock.acquire if lock else (lambda: None)
        _release = lock.release if lock else (lambda: None)
        _acquire()
        try:
            hits = retrieve(store, request, top_k=fetch, reranker=reranker, settings=settings)
        finally:
            _release()
        for r, sc, pid in hits:
            if selected and r.doc not in selected:
                continue
            im = store.get_image(pid)
            if im is None:
                continue
            sources.append({"id": f"p{len(sources)+1}", "kind": "page", "ref": pid,
                            "doc": r.doc, "page": r.page, "score": round(float(sc), 3),
                            "label": f"{r.doc} · p.{r.page}"})
            page_images.append(im)
            if len(page_images) >= top_k:
                break

    for t in tables or []:
        sources.append({"id": f"t{len(sources)+1}", "kind": "table", "ref": t.name,
                        "label": t.name, "text": t.summary(**caps)})
    for nt in notes or []:
        sources.append({"id": f"n{len(sources)+1}", "kind": "note", "ref": nt.name,
                        "label": nt.name, "text": nt.summary()})
    return sources, page_images


# --------------------------------------------------------------------------- LLM path
_INSTR = (
    "You are a systems architect. From the attached datasheet page image(s) and the "
    "table/note text below, design a BLOCK DIAGRAM that satisfies the request. Return "
    "ONLY a JSON object with this shape:\n"
    '{"title": str, "reasoning": str, "assumptions": [str], '
    '"groups": [{"id": str, "label": str}], '
    '"blocks": [{"id": str, "label": str, "kind": one of '
    '["component","system","process","io","store","external","actor"], "group": str, '
    '"cites": [int]}], '
    '"connections": [{"from": block_id, "to": block_id, "label": str, "kind": one of '
    '["data","control","signal","power","bus","dependency"], "cites": [int]}]}\n'
    "Rules: block ids are short slugs; every connection's from/to must match a block id; "
    "cite the sources you used by their bracket number [n] shown above each page and "
    "before each table/note; keep it to the blocks the request actually needs."
)


def _sources_text(sources) -> str:
    parts = []
    for i, s in enumerate(sources, start=1):
        if s.get("text"):
            parts.append(f"[{i}] {s['text']}")
    return "\n\n".join(parts)


def _response_format(tier):
    if tier == "json_schema":
        return {"type": "json_schema",
                "json_schema": {"name": "diagram_spec", "schema": DIAGRAM_JSON_SCHEMA, "strict": True}}
    if tier == "json_object":
        return {"type": "json_object"}
    return None


def _repair_note(viol) -> str:
    """A corrective instruction naming the exact catalog violations to fix on the next attempt."""
    parts = []
    if viol["hallucinated"]:
        parts.append("These labels are not in the catalog and were removed — replace them with "
                     "catalog items or omit them: " + ", ".join(viol["hallucinated"]) + ".")
    if viol["infeasible_edges"]:
        parts.append("These connections are not permitted (the items cannot connect): "
                     + "; ".join(f"{a} -> {b}" for a, b in viol["infeasible_edges"]) + ".")
    if viol["missing"]:
        parts.append("These required items are missing and MUST appear: "
                     + ", ".join(viol["missing"]) + ".")
    return ("Your previous diagram violated the catalog. " + " ".join(parts)
            + " Return corrected JSON using ONLY items that appear in the attached table(s).")


def _llm_diagram(request, page_images, sources, settings, *, mode="auto", max_retries=1,
                 extra_note=""):
    from colpali_rag.generator import _image_data_uri, _post_chat
    import httpx

    head = _INSTR + f"\n\nRequest: {request}"
    if extra_note:
        head += "\n\n" + extra_note
    content = [{"type": "text", "text": head}]
    for i, im in enumerate(page_images, start=1):
        content.append({"type": "text", "text": f"[{i}] page image:"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_uri(im)}})
    stext = _sources_text(sources)
    if stext:
        content.append({"type": "text", "text": "Tables / notes:\n" + stext})
    messages = [{"role": "user", "content": content}]

    tiers = _TIERS if mode == "auto" else [mode]
    raw, errs = "", []
    for tier in tiers:
        for attempt in range(max_retries + 1):
            msgs = messages if attempt == 0 else messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not valid. Return ONLY JSON matching the shape. "
                 + "; ".join(errs[-2:])}]
            try:
                data = _post_chat(settings.vlm_base_url, settings.vlm_api_key, settings.vlm_model,
                                  msgs, response_format=_response_format(tier),
                                  max_tokens=1600, timeout=180.0)
            except httpx.HTTPStatusError as e:
                sc = e.response.status_code if e.response is not None else 0
                if sc in (400, 422):
                    errs.append(f"{tier}: HTTP {sc}")
                    break
                raise
            raw = (data["choices"][0]["message"].get("content") or "").strip()
            obj = parse_diagram(raw)
            if obj is not None:
                try:
                    return validate_diagram_obj(obj, sources, mode=tier)
                except Exception as ve:  # noqa: BLE001 - shape error -> retry / next tier
                    errs.append(f"{tier}: {ve}")
    spec = _demo_diagram(request, sources)     # never fail to produce a diagram
    spec.mode = "demo-fallback"
    spec.errors = errs
    return spec


# --------------------------------------------------------------------------- demo path
_STOP = {"a", "an", "the", "of", "to", "and", "that", "which", "for", "with", "from",
         "into", "diagram", "show", "make", "create", "draw", "block", "please", "i",
         "want", "need", "system", "this", "shows", "does", "it", "as", "on", "in"}
_CONNECT = re.compile(
    r"\s*(?:->|→|=>|\bthen\b|\bfeeds?\b|\bdrives?\b|\bsends?\b|\breads?\b|"
    r"\binto\b|\bto\b|\bconnects? to\b|\bflows? to\b|\boutputs? to\b)\s*", re.I)
_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\bplus\b|\bwith\b)\s*", re.I)
_KIND_HINTS = [
    (("sensor", "input", "adc", "camera", "microphone", "gpio", "signal in"), "io"),
    (("power", "supply", "psu", "battery", "vcc", "regulator", "mains"), "external"),
    (("database", "db", "store", "storage", "memory", "cache", "log"), "store"),
    (("controller", "mcu", "cpu", "processor", "fpga", "soc", "engine", "core"), "system"),
    (("service", "server", "api", "gateway", "broker", "cloud", "network"), "external"),
    (("filter", "encode", "decode", "transform", "process", "convert", "amp"), "process"),
    (("user", "operator", "client", "human", "actor"), "actor"),
]


def _label(phrase: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9+]+", phrase) if w.lower() not in _STOP]
    words = words[:4]
    if not words:
        return ""
    return " ".join(w if w.isupper() else w.capitalize() for w in words)


def _kind_for(label: str) -> str:
    low = label.lower()
    for keys, kind in _KIND_HINTS:
        if any(k in low for k in keys):
            return kind
    return "component"


def _phrases_from_request(request: str):
    """Best-effort parse of the request into an ordered chain of block phrases. Chains
    on connective words ('A feeds B'), else splits on commas/'and' (parallel blocks)."""
    req = request.strip().rstrip(".")
    chain = [p for p in _CONNECT.split(req) if p.strip()]
    if len(chain) >= 2:
        return [_label(p) for p in chain if _label(p)], True
    parts = [p for p in _SPLIT.split(req) if p.strip()]
    return [_label(p) for p in parts if _label(p)], False


def _demo_diagram(request: str, sources: list) -> DiagramSpec:
    """Deterministic, no-LLM diagram: parse the request into blocks, chain or fan them,
    and attach whatever sources were selected. Looks intentional; runs with zero infra."""
    labels, chained = _phrases_from_request(request)
    labels = [l for l in dict.fromkeys(labels) if l][:8]        # dedupe, cap
    src_cites = list(range(1, min(len(sources), 6) + 1))         # cite selected sources

    blocks, connections = [], []
    if labels:
        ids = []
        for i, lbl in enumerate(labels):
            bid = re.sub(r"[^a-z0-9]+", "-", lbl.lower()).strip("-") or f"b{i+1}"
            ids.append(bid)
            blocks.append(Block(id=bid, label=lbl, kind=_kind_for(lbl),
                                cites=[sources[j - 1]["id"] for j in src_cites]))
        if chained:
            for a, b in zip(ids, ids[1:]):
                connections.append(Connection(source=a, target=b, label="", kind="data"))
        elif len(ids) > 1:                                       # fan a hub into the parts
            hub, rest = ids[0], ids[1:]
            for b in rest:
                connections.append(Connection(source=hub, target=b, label="", kind="data"))
    else:
        # nothing parseable -> a block per selected source doc/table
        for i, s in enumerate(sources[:6]):
            bid = f"src-{i+1}"
            blocks.append(Block(id=bid, label=s["label"], kind="component",
                                cites=[s["id"]]))
        for a, b in zip([b.id for b in blocks], [b.id for b in blocks][1:]):
            connections.append(Connection(source=a, target=b, label="", kind="data"))
        if not blocks:
            blocks.append(Block(id="request", label=_label(request) or "Request", kind="actor"))

    title = _label(request) or "Block diagram"
    n_src = len(sources)
    reasoning = (
        f"Demo generation (no answer model configured). Parsed the request into "
        f"{len(blocks)} block(s) {'as a pipeline' if chained else 'as related components'}"
        + (f" and grounded them in {n_src} selected source(s)." if n_src else ".")
        + " Configure an LLM endpoint (VLM_BASE_URL) to have a model read the datasheet "
        "pages and design this properly with per-block citations."
    )
    return DiagramSpec(
        title=title, blocks=blocks, connections=connections,
        reasoning=reasoning,
        assumptions=["Generated without an LLM — structure inferred from the request text."],
        structured=False, mode="demo",
    )


# --------------------------------------------------------------------------- run summary / logs
def build_run_summary(request, spec, sources) -> dict:
    """A structured record of one generation: what it studied (the sources), what it produced
    (nodes/connections), the model's reasoning, and everything the constraint/repair machinery
    did. JSON-serializable; `format_run_summary` renders the human version."""
    pages = [{"id": s["id"], "ref": s.get("ref"), "doc": s.get("doc"), "page": s.get("page"),
              "score": s.get("score"), "label": s.get("label")}
             for s in sources if s.get("kind") == "page"]
    tables = [{"id": s["id"], "name": s.get("ref") or s.get("label")}
              for s in sources if s.get("kind") == "table"]
    notes = [{"id": s["id"], "name": s.get("ref") or s.get("label")}
             for s in sources if s.get("kind") == "note"]
    return {
        "request": request,
        "mode": spec.mode,
        "structured": spec.structured,
        "studied": {"pages": pages, "tables": tables, "notes": notes,
                    "n_pages": len(pages), "n_tables": len(tables), "n_notes": len(notes)},
        "produced": {
            "title": spec.title,
            "n_blocks": len(spec.blocks),
            "n_connections": len(spec.connections),
            "blocks": [{"id": b.id, "label": b.label, "kind": b.kind, "cites": b.cites}
                       for b in spec.blocks],
            "connections": [{"from": c.source, "to": c.target, "kind": c.kind, "label": c.label}
                            for c in spec.connections],
            "groups": [{"id": g.id, "label": g.label} for g in spec.groups],
        },
        "reasoning": spec.reasoning,
        "assumptions": list(spec.assumptions),
        "checks": {
            "repair_attempts": spec.repair_attempts,
            "hallucinated_parts": list(spec.hallucinated_parts),
            "remapped_parts": list(spec.remapped_parts),
            "dropped_blocks": spec.dropped_blocks,
            "dropped_connections": spec.dropped_connections,
            "infeasible_connections": spec.infeasible_connections,
            "missing_required": list(spec.missing_required),
            "hallucinated_citations": list(spec.hallucinated_citations),
            "withheld": spec.withheld,
        },
        "errors": list(spec.errors),
    }


def format_run_summary(s: dict) -> str:
    """Human-readable one-screen summary of a run (for the console trace and the .txt log)."""
    st, pr, c = s["studied"], s["produced"], s["checks"]
    out = [f"request : {s['request']!r}",
           f"mode    : {s['mode']} (structured={s['structured']})",
           f"studied : {st['n_pages']} page(s), {st['n_tables']} table(s), {st['n_notes']} note(s)"]
    out += [f"          · {p['label']}  (score {p.get('score')})" for p in st["pages"]]
    out += [f"          · table {t['name']}" for t in st["tables"]]
    out.append(f"produced: {pr['n_blocks']} node(s), {pr['n_connections']} connection(s) — {pr['title']!r}")
    out += [f"          [{b['kind']}] {b['label']}" for b in pr["blocks"]]
    if c["repair_attempts"]:
        out.append(f"repair  : {c['repair_attempts']} re-prompt(s) to fix violations")
    flags = []
    if c["hallucinated_parts"]:
        flags.append(f"dropped {len(c['hallucinated_parts'])} off-catalog node(s): {', '.join(c['hallucinated_parts'])}")
    if c["remapped_parts"]:
        flags.append(f"remapped {len(c['remapped_parts'])} node(s)")
    if c["missing_required"]:
        flags.append(f"missing required: {', '.join(c['missing_required'])}")
    if c["infeasible_connections"]:
        flags.append(f"{c['infeasible_connections']} infeasible edge(s)")
    if c["dropped_connections"]:
        flags.append(f"dropped {c['dropped_connections']} dangling edge(s)")
    if c["hallucinated_citations"]:
        flags.append(f"out-of-range citations: {c['hallucinated_citations']}")
    if c["withheld"]:
        flags.append("WITHHELD — could not be grounded to the catalog")
    out.append("checks  : " + ("; ".join(flags) if flags else "clean"))
    if s["errors"]:
        out.append("errors  : " + "; ".join(s["errors"]))
    return "\n".join(out)


def _write_run_log(summary, settings):
    """Persist a JSON + text summary of the run if COLPALI_RUN_LOG_DIR is set. Never fatal."""
    run_dir = (getattr(settings, "run_log_dir", "") or "").strip()
    if not run_dir:
        return None
    try:
        import datetime
        import json
        from pathlib import Path

        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        slug = re.sub(r"[^a-z0-9]+", "-", (summary["request"] or "run").lower()).strip("-")[:40] or "run"
        base = d / f"{ts}_{slug}"
        base.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        base.with_suffix(".txt").write_text(format_run_summary(summary))
        log.info("run log written: %s.{json,txt}", base)
        return str(base)
    except Exception as e:  # noqa: BLE001 - a log-write failure must never break generation
        log.warning("could not write run log to %s: %s: %s", run_dir, type(e).__name__, e)
        return None


def _finish(request, spec, sources, settings):
    """Single exit: emit the run summary to the log (INFO) and to a file if configured."""
    summary = build_run_summary(request, spec, sources)
    log.info("run summary —\n%s", format_run_summary(summary))
    _write_run_log(summary, settings)
    return spec, sources


# --------------------------------------------------------------------------- entry
def generate_diagram(request, *, store, settings, selected_docs=None, tables=None,
                     notes=None, reranker=None, lock=None, top_k=6, history=None):
    """Produce (DiagramSpec, sources). Uses the LLM when configured, else the demo
    generator. Never raises on a model/parse failure — always returns a diagram.

    When a catalog is active, the model is re-prompted with the SPECIFIC violations (parts not in
    the catalog, illegal connections, missing required items) up to CATALOG_REPAIR_MAX times
    before a terminal gate decides to emit or abstain. With no catalog configured (the default),
    none of this runs and behavior is unchanged."""
    from colpali_rag.studio.catalog import (
        apply_terminal_gate, build_catalog, project, violation_total)

    log.info("generate: request=%r selected=%s top_k=%s", (request or "")[:120],
             sorted(selected_docs) if selected_docs else "(all)", top_k)

    # optional: rewrite a context-dependent follow-up into a standalone RETRIEVAL query using the
    # session history (the original request still drives generation). Off unless COLPALI_QUERY_REWRITE.
    retrieval_query = request
    if (getattr(settings, "query_rewrite", False) and history
            and getattr(settings, "vlm_enabled", False)):
        from colpali_rag.generator import rewrite_query
        retrieval_query = rewrite_query(history, request,
                                        base_url=getattr(settings, "vlm_base_url", None),
                                        api_key=getattr(settings, "vlm_api_key", None),
                                        model=getattr(settings, "vlm_model", ""))
        if retrieval_query != request:
            log.info("generate: rewrote retrieval query -> %r", retrieval_query[:120])

    sources, page_images = collect_sources(
        retrieval_query, store=store, selected_docs=selected_docs, tables=tables, notes=notes,
        top_k=top_k, reranker=reranker, lock=lock, settings=settings)
    n_pg = sum(1 for s in sources if s.get("kind") == "page")
    n_tb = sum(1 for s in sources if s.get("kind") == "table")
    n_nt = sum(1 for s in sources if s.get("kind") == "note")
    log.info("generate: studied %d page(s) + %d table(s) + %d note(s)", n_pg, n_tb, n_nt)

    catalog = build_catalog(tables, settings)              # None unless CATALOG_ID_COL is set
    active = catalog is not None and catalog.gate != "off"
    if active:
        log.info("generate: catalog constraint ON (gate=%s, %d id(s), %d required, %d with interfaces)",
                 catalog.gate, len(catalog.canonical), len(catalog.required), len(catalog.interfaces))

    # demo / no-model path: one projection, then the terminal gate
    if not getattr(settings, "vlm_enabled", False):
        log.info("generate: no model configured -> demo generation")
        spec = _demo_diagram(request, sources)
        if active:
            spec = apply_terminal_gate(spec, catalog, project(spec, catalog))
        return _finish(request, spec, sources, settings)

    # model path: generate -> project -> (if violations remain) re-prompt with them -> repeat
    repair_max = max(0, int(getattr(settings, "catalog_repair_max", 1))) if active else 0
    extra_note, spec, viol = "", None, None
    for attempt in range(repair_max + 1):
        log.info("generate: model call (attempt %d/%d)%s", attempt + 1, repair_max + 1,
                 " with a repair note" if extra_note else "")
        try:
            spec = _llm_diagram(request, page_images, sources, settings,
                                mode=getattr(settings, "answer_structured_mode", "auto"),
                                max_retries=getattr(settings, "answer_max_retries", 1),
                                extra_note=extra_note)
        except Exception as e:  # noqa: BLE001 - transport/other failure -> demo, never 500
            log.warning("diagram LLM failed, using demo: %s: %s", type(e).__name__, e)
            spec = _demo_diagram(request, sources)
            spec.mode = "demo-fallback"
            spec.errors = [f"{type(e).__name__}: {e}"]
        if not active:
            return _finish(request, spec, sources, settings)
        viol = project(spec, catalog)
        spec.repair_attempts = attempt
        log.info("generate: attempt %d -> %d violation(s) (dropped=%d, infeasible=%d, missing=%d)",
                 attempt, violation_total(viol), viol["dropped"], len(viol["infeasible_edges"]),
                 len(viol["missing"]))
        if violation_total(viol) == 0 or attempt >= repair_max or spec.mode == "demo-fallback":
            break
        extra_note = _repair_note(viol)
    spec = apply_terminal_gate(spec, catalog, viol)
    return _finish(request, spec, sources, settings)
