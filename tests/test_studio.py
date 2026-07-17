"""Studio backend tests — model-free. Cover the enforced DiagramSpec validator, the
demo generator, tabular ingestion, exporters, and the HTTP API in demo mode."""

import types


# --------------------------------------------------------------------------- spec / validator
def test_validate_resolves_cites_and_drops_dangling():
    from colpali_rag.studio.spec import validate_diagram_obj

    sources = [{"id": "p1", "kind": "page", "label": "psu.pdf p.2"},
               {"id": "p2", "kind": "page", "label": "mcu.pdf p.4"}]
    obj = {"title": "T", "reasoning": "r", "assumptions": ["a1"],
           "groups": [{"id": "Board", "label": "Board"}],
           "blocks": [{"id": "psu", "label": "PSU", "kind": "external", "group": "Board", "cites": [1]},
                      {"id": "mcu", "label": "MCU", "kind": "system", "group": "nope", "cites": "2"}],  # scalar cite + bad group
           "connections": [{"from": "psu", "to": "mcu", "label": "5V", "kind": "power", "cites": [1]},
                           {"from": "psu", "to": "ghost", "label": "x", "kind": "data", "cites": []}]}  # dangling
    spec = validate_diagram_obj(obj, sources, mode="json_schema")
    assert [b.id for b in spec.blocks] == ["psu", "mcu"]
    assert spec.blocks[1].cites == ["p2"]                 # scalar "2" coerced + resolved
    assert spec.blocks[1].group is None                   # unknown group dropped
    assert len(spec.connections) == 1 and spec.dropped_connections == 1
    assert spec.connections[0].kind == "power"


def test_validate_flags_out_of_range_and_bad_kind():
    from colpali_rag.studio.spec import validate_diagram_obj

    sources = [{"id": "p1", "kind": "page", "label": "a"}]
    obj = {"blocks": [{"id": "b1", "label": "B", "kind": "wormhole", "cites": [1, 9]}]}
    spec = validate_diagram_obj(obj, sources)
    assert spec.blocks[0].kind == "component"             # invalid kind -> component
    assert spec.blocks[0].cites == ["p1"] and 9 in spec.hallucinated_citations


def test_validate_rejects_no_blocks():
    import pytest

    from colpali_rag.studio.spec import validate_diagram_obj

    with pytest.raises(ValueError):
        validate_diagram_obj({"blocks": []}, [])


# --------------------------------------------------------------------------- demo generator
def test_demo_chained_request_builds_pipeline():
    from colpali_rag.studio.generate import generate_diagram

    s = types.SimpleNamespace(vlm_enabled=False)
    spec, sources = generate_diagram(
        "sensor sends signal to the amplifier then to the ADC", store=None, settings=s)
    assert not spec.structured and spec.mode == "demo"
    assert len(spec.blocks) >= 3
    # a chain -> edges connect consecutive blocks
    assert len(spec.connections) == len(spec.blocks) - 1


def test_demo_parallel_request_fans_from_hub():
    from colpali_rag.studio.generate import generate_diagram

    s = types.SimpleNamespace(vlm_enabled=False)
    spec, _ = generate_diagram("a controller with memory, a display, and a radio",
                               store=None, settings=s)
    assert len(spec.blocks) >= 2
    hub = spec.blocks[0].id
    assert all(c.source == hub for c in spec.connections)   # hub -> each part


# --------------------------------------------------------------------------- tabular
def test_load_csv_summary():
    from colpali_rag.studio.tabular import load_csv

    t = load_csv("bom.csv", b"ref,part,qty\nU1,MCU,1\nU2,PSU,1\n")
    assert t.columns == ["ref", "part", "qty"] and t.total_rows == 2
    assert "bom.csv" in t.summary() and "U1" in t.summary()


# --------------------------------------------------------------------------- render
def test_exporters_emit_expected_shapes():
    from colpali_rag.studio.render import to_drawio, to_mermaid
    from colpali_rag.studio.spec import Block, Connection, DiagramSpec

    spec = DiagramSpec(title="D",
                       blocks=[Block("a", "A", "system"), Block("b", "B", "io")],
                       connections=[Connection("a", "b", "x", "power")])
    mm = to_mermaid(spec)
    assert mm.startswith("flowchart LR") and "==>" in mm     # power edge uses heavy link
    dx = to_drawio(spec)
    assert dx.startswith("<mxfile") and 'edge="1"' in dx and "mxGraphModel" in dx


# --------------------------------------------------------------------------- HTTP API (demo)
def test_xlsx_splits_stacked_sections(monkeypatch):
    # a sheet with two cable sections separated by a blank row must become two clean tables,
    # each with its own header and real source-row numbers (not one mangled blob)
    import colpali_rag.studio.tabular as tab

    matrix = [["Cable", "From", "To"], ["C1", "A", "B"], ["C2", "B", "C"],
              [None, None, None],                       # blank separator row (source row 4)
              ["Wire", "Gauge"], ["W1", "22"]]
    monkeypatch.setattr(tab, "_read_xlsx", lambda data: [("Cables", matrix)])
    tables = tab.load_xlsx_sheets("cables.xlsx", b"x")
    assert len(tables) == 2
    assert tables[0].columns == ["Cable", "From", "To"] and tables[0].total_rows == 2
    assert tables[0].row_numbers == [2, 3] and tables[0].sheet == "Cables · section 1"
    assert tables[1].columns == ["Wire", "Gauge"] and tables[1].total_rows == 1
    assert tables[1].row_numbers == [6]               # absolute source row, after the blank


def test_xlsx_single_section_unchanged(monkeypatch):
    import colpali_rag.studio.tabular as tab

    matrix = [["a", "b"], ["1", "2"], ["3", "4"]]
    monkeypatch.setattr(tab, "_read_xlsx", lambda data: [("S", matrix)])
    tables = tab.load_xlsx_sheets("f.xlsx", b"x")
    assert len(tables) == 1 and tables[0].sheet == "S" and tables[0].total_rows == 2


def _raise_400(text):
    import httpx

    def _post(*a, **k):
        req = httpx.Request("POST", "http://x/chat/completions")
        raise httpx.HTTPStatusError("bad", request=req, response=httpx.Response(400, text=text, request=req))
    return _post


def test_llm_diagram_falls_back_to_text_only_when_images_rejected(monkeypatch):
    # a text-only model/gateway rejects the image request; we must retry text-only and still
    # produce a real diagram from the tables — NOT collapse to demo
    import types

    import colpali_rag.generator as gen
    from colpali_rag.studio.generate import _llm_diagram
    from PIL import Image

    seen = []

    def fake_post(base, key, model, messages, *, response_format=None, max_tokens=800,
                  timeout=90.0, temperature=0.0):
        import httpx
        has_img = any(p.get("type") == "image_url" for p in messages[0]["content"])
        seen.append(has_img)
        if has_img:
            req = httpx.Request("POST", "http://x/chat/completions")
            raise httpx.HTTPStatusError("bad", request=req, response=httpx.Response(
                400, text='{"error":"this model does not support image input"}', request=req))
        return {"choices": [{"message": {"content":
                '{"title":"T","reasoning":"r","assumptions":[],"groups":[],'
                '"blocks":[{"id":"a","label":"A","kind":"system","group":null,"cites":[]}],'
                '"connections":[]}'}}]}

    monkeypatch.setattr(gen, "_post_chat", fake_post)
    settings = types.SimpleNamespace(vlm_base_url="http://x", vlm_api_key=None, vlm_model="m")
    spec = _llm_diagram("draw it", [Image.new("RGB", (20, 20), "white")],
                        [{"id": "p1", "kind": "page", "label": "d.pdf · p.1"}],
                        settings, mode="json_object", max_retries=0)
    assert seen == [True, False]                       # tried vision, fell back to text-only
    assert spec.mode != "demo-fallback" and len(spec.blocks) == 1


def test_llm_diagram_surfaces_the_real_error(monkeypatch):
    # when the endpoint keeps 400-ing, the demo-fallback spec must carry the WHY, not a bare code
    import types

    import colpali_rag.generator as gen
    from colpali_rag.studio.generate import _llm_diagram

    monkeypatch.setattr(gen, "_post_chat", _raise_400('{"error":"model \'vlm\' not found"}'))
    settings = types.SimpleNamespace(vlm_base_url="http://x", vlm_api_key=None, vlm_model="vlm")
    spec = _llm_diagram("draw", [], [{"id": "t1", "kind": "table", "label": "x", "text": "a|b"}],
                        settings, mode="json_object", max_retries=0)
    assert spec.mode == "demo-fallback"
    assert any("not found" in e for e in spec.errors)   # the real reason reaches the caller/UI


def test_spec_to_dict_includes_errors():
    from colpali_rag.studio.spec import Block, DiagramSpec

    spec = DiagramSpec(title="T", blocks=[Block("a", "A")], connections=[],
                       errors=["vision/json_object: HTTP 400 — no image support"])
    assert spec.to_dict([])["errors"] == ["vision/json_object: HTTP 400 — no image support"]


def test_api_demo_flow():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from colpali_rag.studio.api import CTX, router

    CTX["settings"] = types.SimpleNamespace(vlm_enabled=False)
    CTX["store"] = None
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)

    assert c.get("/api/studio/status").json()["mode"] == "demo"
    sid = c.post("/api/studio/session").json()["session_id"]
    up = c.post("/api/studio/upload", data={"session_id": sid},
                files=[("files", ("pins.csv", b"pin,net\n1,VCC\n", "text/csv")),
                       ("files", ("nets.csv", b"a,b\n1,2\n", "text/csv"))])
    status = up.json()["status"]
    assert "pins.csv" in status and "nets.csv" in status      # many files in one request
    r = c.post("/api/studio/generate",
               data={"session_id": sid, "message": "power to mcu to sensor"})
    spec = r.json()["spec"]
    assert spec["blocks"] and spec["mode"] == "demo"
    assert c.get(f"/api/studio/export?session_id={sid}&fmt=mermaid").status_code == 200
    assert c.get(f"/api/studio/export?session_id={sid}&fmt=drawio").status_code == 200

    ej = c.get(f"/api/studio/export?session_id={sid}&fmt=json")
    assert ej.status_code == 200 and ej.headers["content-type"].startswith("application/json")
    assert ej.json()["blocks"]                                  # structured spec downloadable
    es = c.get(f"/api/studio/export?session_id={sid}&fmt=summary")
    assert es.status_code == 200 and "produced:" in es.text     # human-readable run summary
