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
                files={"file": ("pins.csv", b"pin,net\n1,VCC\n", "text/csv")})
    assert "pins.csv" in up.json()["status"]
    r = c.post("/api/studio/generate",
               data={"session_id": sid, "message": "power to mcu to sensor"})
    spec = r.json()["spec"]
    assert spec["blocks"] and spec["mode"] == "demo"
    assert c.get(f"/api/studio/export?session_id={sid}&fmt=mermaid").status_code == 200
    assert c.get(f"/api/studio/export?session_id={sid}&fmt=drawio").status_code == 200
