"""Per-generation run summary + saved log file — model-free."""

import json
import types


def _spec_and_sources():
    from colpali_rag.studio.spec import Block, Connection, DiagramSpec

    spec = DiagramSpec(
        title="D",
        blocks=[Block("a", "AX-100", "component", cites=["p1"]), Block("b", "BX-200", "system")],
        connections=[Connection("a", "b", "5V", "power")],
        reasoning="because", assumptions=["assumed X"], mode="json_object",
        hallucinated_parts=["ZX-9"], remapped_parts=[{"from": "AX 100", "to": "AX-100"}],
        dropped_blocks=1, repair_attempts=1, missing_required=["CX-3"],
    )
    sources = [{"id": "p1", "kind": "page", "ref": "d.pdf::p2", "doc": "d.pdf", "page": 2,
                "score": 0.9, "label": "d.pdf · p.2"},
               {"id": "t1", "kind": "table", "ref": "sheet.csv", "label": "sheet.csv"}]
    return spec, sources


def test_build_run_summary_captures_what_happened():
    from colpali_rag.studio.generate import build_run_summary

    spec, sources = _spec_and_sources()
    s = build_run_summary("draw it", spec, sources)
    assert s["request"] == "draw it" and s["mode"] == "json_object"
    assert s["studied"]["n_pages"] == 1 and s["studied"]["n_tables"] == 1
    assert s["produced"]["n_blocks"] == 2 and s["produced"]["n_connections"] == 1
    assert [b["label"] for b in s["produced"]["blocks"]] == ["AX-100", "BX-200"]
    assert s["checks"]["hallucinated_parts"] == ["ZX-9"] and s["checks"]["repair_attempts"] == 1
    assert s["checks"]["missing_required"] == ["CX-3"]
    json.dumps(s)   # must be JSON-serializable


def test_format_run_summary_is_readable():
    from colpali_rag.studio.generate import build_run_summary, format_run_summary

    txt = format_run_summary(build_run_summary("draw it", *_spec_and_sources()))
    assert "studied : 1 page(s), 1 table(s)" in txt
    assert "produced: 2 node(s), 1 connection(s)" in txt
    assert "dropped 1 off-catalog node(s): ZX-9" in txt
    assert "missing required: CX-3" in txt


def test_format_shows_surfaced_rows_and_trajectory():
    from colpali_rag.studio.generate import build_run_summary, format_run_summary
    from colpali_rag.studio.spec import Block, DiagramSpec

    spec = DiagramSpec(title="D", blocks=[Block("a", "AX-100")], connections=[],
                       refine_trajectory=[{"attempt": 0, "violations": 2}, {"attempt": 1, "violations": 0}])
    sources = [{"id": "t1", "kind": "table", "ref": "big.csv", "label": "big.csv",
                "total_rows": 500, "shown_rows": [316, 42, 7]}]
    txt = format_run_summary(build_run_summary("draw", spec, sources))
    assert "surfaced source rows 316, 42, 7 of 500" in txt
    assert "refine  : a0:2v  a1:0v" in txt


def test_run_log_written_when_dir_set(tmp_path):
    from colpali_rag.studio.generate import _write_run_log, build_run_summary

    spec, sources = _spec_and_sources()
    summary = build_run_summary("draw it", spec, sources)
    path = _write_run_log(summary, types.SimpleNamespace(run_log_dir=str(tmp_path)))
    assert path is not None
    js, tx = list(tmp_path.glob("*.json")), list(tmp_path.glob("*.txt"))
    assert len(js) == 1 and len(tx) == 1
    assert json.loads(js[0].read_text())["produced"]["n_blocks"] == 2
    assert "produced: 2 node(s)" in tx[0].read_text()
    # no dir configured -> no write, no error
    assert _write_run_log(summary, types.SimpleNamespace(run_log_dir="")) is None
