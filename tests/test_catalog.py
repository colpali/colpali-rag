"""Closed-vocabulary constraint tests — model-free, generic fixtures only.

Covers: full-table retention (the constraint channel is never truncated), the ID-preserving
canonicalizer vs the renderer's slug, the matcher (exact / fuzzy remap / threshold + margin
reject), catalog compilation (id/alias/required, collisions, feature-off), and the projection
Pi (drop + record, remap + record, withhold), including the demo path gated through it.
"""

import types

from colpali_rag.studio.catalog import Catalog, apply_catalog, build_catalog, canon, id_tokens
from colpali_rag.studio.tabular import Table


def _table(name, columns, rows):
    return Table(name=name, columns=columns, rows=rows, total_rows=len(rows))


def _cat(ids, *, required=(), gate="flag", tau=0.84, delta=0.08):
    return Catalog(keys={canon(i): i for i in ids}, canonical=set(ids), required=set(required),
                   gate=gate, tau=tau, delta=delta)


def _spec(blocks, connections=(), sources=None):
    from colpali_rag.studio.spec import validate_diagram_obj
    sources = sources or [{"id": "p1", "kind": "page", "label": "a"}]
    obj = {"blocks": [{"id": bid, "label": lbl, "kind": "component", "cites": [1]}
                      for bid, lbl in blocks],
           "connections": [{"from": s, "to": t, "label": "", "kind": "data", "cites": []}
                           for s, t in connections]}
    return validate_diagram_obj(obj, sources)


# --------------------------------------------------------------------------- full retention
def test_table_retains_full_data():
    from colpali_rag.studio.tabular import load_csv

    header = ",".join(f"c{i}" for i in range(30))
    big = "x" * 200
    body = []
    for r in range(60):
        cells = [f"r{r}"] + [big if (r == 0 and c == 1) else f"v{r}_{c}" for c in range(1, 30)]
        body.append(",".join(cells))
    data = (header + "\n" + "\n".join(body) + "\n").encode()
    t = load_csv("big.csv", data)

    # constraint channel: every row, every column, uncapped cell retained
    assert len(t.columns) == 30
    assert t.total_rows == 60 and len(t.rows) == 60
    assert t.rows[0][1] == big and len(t.rows[0][1]) == 200

    # display channel: capped + honest about what it hid
    s = t.summary()
    assert "… (20 more row(s))" in s                 # 60 rows, 40 shown
    assert ("x" * 80 + "…") in s                      # the 200-char cell is clipped for display
    assert "… (55 more row(s))" in t.summary(max_rows=5)


def test_summary_respects_caps():
    from colpali_rag.studio.tabular import load_csv

    t = load_csv("t.csv", b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
    assert t.total_rows == 3
    assert "… (" not in t.summary()                   # 3 <= 40, nothing hidden
    assert "… (2 more row(s))" in t.summary(max_rows=1)


# --------------------------------------------------------------------------- canon / tokens
def test_canon_preserves_internal_id_punctuation():
    assert canon("(AX-1234)") == "ax-1234"            # surrounding punctuation stripped
    assert canon("AX.12/34") == "ax.12/34"            # internal . and / preserved
    assert canon("AX   12   34") == "ax 12 34"        # whitespace collapsed, not removed
    assert canon("AX-12.34") != canon("AX 12 34")     # distinct ids stay distinct
    assert id_tokens("AX-1234 net") == {"ax-1234", "net"}   # id kept whole


def test_slug_would_mangle_ids():
    from colpali_rag.studio.spec import _slug

    # the renderer slug collapses distinct separators to '-', merging keys canon keeps apart
    assert _slug("AX.12/34", "") == "ax-12-34"
    assert _slug("AX 12 34", "") == "ax-12-34"
    assert canon("AX.12/34") != canon("AX 12 34")     # why slug is unsafe as a match key


# --------------------------------------------------------------------------- matcher
def test_match_exact_bypasses_margin():
    cat = _cat(["AX-100", "AX-101"])                  # near neighbor present
    m = cat.match("AX-100")
    assert m.status == "exact" and m.canonical == "AX-100" and m.score == 1.0
    assert cat.match("ax-100").status == "exact"      # casefold still exact


def test_match_fuzzy_remap():
    cat = _cat(["AX-100", "AX-101", "BX-200"])
    m = cat.match("AX 100")                            # space-for-hyphen variant
    assert m.status == "remap" and m.canonical == "AX-100"
    assert m.score >= cat.tau


def test_match_unresolved_below_threshold():
    cat = _cat(["AX-100", "AX-101", "BX-200"])
    assert cat.match("ZX-999").status == "unresolved"


def test_match_margin_guard_rejects_ambiguous():
    # lower tau so both candidates clear threshold; equidistant label must still be rejected
    cat = _cat(["AX-100", "AX-900"], tau=0.5, delta=0.2)
    m = cat.match("AX-500")
    assert m.status == "unresolved"                   # score above tau but margin below delta


# --------------------------------------------------------------------------- compilation
def test_build_catalog_none_when_unset_or_absent():
    t = _table("t.csv", ["a", "b"], [["1", "2"]])
    assert build_catalog([t], types.SimpleNamespace(catalog_id_col="")) is None
    # id col configured but no uploaded table carries it -> feature stays off
    assert build_catalog([t], types.SimpleNamespace(catalog_id_col="part_id")) is None


def test_build_catalog_compiles_ids_aliases_required():
    t = _table("c.csv", ["part_id", "name", "must"],
               [["AX-100", "Alpha", "yes"], ["BX-200", "Bravo", ""], ["CX-300", "Charlie", "1"]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_name_cols="name",
                              catalog_required_col="must", catalog_gate="flag")
    cat = build_catalog([t], s)
    assert cat is not None
    assert cat.canonical == {"AX-100", "BX-200", "CX-300"}
    assert cat.required == {"AX-100", "CX-300"}
    assert cat.match("Alpha").canonical == "AX-100"   # alias resolves to the id


def test_build_catalog_surfaces_collisions():
    t = _table("c.csv", ["part_id", "name"], [["AX-100", "Alpha"], ["AX-200", "alpha"]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_name_cols="name",
                              catalog_gate="flag")
    cat = build_catalog([t], s)
    assert cat.conflicts                              # 'alpha' alias claimed by two ids
    assert cat.keys[canon("alpha")] == "AX-100"       # first occurrence kept, not silently swapped


# --------------------------------------------------------------------------- projection Pi
def test_apply_catalog_none_and_off_are_noops():
    spec = _spec([("b1", "Zzz")])
    before = spec.to_dict()
    assert apply_catalog(spec, None).to_dict() == before
    spec2 = _spec([("b1", "Zzz")])
    apply_catalog(spec2, _cat(["AX-100"], gate="off"))
    assert spec2.blocks[0].label == "Zzz" and spec2.dropped_blocks == 0


def test_projection_drops_and_records():
    from colpali_rag.studio.spec import validate_diagram_obj

    cat = _cat(["AX-100", "BX-200"], gate="flag")
    obj = {"blocks": [{"id": "ax", "label": "AX-100", "kind": "component", "cites": [1]},
                      {"id": "zz", "label": "ZX-999", "kind": "component", "cites": [1]}],
           "connections": [{"from": "ax", "to": "zz", "label": "", "kind": "data", "cites": []}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    assert [b.label for b in spec.blocks] == ["AX-100"]
    assert [b.id for b in spec.blocks] == ["ax"]              # id stays stable
    assert spec.hallucinated_parts == ["ZX-999"] and spec.dropped_blocks == 1
    assert spec.connections == [] and spec.dropped_connections == 1


def test_projection_remap_records_assumption():
    from colpali_rag.studio.spec import validate_diagram_obj

    cat = _cat(["AX-100", "AX-101", "BX-200"], gate="flag")
    obj = {"blocks": [{"id": "n1", "label": "AX 100", "kind": "component", "cites": [1]}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    assert spec.blocks[0].label == "AX-100" and spec.blocks[0].id == "n1"   # remapped, id kept
    assert spec.remapped_parts[0]["from"] == "AX 100" and spec.remapped_parts[0]["to"] == "AX-100"
    assert any("AX 100" in a and "AX-100" in a for a in spec.assumptions)   # never silent


def test_projection_withhold_abstains():
    from colpali_rag.studio.spec import validate_diagram_obj

    cat = _cat(["AX-100", "BX-200"], gate="withhold")
    obj = {"blocks": [{"id": "z1", "label": "ZX-1", "kind": "component", "cites": [1]},
                      {"id": "z2", "label": "ZX-2", "kind": "component", "cites": [1]}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    assert spec.withheld is True and spec.blocks == [] and spec.connections == []
    assert spec.hallucinated_parts == ["ZX-1", "ZX-2"]


def test_withhold_keeps_when_mostly_grounded():
    from colpali_rag.studio.spec import validate_diagram_obj

    cat = _cat(["AX-100", "BX-200", "CX-300"], gate="withhold")
    obj = {"blocks": [{"id": "a", "label": "AX-100", "kind": "component", "cites": [1]},
                      {"id": "b", "label": "BX-200", "kind": "component", "cites": [1]},
                      {"id": "z", "label": "ZX-9", "kind": "component", "cites": [1]}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    assert spec.withheld is False                             # 1/3 dropped < 0.5
    assert {b.label for b in spec.blocks} == {"AX-100", "BX-200"}


# --------------------------------------------------------------------------- demo path gated
def test_demo_path_gated_through_projection():
    from colpali_rag.studio.generate import generate_diagram

    cat_table = _table("cat.csv", ["part_id"], [["AX-100"], ["BX-200"], ["CX-300"]])
    s = types.SimpleNamespace(vlm_enabled=False, catalog_id_col="part_id", catalog_gate="flag")
    spec, _ = generate_diagram("AX-100 feeds BX-200 then ZX-999",
                               store=None, settings=s, tables=[cat_table])
    labels = [b.label for b in spec.blocks]
    assert labels and all(l in {"AX-100", "BX-200", "CX-300"} for l in labels)  # catalog-only
    assert spec.dropped_blocks >= 1                          # the out-of-catalog node dropped
    assert any(r["to"] == "AX-100" for r in spec.remapped_parts)   # demo label remapped


# --------------------------------------------------------------------------- fix regressions
def test_canon_edge_id_chars_preserved():
    # a leading/trailing id character is NOT stripped, so distinct ids stay distinct...
    assert canon(".net") == ".net" and canon("net") == "net"
    assert canon("5.") == "5." and canon(".5") == ".5"
    assert canon(".net") != canon("net")                    # invariant (d) holds
    assert canon("(5)") == "5"                              # ...but wrapping punctuation still goes
    # '.net' no longer swallows 'net' as a confident-exact (score 1.0) hash hit
    m = _cat([".net"]).match("net")
    assert m.status != "exact" and m.score < 1.0


def test_short_id_trailing_punctuation_resolves():
    # a model that appends a sentence period to a SHORT id still resolves (as a remap), not drops
    cat = _cat(["R1", "R2", "AX-100"])
    assert cat.match("R1.").status == "remap" and cat.match("R1.").canonical == "R1"
    assert cat.match("AX-100.").canonical == "AX-100"
    # ...yet genuinely-distinct ids remain distinct: literal-exact wins, no merge
    cat2 = _cat(["R1", "R1-"])
    assert cat2.match("R1").status == "exact" and cat2.match("R1").canonical == "R1"
    assert cat2.match("R1-").status == "exact" and cat2.match("R1-").canonical == "R1-"


# --------------------------------------------------------------------------- feasibility (edges)
def test_feasibility_shared_interface():
    t = _table("cat.csv", ["part_id", "iface"],
               [["AX-100", "busX"], ["BX-200", "busX"], ["CX-300", "busY"]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_iface_cols="iface",
                              catalog_gate="flag")
    cat = build_catalog([t], s)
    assert cat.feasible("AX-100", "BX-200") is True       # share busX
    assert cat.feasible("AX-100", "CX-300") is False      # busX vs busY
    assert _cat(["AX-100", "BX-200"]).feasible("AX-100", "BX-200") is None  # no interface data


def test_infeasible_edges_flagged_not_dropped():
    from colpali_rag.studio.spec import validate_diagram_obj

    t = _table("cat.csv", ["part_id", "iface"],
               [["AX-100", "busX"], ["BX-200", "busX"], ["CX-300", "busY"]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_iface_cols="iface",
                              catalog_gate="flag")
    cat = build_catalog([t], s)
    obj = {"blocks": [{"id": "a", "label": "AX-100", "kind": "component", "cites": [1]},
                      {"id": "b", "label": "BX-200", "kind": "component", "cites": [1]},
                      {"id": "c", "label": "CX-300", "kind": "component", "cites": [1]}],
           "connections": [{"from": "a", "to": "b", "kind": "data", "label": "", "cites": []},
                           {"from": "a", "to": "c", "kind": "data", "label": "", "cites": []}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    assert spec.infeasible_connections == 1               # a->c is illegal
    assert len(spec.connections) == 2                     # ...but flagged, NOT deleted


# --------------------------------------------------------------------------- required completeness
def test_required_completeness_reports_missing():
    from colpali_rag.studio.spec import validate_diagram_obj

    t = _table("cat.csv", ["part_id", "must"],
               [["AX-100", "yes"], ["BX-200", "yes"], ["CX-300", ""]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_required_col="must",
                              catalog_gate="flag")
    cat = build_catalog([t], s)
    assert cat.required == {"AX-100", "BX-200"}
    obj = {"blocks": [{"id": "a", "label": "AX-100", "kind": "component", "cites": [1]}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    assert spec.missing_required == ["BX-200"]


def test_withhold_on_hard_violations():
    from colpali_rag.studio.spec import validate_diagram_obj

    # missing a required item -> abstain
    t = _table("c.csv", ["part_id", "must"], [["AX-100", "yes"], ["BX-200", "yes"]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_required_col="must",
                              catalog_gate="withhold")
    cat = build_catalog([t], s)
    obj = {"blocks": [{"id": "a", "label": "AX-100", "kind": "component", "cites": [1]}]}
    assert validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat).withheld is True

    # an infeasible connection -> abstain
    t2 = _table("c.csv", ["part_id", "iface"], [["AX-100", "busX"], ["CX-300", "busY"]])
    s2 = types.SimpleNamespace(catalog_id_col="part_id", catalog_iface_cols="iface",
                               catalog_gate="withhold")
    cat2 = build_catalog([t2], s2)
    obj2 = {"blocks": [{"id": "a", "label": "AX-100", "kind": "component", "cites": [1]},
                       {"id": "c", "label": "CX-300", "kind": "component", "cites": [1]}],
            "connections": [{"from": "a", "to": "c", "kind": "data", "label": "", "cites": []}]}
    assert validate_diagram_obj(obj2, [{"id": "p1", "label": "a"}], catalog=cat2).withheld is True


# --------------------------------------------------------------------------- verify-and-repair loop
def _fake_settings(**kw):
    base = dict(vlm_enabled=True, catalog_id_col="part_id", catalog_gate="flag",
                catalog_repair_max=1, answer_structured_mode="auto", answer_max_retries=1,
                tabular_max_preview_rows=40, tabular_max_cols=24, tabular_max_cell=80)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_repair_loop_reprompts_with_named_violations(monkeypatch):
    from colpali_rag.studio import generate as gen
    from colpali_rag.studio.spec import Block, DiagramSpec

    cat_table = _table("cat.csv", ["part_id"], [["AX-100"], ["BX-200"]])
    calls = []

    def fake_llm(request, page_images, sources, settings, *, mode="auto", max_retries=1, extra_note=""):
        calls.append(extra_note)
        if len(calls) == 1:                               # first try hallucinates a node
            return DiagramSpec(title="T", blocks=[Block("a", "AX-100"), Block("z", "ZX-999")],
                               connections=[])
        return DiagramSpec(title="T", blocks=[Block("a", "AX-100"), Block("b", "BX-200")],
                           connections=[])                # second try is clean

    monkeypatch.setattr(gen, "_llm_diagram", fake_llm)
    spec, _ = gen.generate_diagram("draw it", store=None, settings=_fake_settings(),
                                   tables=[cat_table])
    assert len(calls) == 2 and calls[0] == "" and "ZX-999" in calls[1]   # re-prompted with the violation
    assert {b.label for b in spec.blocks} == {"AX-100", "BX-200"} and spec.repair_attempts == 1


def test_repair_loop_is_bounded(monkeypatch):
    from colpali_rag.studio import generate as gen
    from colpali_rag.studio.spec import Block, DiagramSpec

    cat_table = _table("cat.csv", ["part_id"], [["AX-100"]])
    n = []

    def fake_llm(*a, extra_note="", **k):
        n.append(1)
        return DiagramSpec(title="T", blocks=[Block("a", "AX-100"), Block("z", "ZX-999")],
                           connections=[])

    monkeypatch.setattr(gen, "_llm_diagram", fake_llm)
    spec, _ = gen.generate_diagram("x", store=None, settings=_fake_settings(catalog_repair_max=2),
                                   tables=[cat_table])
    assert len(n) == 3                                    # initial + 2 repairs, then stop
    assert spec.repair_attempts == 2
    assert [b.label for b in spec.blocks] == ["AX-100"]   # hallucinated node dropped, emitted under flag


# --------------------------------------------------------------------------- foolproof fix regressions
def test_iface_separator_variants_are_compatible():
    t = _table("c.csv", ["part_id", "iface"], [["AX-100", "USB-C"], ["BX-200", "USB C"]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_iface_cols="iface", catalog_gate="flag")
    cat = build_catalog([t], s)
    assert cat.feasible("AX-100", "BX-200") is True       # USB-C == USB C == USBC


def test_iface_placeholders_ignored():
    t = _table("c.csv", ["part_id", "iface"], [["A", "busX, N/A"], ["B", "busY, N/A"]])
    cat = build_catalog([t], types.SimpleNamespace(catalog_id_col="part_id",
                        catalog_iface_cols="iface", catalog_gate="flag"))
    assert cat.feasible("A", "B") is False                # 'N/A' dropped -> busX vs busY
    t2 = _table("c.csv", ["part_id", "iface"], [["A", "N/A"], ["B", "-"]])
    cat2 = build_catalog([t2], types.SimpleNamespace(catalog_id_col="part_id",
                         catalog_iface_cols="iface", catalog_gate="flag"))
    assert cat2.feasible("A", "B") is None                # only placeholders -> no interface data


def test_negative_repair_max_still_runs_once(monkeypatch):
    from colpali_rag.studio import generate as gen
    from colpali_rag.studio.spec import Block, DiagramSpec

    cat_table = _table("cat.csv", ["part_id"], [["AX-100"]])
    n = []

    def fake_llm(*a, extra_note="", **k):
        n.append(1)
        return DiagramSpec(title="T", blocks=[Block("a", "AX-100")], connections=[])

    monkeypatch.setattr(gen, "_llm_diagram", fake_llm)
    spec, _ = gen.generate_diagram("x", store=None, tables=[cat_table],
                                   settings=_fake_settings(catalog_repair_max=-1, catalog_gate="withhold"))
    assert len(n) == 1 and spec is not None and [b.label for b in spec.blocks] == ["AX-100"]


def test_apply_catalog_withheld_is_idempotent():
    from colpali_rag.studio.spec import validate_diagram_obj

    t = _table("c.csv", ["part_id", "must"], [["AX-100", "yes"]])
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_required_col="must",
                              catalog_gate="withhold", catalog_withhold_max_drop=0.5)
    cat = build_catalog([t], s)
    obj = {"blocks": [{"id": "a", "label": "AX-100", "kind": "component", "cites": [1]},
                      {"id": "j1", "label": "JUNK1", "kind": "component", "cites": [1]},
                      {"id": "j2", "label": "JUNK2", "kind": "component", "cites": [1]}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    assert spec.withheld is True
    before = (list(spec.missing_required), sum("Withheld" in a for a in spec.assumptions))
    apply_catalog(spec, cat)                              # second pass must be a no-op
    after = (list(spec.missing_required), sum("Withheld" in a for a in spec.assumptions))
    assert after == before                                # missing_required not corrupted; no dup assumption


def test_apply_catalog_is_idempotent():
    from colpali_rag.studio.spec import validate_diagram_obj

    cat = _cat(["AX-100"], gate="flag")
    obj = {"blocks": [{"id": "a", "label": "AX 100", "kind": "component", "cites": [1]},
                      {"id": "z", "label": "junk", "kind": "component", "cites": [1]}]}
    spec = validate_diagram_obj(obj, [{"id": "p1", "label": "a"}], catalog=cat)
    snap = (list(spec.hallucinated_parts), list(spec.remapped_parts),
            spec.dropped_blocks, spec.dropped_connections)
    apply_catalog(spec, cat)                                 # a second pass must not wipe the audit
    assert (spec.hallucinated_parts, spec.remapped_parts,
            spec.dropped_blocks, spec.dropped_connections) == snap


def test_multi_sheet_workbook_reaches_the_vocabulary():
    import io

    import pytest
    openpyxl = pytest.importorskip("openpyxl")
    from colpali_rag.studio.tabular import load_tables

    wb = openpyxl.Workbook()
    cover = wb.active
    cover.title = "Cover"
    cover.append(["notes"])
    cover.append(["intro"])
    parts = wb.create_sheet("Parts")             # the catalog lives on a NON-active sheet
    parts.append(["part_id", "name"])
    parts.append(["AX-100", "Alpha"])
    parts.append(["BX-200", "Bravo"])
    buf = io.BytesIO()
    wb.save(buf)

    ts = load_tables("book.xlsx", buf.getvalue())
    assert any("part_id" in t.columns for t in ts)          # non-active sheet retained
    s = types.SimpleNamespace(catalog_id_col="part_id", catalog_gate="flag")
    cat = build_catalog(ts, s)
    assert cat is not None and cat.canonical == {"AX-100", "BX-200"}
