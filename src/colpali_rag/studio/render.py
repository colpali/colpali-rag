"""Export a DiagramSpec to portable formats. The React canvas renders the spec JSON
directly; these give you a diagram you can paste into a Markdown doc (Mermaid) or open
and hand-edit anywhere (draw.io / diagrams.net .drawio XML).

No layout engine is pulled in — draw.io export uses a simple layered grid and lets you
re-flow with draw.io's own arrange tools. Mermaid delegates layout to Mermaid.
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

from colpali_rag.studio.spec import DiagramSpec

_MERMAID_SHAPE = {              # block kind -> mermaid node bracket pair
    "component": ("[", "]"), "system": ("[[", "]]"), "process": ("([", "])"),
    "io": ("[/", "/]"), "store": ("[(", ")]"), "external": ("{{", "}}"), "actor": ("((", "))"),
}
_MERMAID_LINK = {"control": "-.->", "dependency": "-.->", "power": "==>", "bus": "==>"}


def _mm_id(bid: str) -> str:
    return "n_" + bid.replace("-", "_")


def to_mermaid(spec: DiagramSpec) -> str:
    lines = ["flowchart LR"]
    by_group: dict[str | None, list] = {}
    for b in spec.blocks:
        by_group.setdefault(b.group, []).append(b)
    glabels = {g.id: g.label for g in spec.groups}

    def emit_block(b, indent="    "):
        o, c = _MERMAID_SHAPE.get(b.kind, ("[", "]"))
        lines.append(f'{indent}{_mm_id(b.id)}{o}"{b.label}"{c}')

    for b in by_group.get(None, []):
        emit_block(b)
    for gid, members in by_group.items():
        if gid is None:
            continue
        lines.append(f'    subgraph {_mm_id(gid)}["{glabels.get(gid, gid)}"]')
        for b in members:
            emit_block(b, "        ")
        lines.append("    end")
    for e in spec.connections:
        link = _MERMAID_LINK.get(e.kind, "-->")
        lbl = f'|"{e.label}"|' if e.label else ""
        lines.append(f"    {_mm_id(e.source)} {link}{lbl} {_mm_id(e.target)}")
    return "\n".join(lines)


_STYLE = {
    "component": "rounded=1;whiteSpace=wrap;html=1;fillColor=#1f2937;strokeColor=#38bdf8;fontColor=#e5e7eb;",
    "system": "shape=process;whiteSpace=wrap;html=1;fillColor=#111827;strokeColor=#818cf8;fontColor=#e5e7eb;",
    "process": "rounded=1;whiteSpace=wrap;html=1;fillColor=#0b3b3b;strokeColor=#2dd4bf;fontColor=#e5e7eb;",
    "io": "shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#3b2f0b;strokeColor=#fbbf24;fontColor=#e5e7eb;",
    "store": "shape=cylinder;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#a78bfa;fontColor=#e5e7eb;",
    "external": "rhombus;whiteSpace=wrap;html=1;fillColor=#3f1d2e;strokeColor=#fb7185;fontColor=#e5e7eb;",
    "actor": "ellipse;whiteSpace=wrap;html=1;fillColor=#132a1a;strokeColor=#4ade80;fontColor=#e5e7eb;",
}
_EDGE_STYLE = {
    "power": "endArrow=block;strokeWidth=2.5;strokeColor=#fbbf24;html=1;",
    "control": "endArrow=open;dashed=1;strokeColor=#818cf8;html=1;",
    "dependency": "endArrow=open;dashed=1;strokeColor=#94a3b8;html=1;",
    "signal": "endArrow=classic;strokeColor=#2dd4bf;html=1;",
    "bus": "endArrow=block;strokeWidth=3;strokeColor=#a78bfa;html=1;",
}
_DEFAULT_EDGE = "endArrow=classic;html=1;strokeColor=#38bdf8;"


def to_drawio(spec: DiagramSpec) -> str:
    """Minimal mxGraphModel with a layered grid layout. Blocks flow left→right by their
    depth from a source (topological-ish); grouped blocks stack together."""
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    # crude longest-path layering so arrows mostly point rightward
    ids = [b.id for b in spec.blocks]
    incoming = {i: 0 for i in ids}
    for e in spec.connections:
        if e.target in incoming:
            incoming[e.target] += 1
    layer = {i: 0 for i in ids}
    for _ in range(len(ids)):
        for e in spec.connections:
            if e.source in layer and e.target in layer:
                layer[e.target] = max(layer[e.target], layer[e.source] + 1)
    col_count: dict[int, int] = {}
    W, H, GAP_X, GAP_Y = 160, 60, 90, 40
    for b in spec.blocks:
        lx = layer[b.id]
        row = col_count.get(lx, 0)
        col_count[lx] = row + 1
        x = 40 + lx * (W + GAP_X)
        y = 40 + row * (H + GAP_Y)
        style = _STYLE.get(b.kind, _STYLE["component"])
        cells.append(
            f'<mxCell id={quoteattr(b.id)} value={quoteattr(b.label)} style={quoteattr(style)} '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{W}" height="{H}" '
            f'as="geometry"/></mxCell>'
        )
    for k, e in enumerate(spec.connections):
        style = _EDGE_STYLE.get(e.kind, _DEFAULT_EDGE)
        cells.append(
            f'<mxCell id={quoteattr("e" + str(k))} value={quoteattr(e.label)} style={quoteattr(style)} '
            f'edge="1" parent="1" source={quoteattr(e.source)} target={quoteattr(e.target)}>'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
    body = "".join(cells)
    return (
        '<mxfile host="colpali-studio">'
        f'<diagram name={quoteattr(spec.title or "diagram")}>'
        f'<mxGraphModel dx="800" dy="600" grid="1" gridSize="10" guides="1" '
        f'background="#0b1020"><root>{body}</root></mxGraphModel>'
        '</diagram></mxfile>'
    )
