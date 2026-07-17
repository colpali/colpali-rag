"""DiagramSpec — the enforced, cited structure a diagram request must produce.

Same discipline as colpali_rag.schemas: the model returns bracket-index citations
([1], [2], …) into the *attached sources* (retrieved datasheet pages + selected
tables) in the order they were shown — NOT free-form ids it might mangle. We resolve
each index back to the real source here and flag out-of-range indices as
hallucinated citations. Zero third-party deps so it imports in the engine core.

The shape is deliberately renderer-agnostic: `blocks` + `connections` + `groups` map
cleanly onto React Flow nodes/edges, Mermaid, or draw.io without any of them leaking
into the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from colpali_rag.schemas import parse_json  # reuse the robust json.loads -> extract -> repair

BLOCK_KINDS = ("component", "system", "process", "io", "store", "external", "actor")
EDGE_KINDS = ("data", "control", "signal", "power", "bus", "dependency")

# Strict-mode friendly (all required, additionalProperties false) so a json_schema
# endpoint accepts it; degrades cleanly to json_object / prompt for endpoints that don't.
DIAGRAM_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "reasoning": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
                "required": ["id", "label"],
            },
        },
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": list(BLOCK_KINDS)},
                    "group": {"type": "string"},
                    "cites": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["id", "label", "kind", "group", "cites"],
            },
        },
        "connections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": list(EDGE_KINDS)},
                    "cites": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["from", "to", "label", "kind", "cites"],
            },
        },
    },
    "required": ["title", "reasoning", "assumptions", "groups", "blocks", "connections"],
}


@dataclass
class Block:
    id: str
    label: str
    kind: str = "component"
    group: str | None = None
    cites: list[str] = field(default_factory=list)   # resolved source ids


@dataclass
class Connection:
    source: str                      # block id (JSON key is "from")
    target: str                      # block id (JSON key is "to")
    label: str = ""
    kind: str = "data"
    cites: list[str] = field(default_factory=list)


@dataclass
class Group:
    id: str
    label: str


@dataclass
class DiagramSpec:
    title: str
    blocks: list[Block]
    connections: list[Connection]
    groups: list[Group] = field(default_factory=list)
    reasoning: str = ""
    assumptions: list[str] = field(default_factory=list)
    structured: bool = True          # False => model didn't return usable JSON (fallback)
    mode: str = "prompt"             # which cascade tier produced it, or "demo"
    hallucinated_citations: list[int] = field(default_factory=list)
    dropped_connections: int = 0     # edges referencing unknown blocks (removed)
    hallucinated_parts: list[str] = field(default_factory=list)   # node labels not in the catalog (dropped)
    remapped_parts: list[dict] = field(default_factory=list)      # {from,to,score,block} catalog rewrites
    dropped_blocks: int = 0          # nodes removed because they weren't in the catalog
    infeasible_connections: int = 0  # edges flagged: the two catalog items may not connect
    missing_required: list[str] = field(default_factory=list)     # required catalog ids absent from the output
    repair_attempts: int = 0         # how many times the model was re-prompted to fix violations
    refine_trajectory: list[dict] = field(default_factory=list)   # per-attempt critic record (violations over time)
    withheld: bool = False           # True => abstained: output couldn't be grounded to the catalog
    errors: list[str] = field(default_factory=list)

    def to_dict(self, sources: list | None = None) -> dict:
        by_id = {s["id"]: s for s in (sources or [])}
        def cite_objs(ids):
            return [by_id[i] for i in ids if i in by_id]
        return {
            "title": self.title,
            "reasoning": self.reasoning,
            "assumptions": self.assumptions,
            "groups": [{"id": g.id, "label": g.label} for g in self.groups],
            "blocks": [{"id": b.id, "label": b.label, "kind": b.kind, "group": b.group,
                        "cites": b.cites, "citations": cite_objs(b.cites)} for b in self.blocks],
            "connections": [{"from": c.source, "to": c.target, "label": c.label, "kind": c.kind,
                             "cites": c.cites, "citations": cite_objs(c.cites)}
                            for c in self.connections],
            "structured": self.structured,
            "mode": self.mode,
            "hallucinated_citations": self.hallucinated_citations,
            "dropped_connections": self.dropped_connections,
            "hallucinated_parts": self.hallucinated_parts,
            "remapped_parts": self.remapped_parts,
            "dropped_blocks": self.dropped_blocks,
            "infeasible_connections": self.infeasible_connections,
            "missing_required": self.missing_required,
            "repair_attempts": self.repair_attempts,
            "refine_trajectory": self.refine_trajectory,
            "withheld": self.withheld,
            "errors": self.errors,
        }


_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str, fallback: str) -> str:
    s = _SLUG.sub("-", str(text).lower()).strip("-")
    return s or fallback


def _coerce_cites(raw, n: int, hallucinated: list[int], source_ids: list[str]) -> list[str]:
    """Resolve bracket-index cites -> source ids. Accepts a bare int/str too (a model
    that wrote "cites": 1 instead of [1]); out-of-range indices are flagged, not fatal."""
    if isinstance(raw, (int, str)):
        raw = [raw]
    elif not isinstance(raw, list):
        raw = []
    out: list[str] = []
    for idx in raw:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n:
            sid = source_ids[i - 1]
            if sid not in out:
                out.append(sid)
        else:
            hallucinated.append(i)
    return out


def validate_diagram_obj(obj, sources: list, *, mode: str = "prompt",
                         catalog=None, feasibility=None) -> DiagramSpec:
    """Validate a parsed diagram object and resolve citations against `sources`
    (each a dict with an "id"). Raises ValueError on an unusable shape so the caller
    can retry or fall back. Tolerant of missing/extra fields the way real models emit.

    `catalog` is an optional forward-compat seam: when a compiled closed vocabulary is
    passed, the validated spec is projected onto it (colpali_rag.studio.catalog.apply_catalog).
    Studio normally projects at the generate_diagram choke point instead (so the demo path is
    gated too), leaving this None. `feasibility` reserves a seam for connection-feasibility
    rules and is currently unused. With no catalog this behaves exactly as before."""
    if not isinstance(obj, dict):
        raise ValueError("diagram is not an object")
    source_ids = [s["id"] for s in sources]
    n = len(source_ids)
    hallucinated: list[int] = []

    groups = []
    seen_g = set()
    for g in obj.get("groups") or []:
        if not isinstance(g, dict):
            continue
        gid = _slug(g.get("id") or g.get("label") or "", f"g{len(groups)+1}")
        if gid in seen_g:
            continue
        seen_g.add(gid)
        groups.append(Group(id=gid, label=str(g.get("label", gid))))
    valid_groups = {g.id for g in groups}

    blocks = []
    seen_b = set()
    for b in obj.get("blocks") or []:
        if not isinstance(b, dict):
            continue
        bid = _slug(b.get("id") or b.get("label") or "", f"b{len(blocks)+1}")
        if bid in seen_b:
            continue
        seen_b.add(bid)
        kind = str(b.get("kind", "component")).lower()
        if kind not in BLOCK_KINDS:
            kind = "component"
        grp = _slug(b.get("group"), "") if b.get("group") else None
        if grp and grp not in valid_groups:
            grp = None
        blocks.append(Block(id=bid, label=str(b.get("label", bid)), kind=kind, group=grp,
                            cites=_coerce_cites(b.get("cites"), n, hallucinated, source_ids)))
    if not blocks:
        raise ValueError("no valid blocks")
    valid_blocks = {b.id for b in blocks}

    connections = []
    dropped = 0
    for c in obj.get("connections") or []:
        if not isinstance(c, dict):
            continue
        src = _slug(c.get("from") or c.get("source") or "", "")
        dst = _slug(c.get("to") or c.get("target") or "", "")
        if src not in valid_blocks or dst not in valid_blocks or src == dst:
            dropped += 1                      # edge to a block that doesn't exist -> drop it
            continue
        kind = str(c.get("kind", "data")).lower()
        if kind not in EDGE_KINDS:
            kind = "data"
        connections.append(Connection(source=src, target=dst, label=str(c.get("label", "")),
                                       kind=kind,
                                       cites=_coerce_cites(c.get("cites"), n, hallucinated, source_ids)))

    spec = DiagramSpec(
        title=str(obj.get("title", "Untitled diagram")),
        blocks=blocks, connections=connections, groups=groups,
        reasoning=str(obj.get("reasoning", "")),
        assumptions=[str(a) for a in (obj.get("assumptions") or []) if a],
        structured=True, mode=mode,
        hallucinated_citations=sorted(set(hallucinated)),
        dropped_connections=dropped,
    )
    if catalog is not None:
        from colpali_rag.studio.catalog import apply_catalog  # local import: no import cycle
        spec = apply_catalog(spec, catalog)
    return spec


def parse_diagram(text: str):
    """Parse a model response into a dict (json.loads -> brace-extract -> repair)."""
    return parse_json(text)
