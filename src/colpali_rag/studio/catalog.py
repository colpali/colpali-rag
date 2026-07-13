"""Closed-vocabulary constraint for structured outputs — compile, match, project.

The studio can emit a node/connection graph whose nodes are free strings. When an operator
uploads a table and names its id column at runtime (`CATALOG_ID_COL`), we compile that table
into a closed vocabulary `V` and *project* every proposed node onto it: a confident match is
rewritten to its canonical name, anything else is dropped and flagged. That is what makes a
"only from the catalog" guarantee true by construction rather than by trusting the model.

Three deliberate properties:

* **The whole table is the constraint.** We read the parsed `Table` in full (every row, every
  column). Display caps live elsewhere and can never reach here, so a large upload is never
  silently clipped out of the vocabulary.
* **Ids are preserved, not slugged.** Matching uses a conservative `canon` that casefolds and
  strips *surrounding* punctuation but keeps internal `- . / _ +`, so `AX-12.34` and `AX 12 34`
  stay distinct. (Contrast the renderer's `_slug`, which collapses them to one key.)
* **Nothing is corrected silently.** Every rewrite is recorded (`remapped_parts` + an
  assumption), every drop is recorded (`hallucinated_parts`), and the whole output can abstain.

Pure-Python, zero third-party deps, and it imports nothing from the rest of the studio — it
operates on duck-typed `Table`/`DiagramSpec`/`Settings`, so there is no import cycle. Every name
and value is injected at runtime; this module hard-codes no schema.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Wrapping punctuation stripped from the *edges* of a match key. Deliberately excludes the
# id characters - . / _ + so a leading/trailing one (".net", "5.") is preserved and distinct
# ids never collapse; a genuine trailing separator variant is still caught by the fuzzy tier.
_EDGE = " \t\r\n\"'`()[]{}<>,;:!?"
# ID-aware tokenizer: a token may contain internal - . / _ + so "ax-1234" stays ONE token.
_ID_TOKEN = re.compile(r"[a-z0-9]+(?:[-./_+][a-z0-9]+)*")
# Sub-token splitter for the fuzzy overlap channel: maximal alphanumeric runs only, so that
# "ax-100" and "ax 100" both yield {ax, 100} and a hyphen-vs-space variant can still match.
_SUBTOKEN = re.compile(r"[a-z0-9]+")

_FALSEY = {"", "0", "false", "no", "n", "f", "none", "null", "na", "n/a"}


def canon(s) -> str:
    """Match key: NFKC + casefold + collapse whitespace + strip *wrapping* punctuation.
    PRESERVES the id characters - . / _ + everywhere (interior and edges) so distinct ids
    stay distinct. NFKC + casefold intentionally unify typographic variants (full-width vs
    ASCII digits, etc.); on the rare occasion that folds two verbatim ids together,
    build_catalog records it as a surfaced conflict rather than merging silently."""
    s = unicodedata.normalize("NFKC", "" if s is None else str(s)).casefold()
    s = " ".join(s.split())
    return s.strip(_EDGE)


def id_tokens(s) -> set[str]:
    """ID-aware token set (atomic): keeps 'ax-1234' as one token. Used for the prefilter
    seam and the "ids are preserved" contract — NOT for the fuzzy overlap score."""
    return set(_ID_TOKEN.findall(canon(s)))


def _subtokens(s) -> set[str]:
    """Sub-token set for the fuzzy overlap channel: splits on id punctuation too, so a
    separator variant ('ax 100' vs 'ax-100') still overlaps."""
    return set(_SUBTOKEN.findall(canon(s)))


def _lev_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0,1] (two-row Wagner-Fischer, pure Python)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


@dataclass
class MatchResult:
    status: str                  # "exact" | "remap" | "unresolved"
    canonical: str | None        # verbatim catalog id to use as the node label, if resolved
    score: float


@dataclass
class Catalog:
    keys: dict[str, str]                                      # canon(id|alias) -> verbatim id
    canonical: set[str]                                       # V: the set of verbatim ids
    required: set[str] = field(default_factory=set)          # Req: ids that must appear
    tok_index: dict[str, set] = field(default_factory=dict)  # id -> id_tokens (prefilter seam)
    conflicts: list = field(default_factory=list)            # (key, [ids...]) surfaced, first kept
    iface_cols: list[str] = field(default_factory=list)      # interface columns actually used
    interfaces: dict[str, set] = field(default_factory=dict) # id -> interface tokens (edge rules)
    gate: str = "off"
    tau: float = 0.84
    delta: float = 0.08
    w_lev: float = 0.6
    w_jac: float = 0.4
    withhold_max_drop: float = 0.5

    def match(self, label) -> MatchResult:
        k = canon(label)
        if not k:
            return MatchResult("unresolved", None, 0.0)
        if k in self.keys:                                   # exact hash hit: decisive, no margin
            return MatchResult("exact", self.keys[k], 1.0)
        # near-exact: the model appended/prepended id punctuation ("R1." for id "R1"). Strip it
        # from the LABEL only and retry the key lookup. This restores short-id recall without the
        # build-time merge that stripping in canon would cause — distinct ids stay distinct keys,
        # and the literal-canon exact above always wins first, so an id that really ends in '.'
        # is unaffected. Surfaced as a remap (the surface differed), never a silent exact.
        stripped = k.strip("-./_+")
        if stripped != k and stripped in self.keys:
            return MatchResult("remap", self.keys[stripped], 0.99)
        sub = _subtokens(label)
        best: dict[str, float] = {}                          # best score per distinct id
        for ck, disp in self.keys.items():
            lev = _lev_sim(k, ck)
            kt = set(_SUBTOKEN.findall(ck))
            jac = (len(sub & kt) / len(sub | kt)) if (sub or kt) else 0.0
            s = self.w_lev * lev + self.w_jac * jac
            if s > best.get(disp, -1.0):
                best[disp] = s
        if not best:
            return MatchResult("unresolved", None, 0.0)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        disp1, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        if s1 >= self.tau and (s1 - s2) >= self.delta:
            return MatchResult("remap", disp1, s1)
        return MatchResult("unresolved", None, s1)

    def accept(self, label) -> bool:
        """True when a raw label resolves to the vocabulary — used to score raw model output."""
        return self.match(label).status in ("exact", "remap")

    def feasible(self, a_id, b_id):
        """Is a connection between two catalog ids permitted? Compatibility is a shared
        interface token (both items 'speak' a common bus/connector/type, from the configured
        interface columns). Returns None when either side has no interface data — 'can't judge',
        so the edge is never deleted on missing info; True/False only when both are known."""
        ia = self.interfaces.get(a_id)
        ib = self.interfaces.get(b_id)
        if not ia or not ib:
            return None
        return bool(ia & ib)


def _split(s) -> list[str]:
    return [p.strip() for p in str(s or "").split(",") if p.strip()]


def _col_index(columns, name: str):
    """Header lookup by canonical equality; returns the first matching index or None."""
    want = canon(name)
    if not want:
        return None
    hits = [i for i, c in enumerate(columns) if canon(c) == want]
    if len(hits) > 1:
        log.warning("catalog: column %r matches %d headers; using the first", name, len(hits))
    return hits[0] if hits else None


def _truthy(v) -> bool:
    return canon(v) not in _FALSEY


_IFACE_STRIP = re.compile(r"[\s\-./_+]")


def _iface_tokens(cell) -> set[str]:
    """Interface tokens for edge compatibility — one per list item (separated by , ; |).
    Normalized separator-insensitively so 'USB-C' == 'USB C' == 'USB_C' == 'USBC'. Placeholder
    or empty values (N/A, none, -, ...) are dropped, so two parts with only 'unknown' interfaces
    do not look mutually compatible (feasible() then correctly reports 'no interface data')."""
    out = set()
    for p in re.split(r"[,;|]", str(cell or "")):
        c = canon(p)
        if not c or c in _FALSEY:
            continue
        norm = _IFACE_STRIP.sub("", c)
        if norm:
            out.add(norm)
    return out


def build_catalog(tables, settings) -> Catalog | None:
    """Compile every uploaded table that carries the configured id column into a closed
    vocabulary. Returns None (feature OFF) when no id column is configured, no uploaded table
    has it, or the compiled vocabulary is empty — so an unset/misparsed catalog degrades to
    today's unconstrained behavior rather than nuking every output."""
    id_col = (getattr(settings, "catalog_id_col", "") or "").strip()
    if not id_col:
        return None
    name_cols = _split(getattr(settings, "catalog_name_cols", ""))
    req_col = (getattr(settings, "catalog_required_col", "") or "").strip()
    iface = _split(getattr(settings, "catalog_iface_cols", ""))

    keys: dict[str, str] = {}
    canonical: set[str] = set()
    required: set[str] = set()
    tok_index: dict[str, set] = {}
    interfaces: dict[str, set] = {}
    conflicts: list = []
    found = False
    rows_seen = 0

    for t in tables or []:
        idx = _col_index(t.columns, id_col)
        if idx is None:
            continue
        found = True
        name_idx = [j for j in (_col_index(t.columns, c) for c in name_cols) if j is not None]
        req_idx = _col_index(t.columns, req_col) if req_col else None
        iface_idx = [j for j in (_col_index(t.columns, c) for c in iface) if j is not None]
        for row in t.rows:                                   # FULL rows — the constraint channel
            rows_seen += 1
            disp = (row[idx] if idx < len(row) else "").strip()
            if not disp:
                continue
            canonical.add(disp)
            tok_index.setdefault(disp, id_tokens(disp))
            aliases = [disp] + [row[j] for j in name_idx if j < len(row)]
            for a in aliases:
                ck = canon(a)
                if not ck:
                    continue
                if ck in keys and keys[ck] != disp:
                    conflicts.append((ck, [keys[ck], disp]))  # ambiguous: keep first, surface it
                else:
                    keys.setdefault(ck, disp)
            if req_idx is not None and req_idx < len(row) and _truthy(row[req_idx]):
                required.add(disp)
            if iface_idx:
                toks = set()
                for j in iface_idx:
                    if j < len(row):
                        toks |= _iface_tokens(row[j])
                if toks:
                    interfaces.setdefault(disp, set()).update(toks)

    if not found:
        log.warning("catalog: id column %r is configured but no uploaded table carries it; "
                    "the constraint is INACTIVE and outputs are unconstrained", id_col)
        return None
    if not canonical:
        log.warning("catalog: id column %r found but yielded no ids; constraint INACTIVE", id_col)
        return None
    if conflicts:
        log.warning("catalog: %d ambiguous key(s) — kept first occurrence", len(conflicts))
    log.info("catalog: compiled %d id(s) from %d row(s), %d required, %d alias key(s)",
             len(canonical), rows_seen, len(required), len(keys))
    return Catalog(
        keys=keys, canonical=canonical, required=required, tok_index=tok_index,
        conflicts=conflicts, iface_cols=iface, interfaces=interfaces,
        gate=getattr(settings, "catalog_gate", "off"),
        tau=float(getattr(settings, "catalog_match_threshold", 0.84)),
        delta=float(getattr(settings, "catalog_match_margin", 0.08)),
        withhold_max_drop=float(getattr(settings, "catalog_withhold_max_drop", 0.5)),
    )


def project(spec, catalog):
    """Project a DiagramSpec onto the catalog and check the graph rules — node membership,
    connection feasibility, required-item completeness. Records every violation on `spec` but
    does NOT withhold (the terminal gate decides that, after any repair attempts). Returns a
    violation summary the repair loop and gate consume. Duck-typed on `spec`; mutates in place.

    Delta audit fields (hallucinated_parts/remapped_parts/dropped_blocks/dropped_connections)
    accumulate (+=), so re-projecting an already-clean spec is a no-op. Current-state fields
    (infeasible_connections/missing_required) are recomputed (=), so they never double-count."""
    n_raw = len(spec.blocks)
    drops: list[str] = []
    remaps: list[dict] = []
    kept_blocks = []
    surviving_ids: set[str] = set()
    id2canon: dict[str, str] = {}

    for b in spec.blocks:
        raw = b.label
        m = catalog.match(raw)
        if m.status == "unresolved":
            drops.append(raw)
            continue
        if m.canonical is not None and m.canonical != b.label:
            remaps.append({"from": raw, "to": m.canonical, "score": round(m.score, 3),
                           "block": b.id})
            b.label = m.canonical                            # id stays stable; edges keep resolving
        surviving_ids.add(b.id)
        id2canon[b.id] = b.label
        kept_blocks.append(b)

    dropped_edges = 0
    kept_conns = []
    for c in spec.connections:
        if c.source in surviving_ids and c.target in surviving_ids:
            kept_conns.append(c)
        else:
            dropped_edges += 1

    # connection feasibility: FLAG (never drop) edges the catalog says can't connect, so a real
    # edge is never deleted when interface data is incomplete (feasible() returns None then).
    infeasible_edges = [(id2canon[c.source], id2canon[c.target]) for c in kept_conns
                        if catalog.feasible(id2canon.get(c.source), id2canon.get(c.target)) is False]
    # required-item completeness
    missing = sorted(catalog.required - {b.label for b in kept_blocks})

    spec.blocks = kept_blocks
    spec.connections = kept_conns
    spec.hallucinated_parts += drops
    spec.remapped_parts += remaps
    spec.dropped_blocks += len(drops)
    spec.dropped_connections += dropped_edges
    spec.infeasible_connections = len(infeasible_edges)     # current-state (=), not a delta
    spec.missing_required = missing                         # current-state (=)

    for r in remaps:
        spec.assumptions.append(f"Mapped '{r['from']}' -> '{r['to']}' (match {r['score']}).")
    if drops:
        spec.assumptions.append(
            f"Dropped {len(drops)} node(s) not in the catalog: {', '.join(drops)}.")
    if catalog.conflicts and not any("ambiguous key" in e for e in spec.errors):
        spec.errors.append(f"catalog has {len(catalog.conflicts)} ambiguous key(s)")

    return {
        "n_raw": n_raw,
        "dropped": len(drops),
        "drop_frac": (len(drops) / n_raw) if n_raw else 1.0,
        "hallucinated": list(drops),
        "infeasible_edges": infeasible_edges,
        "missing": list(missing),
    }


def violation_total(viol) -> int:
    """How many catalog violations remain — drives the repair loop and the terminal gate."""
    return viol["dropped"] + len(viol["infeasible_edges"]) + len(viol["missing"])


def apply_terminal_gate(spec, catalog, viol):
    """Terminal decision after projection (and any repair attempts). Under gate='withhold',
    abstain when the emitted diagram is still invalid: nothing grounded, too much dropped, a
    required item missing, or an infeasible connection remaining. flag/off never abstain."""
    if catalog is None or getattr(catalog, "gate", "off") != "withhold" or getattr(spec, "withheld", False):
        return spec
    degenerate = (not spec.blocks) or (viol["n_raw"] and viol["drop_frac"] >= catalog.withhold_max_drop)
    hard = bool(spec.missing_required) or spec.infeasible_connections > 0
    if degenerate or hard:
        spec.withheld = True
        spec.blocks = []
        spec.connections = []
        spec.assumptions.append(
            "Withheld: the request could not be satisfied using only the catalog.")
    return spec


def apply_catalog(spec, catalog):
    """Single-shot projection + terminal gate. A None catalog or gate='off' is a byte-identical
    no-op. generate_diagram calls project()/apply_terminal_gate() directly so it can repair
    between them; this wrapper keeps the one-call behavior for the validate_diagram_obj seam.
    An already-withheld spec is returned untouched (re-projecting an emptied diagram would
    falsely report its required items as missing)."""
    if catalog is None or getattr(catalog, "gate", "off") == "off" or getattr(spec, "withheld", False):
        return spec
    viol = project(spec, catalog)
    return apply_terminal_gate(spec, catalog, viol)
