"""Turn an uploaded CSV / Excel file into grounded context for structured generation.

CSV uses the stdlib (zero deps). Excel (.xlsx) uses openpyxl only if it's installed
(`pip install -e '.[studio]'`); without it we raise a clear, actionable error rather
than a mystery ImportError.

The parsed `Table` retains the **full** sheet — every row, every column, uncapped
cells. That full copy is the *constraint channel*: a closed-vocabulary compiler reads it
whole, so a large uploaded table can never be silently clipped away. The model-facing
`summary()` is a separate *display channel* that applies size caps (config-driven, passed
in) so a huge sheet doesn't blow up the prompt — but those caps can never touch the stored
data. Memory is bounded by the caller's upload-size limit.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

# Display-preview defaults (mirrored by config's tabular_* knobs, which override at call
# time). These bound only summary(); the stored Table is always complete.
MAX_PREVIEW_ROWS = 40
MAX_COLS = 24
MAX_CELL = 80


@dataclass
class Table:
    name: str                        # file name / logical id
    columns: list[str]               # full header (all columns, uncapped)
    rows: list[list[str]]            # full body (all rows, all columns, uncapped cells)
    total_rows: int
    sheet: str | None = None
    note: str = ""
    row_numbers: list[int] = field(default_factory=list)   # 1-based SOURCE row of each stored row

    def summary(self, *, max_rows: int = MAX_PREVIEW_ROWS, max_cols: int = MAX_COLS,
                max_cell: int = MAX_CELL) -> str:
        """A compact, citable text rendering the model sees as one source. The stored table is
        complete; these caps bound only this preview, never the constraint channel. Each row is
        labeled with its real source-file row number, so a model can cite an exact row."""
        cols = [_clip(c, max_cell) for c in self.columns[:max_cols]]
        head = " | ".join(cols) if cols else "(no header)"
        note = "  (leading number = source-file row)" if self.row_numbers else ""
        lines = [f"Table {self.name}" + (f" [sheet {self.sheet}]" if self.sheet else "")
                 + f" — {self.total_rows} row(s), {len(self.columns)} column(s)" + note,
                 head, "-" * min(len(head), 80)]
        shown = self.rows[:max_rows]
        for i, r in enumerate(shown):
            rn = self.row_numbers[i] if i < len(self.row_numbers) else i + 1
            lines.append(f"{rn:>4}: " + " | ".join(_clip(c, max_cell) for c in r[:max_cols]))
        if self.total_rows > len(shown):
            lines.append(f"… ({self.total_rows - len(shown)} more row(s))")
        return "\n".join(lines)

    def source_row(self, i: int) -> int | None:
        """The 1-based source-file row number of stored row `i` (0-based), or None if unknown."""
        return self.row_numbers[i] if 0 <= i < len(self.row_numbers) else None


def _norm(v) -> str:
    """Storage form: stringify + collapse internal whitespace. No length cap — the full
    value is retained so the constraint channel sees every character."""
    return "" if v is None else " ".join(str(v).split())


def _clip(v, max_cell: int = MAX_CELL) -> str:
    """Display form: normalize, then truncate for the model-facing preview only."""
    s = _norm(v)
    return s[:max_cell] + ("…" if len(s) > max_cell else "")


def _from_matrix(name: str, matrix: list[list], sheet: str | None = None) -> Table:
    # keep each non-empty row's ORIGINAL 1-based position, so citations point at the real source row
    numbered = [(i, row) for i, row in enumerate(matrix, start=1)
                if any(c not in (None, "") for c in row)]
    if not numbered:
        return Table(name=name, columns=[], rows=[], total_rows=0, sheet=sheet, note="empty")
    header = [_norm(c) for c in numbered[0][1]]            # full width, uncapped
    body = numbered[1:]
    rows = [[_norm(c) for c in row] for _, row in body]    # full rows, full cells, uncapped
    return Table(name=name, columns=header, rows=rows, total_rows=len(body),
                 row_numbers=[n for n, _ in body], sheet=sheet)


def load_csv(name: str, data: bytes) -> Table:
    text = data.decode("utf-8-sig", errors="replace")
    # sniff the delimiter; default to comma if the sniffer is unsure
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    matrix = list(csv.reader(io.StringIO(text), dialect))
    return _from_matrix(name, matrix)


def _read_xlsx(data: bytes):
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "reading .xlsx needs openpyxl — install with: pip install -e '.[studio]' "
            "(or convert the sheet to .csv)"
        ) from e
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    return [(ws.title, [list(row) for row in ws.iter_rows(values_only=True)])
            for ws in wb.worksheets]


def load_xlsx_sheets(name: str, data: bytes) -> list[Table]:
    """Every non-empty worksheet as its own Table. A multi-sheet workbook must contribute
    ALL of its sheets to the constraint channel — the vocabulary compiler scans every table —
    so we never silently drop the sheet the catalog happens to live on."""
    tables = [_from_matrix(name, matrix, sheet=title) for title, matrix in _read_xlsx(data)]
    tables = [t for t in tables if t.total_rows or t.columns]
    return tables or [Table(name=name, columns=[], rows=[], total_rows=0, note="empty")]


def load_xlsx(name: str, data: bytes) -> Table:
    """First non-empty worksheet as a single Table (single-table dispatch)."""
    return load_xlsx_sheets(name, data)[0]


def load_table(name: str, data: bytes) -> Table:
    """Dispatch on extension, single Table. Unknown text extensions are treated as CSV.
    For a workbook this is the first sheet only — use load_tables to keep every sheet."""
    low = name.lower()
    if low.endswith((".xlsx", ".xlsm")):
        return load_xlsx(name, data)
    if low.endswith(".xls"):
        raise RuntimeError("legacy .xls isn't supported — save as .xlsx or .csv")
    return load_csv(name, data)


def load_tables(name: str, data: bytes) -> list[Table]:
    """All tables in an upload: every sheet of a workbook, or a single CSV. This is the
    ingestion path — it keeps the whole workbook so the constraint channel is complete."""
    low = name.lower()
    if low.endswith((".xlsx", ".xlsm")):
        return load_xlsx_sheets(name, data)
    if low.endswith(".xls"):
        raise RuntimeError("legacy .xls isn't supported — save as .xlsx or .csv")
    return [load_csv(name, data)]


def plan_table(table: "Table", request: str, *, max_rows: int = MAX_PREVIEW_ROWS,
               max_cols: int = MAX_COLS, max_cell: int = MAX_CELL, rank: bool = True):
    """Request-aware model-facing rendering of one table. Returns (text, shown_source_rows).

    With rank=True, a table LARGER than the row budget is ranked by lexical relevance to the
    request and the most relevant rows are surfaced (labeled by source-file row), with an explicit
    omission audit — so a huge multi-sheet workbook puts the RELEVANT rows in front of the model
    instead of just the first N. A table that fits (or rank=False, or a blank request) is shown in
    order. The full table always reaches the constraint channel (the catalog compiler reads
    table.rows untouched); this only shapes the prompt view. `shown_source_rows` is the list of
    1-based source-file row numbers actually put in front of the model (for the run log)."""
    if not rank or table.total_rows <= max_rows or not str(request or "").strip():
        text = table.summary(max_rows=max_rows, max_cols=max_cols, max_cell=max_cell)
        shown = [table.source_row(i) or (i + 1) for i in range(min(table.total_rows, max_rows))]
        return text, shown

    from colpali_rag.lexical import LexicalIndex

    idx = LexicalIndex([(str(i), " ".join(str(c) for c in row)) for i, row in enumerate(table.rows)])
    sel = [int(i) for i, _ in idx.search(request, top_k=max_rows)]   # relevance-ranked row indices
    if len(sel) < max_rows:                                          # top up with earliest rows
        seen = set(sel)
        sel += [i for i in range(table.total_rows) if i not in seen][:max_rows - len(sel)]
    sel = sel[:max_rows]

    cols = [_clip(c, max_cell) for c in table.columns[:max_cols]]
    head = " | ".join(cols) if cols else "(no header)"
    lines = [f"Table {table.name}" + (f" [sheet {table.sheet}]" if table.sheet else "")
             + f" — {table.total_rows} row(s), {len(table.columns)} column(s); showing the "
             f"{len(sel)} most relevant to the request ({table.total_rows - len(sel)} not shown). "
             "Leading number = source-file row.",
             head, "-" * min(len(head), 80)]
    shown = []
    for i in sel:
        rn = table.source_row(i) or (i + 1)
        shown.append(rn)
        lines.append(f"{rn:>4}: " + " | ".join(_clip(c, max_cell) for c in table.rows[i][:max_cols]))
    return "\n".join(lines), shown


def plan_table_text(table: "Table", request: str, *, max_rows: int = MAX_PREVIEW_ROWS,
                    max_cols: int = MAX_COLS, max_cell: int = MAX_CELL) -> str:
    """The model-facing text of plan_table() (which also returns the surfaced source rows)."""
    return plan_table(table, request, max_rows=max_rows, max_cols=max_cols, max_cell=max_cell)[0]
