"""Turn an uploaded CSV / Excel file into grounded context for diagram generation.

CSV uses the stdlib (zero deps). Excel (.xlsx) uses openpyxl only if it's installed
(`pip install -e '.[studio]'`); without it we raise a clear, actionable error rather
than a mystery ImportError. The output is deliberately small and text-first: a header
list, row count, and a truncated preview the LLM can read and cite ("bom.xlsx row 4"),
not the whole sheet — a datasheet corpus is the visual channel; tables are the
structured supplement.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

MAX_PREVIEW_ROWS = 40
MAX_COLS = 24
MAX_CELL = 80


@dataclass
class Table:
    name: str                        # file name / logical id
    columns: list[str]
    rows: list[list[str]]            # preview rows (truncated)
    total_rows: int
    sheet: str | None = None
    note: str = ""

    def summary(self) -> str:
        """A compact, citable text rendering the LLM sees as one source."""
        head = " | ".join(self.columns) if self.columns else "(no header)"
        lines = [f"Table {self.name}" + (f" [sheet {self.sheet}]" if self.sheet else "")
                 + f" — {self.total_rows} row(s), {len(self.columns)} column(s)",
                 head, "-" * min(len(head), 80)]
        for i, r in enumerate(self.rows, start=1):
            lines.append(f"{i:>3}: " + " | ".join(r))
        if self.total_rows > len(self.rows):
            lines.append(f"… ({self.total_rows - len(self.rows)} more row(s))")
        return "\n".join(lines)


def _clip(v) -> str:
    s = "" if v is None else str(v)
    s = " ".join(s.split())
    return s[:MAX_CELL] + ("…" if len(s) > MAX_CELL else "")


def _from_matrix(name: str, matrix: list[list], sheet: str | None = None) -> Table:
    matrix = [row for row in matrix if any(c not in (None, "") for c in row)]
    if not matrix:
        return Table(name=name, columns=[], rows=[], total_rows=0, sheet=sheet, note="empty")
    header = [_clip(c) for c in matrix[0][:MAX_COLS]]
    body = matrix[1:]
    rows = [[_clip(c) for c in row[:MAX_COLS]] for row in body[:MAX_PREVIEW_ROWS]]
    return Table(name=name, columns=header, rows=rows, total_rows=len(body), sheet=sheet)


def load_csv(name: str, data: bytes) -> Table:
    text = data.decode("utf-8-sig", errors="replace")
    # sniff the delimiter; default to comma if the sniffer is unsure
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    matrix = list(csv.reader(io.StringIO(text), dialect))
    return _from_matrix(name, matrix)


def load_xlsx(name: str, data: bytes) -> Table:
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "reading .xlsx needs openpyxl — install with: pip install -e '.[studio]' "
            "(or convert the sheet to .csv)"
        ) from e
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    matrix = [list(row) for row in ws.iter_rows(values_only=True)]
    return _from_matrix(name, matrix, sheet=ws.title)


def load_table(name: str, data: bytes) -> Table:
    """Dispatch on extension. Unknown text extensions are treated as CSV."""
    low = name.lower()
    if low.endswith((".xlsx", ".xlsm")):
        return load_xlsx(name, data)
    if low.endswith(".xls"):
        raise RuntimeError("legacy .xls isn't supported — save as .xlsx or .csv")
    return load_csv(name, data)
