"""In-memory studio sessions. A session holds what the user has chosen and uploaded:
which indexed datasheets to apply, any CSV/Excel tables, free-text notes, and the
chat/output history. Process-local and ephemeral by design — the durable corpus lives
in the engine's store; a session is just the working set for one conversation.

Kept intentionally simple (a dict + a lock). Swap in a shared store later if you need
multi-process; nothing else depends on this being in-memory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from colpali_rag.studio.tabular import Table


@dataclass
class Note:
    name: str
    text: str

    def summary(self) -> str:
        body = self.text if len(self.text) <= 1200 else self.text[:1200] + "\n… (truncated)"
        return f"Note {self.name}:\n{body}"


@dataclass
class Turn:
    request: str
    spec: dict | None = None         # DiagramSpec.to_dict(sources)


@dataclass
class Session:
    id: str
    selected_docs: set[str] = field(default_factory=set)   # doc ids to apply; empty => all
    tables: list[Table] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    history: list[Turn] = field(default_factory=list)
    last_spec: object | None = None      # the most recent DiagramSpec object (for export)
    last_sources: list = field(default_factory=list)

    def add_upload(self, name: str, data: bytes) -> str:
        """Ingest an uploaded file into the session. Returns a short human status."""
        low = name.lower()
        if low.endswith((".csv", ".tsv", ".xlsx", ".xlsm", ".xls")):
            from colpali_rag.studio.tabular import load_table

            t = load_table(name, data)
            self.tables = [x for x in self.tables if x.name != name] + [t]
            return f"{name}: {t.total_rows} row(s), {len(t.columns)} column(s)"
        # everything else -> a text note (markdown, txt, or unknown)
        text = data.decode("utf-8", errors="replace")
        self.notes = [x for x in self.notes if x.name != name] + [Note(name, text)]
        return f"{name}: {len(text)} chars of notes"


class SessionStore:
    def __init__(self):
        self._d: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._n = 0

    def create(self) -> Session:
        with self._lock:
            self._n += 1
            sid = f"s{self._n:04d}"
            s = Session(id=sid)
            self._d[sid] = s
            return s

    def get(self, sid: str) -> Session | None:
        return self._d.get(sid)

    def get_or_create(self, sid: str | None) -> Session:
        if sid and sid in self._d:
            return self._d[sid]
        return self.create()
