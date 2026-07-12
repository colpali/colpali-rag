import { useRef, useState, type ReactNode } from "react";
import type { DocInfo, StudioStatus } from "../types";
import type { SessionState } from "../api";

export function SourcesPanel({
  status,
  docs,
  indexNote,
  selected,
  onToggle,
  onSelectAll,
  session,
  onUpload,
  uploading,
}: {
  status: StudioStatus | null;
  docs: DocInfo[];
  indexNote?: string;
  selected: string[];
  onToggle: (doc: string) => void;
  onSelectAll: (all: boolean) => void;
  session: SessionState | null;
  onUpload: (file: File) => void;
  uploading: boolean;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const allOn = selected.length === 0; // empty selection => all datasheets apply

  return (
    <aside className="flex h-full w-[300px] shrink-0 flex-col border-r border-ink-600 bg-ink-800/40">
      <Header />

      {status && (
        <div className="flex items-center gap-2 px-4 pb-3 text-xs text-slate-400">
          <span
            className={`h-2 w-2 rounded-full ${status.mode === "llm" ? "bg-emerald-400" : "bg-sky-400"}`}
          />
          {status.mode === "llm" ? "model connected" : "demo mode"}
          <span className="text-slate-600">·</span>
          {status.pages} page{status.pages === 1 ? "" : "s"} indexed
        </div>
      )}

      <Section
        title="Datasheets"
        right={
          docs.length > 0 && (
            <button
              onClick={() => onSelectAll(!allOn)}
              className="font-mono text-[10px] uppercase tracking-wide text-brand hover:underline"
            >
              {allOn ? "pick" : "all"}
            </button>
          )
        }
      >
        {docs.length === 0 ? (
          <p className="px-1 text-xs leading-relaxed text-slate-500">
            {indexNote || "No datasheets indexed yet."}
            <br />
            <code className="mt-1 inline-block rounded bg-ink-900 px-1.5 py-0.5 text-[11px] text-sky-300">
              colpali-rag index ./pdfs
            </code>
          </p>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {docs.map((d) => {
              const on = allOn || selected.includes(d.doc);
              return (
                <li key={d.doc}>
                  <button
                    onClick={() => onToggle(d.doc)}
                    className="group flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition hover:bg-ink-700/60"
                  >
                    <span
                      className={`grid h-4 w-4 shrink-0 place-items-center rounded border text-[10px] ${
                        on
                          ? "border-brand bg-brand/20 text-brand"
                          : "border-ink-600 text-transparent"
                      }`}
                    >
                      ✓
                    </span>
                    <span className="min-w-0 flex-1 truncate text-[13px] text-slate-200">
                      {d.doc}
                    </span>
                    <span className="font-mono text-[10px] text-slate-500">{d.pages}p</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </Section>

      <Section title="Your uploads">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            const f = e.dataTransfer.files?.[0];
            if (f) onUpload(f);
          }}
          onClick={() => fileRef.current?.click()}
          className={`cursor-pointer rounded-xl border border-dashed px-3 py-4 text-center text-xs transition ${
            drag ? "border-brand bg-brand/10 text-brand" : "border-ink-600 text-slate-500 hover:border-slate-500"
          }`}
        >
          {uploading ? "uploading…" : (
            <>
              <span className="text-slate-300">Drop CSV · Excel · notes</span>
              <br />
              or click to browse
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            hidden
            accept=".csv,.tsv,.xlsx,.xlsm,.txt,.md,.json"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
              e.target.value = "";
            }}
          />
        </div>

        <div className="mt-2 flex flex-col gap-1">
          {session?.tables.map((t) => (
            <Chip key={t.name} icon="▦" title={`${t.rows} rows × ${t.columns.length} cols`}>
              {t.name}
            </Chip>
          ))}
          {session?.notes.map((n) => (
            <Chip key={n.name} icon="✎" title={`${n.chars} chars`}>
              {n.name}
            </Chip>
          ))}
        </div>
      </Section>

      <div className="mt-auto px-4 py-3 text-[10px] leading-relaxed text-slate-600">
        Blocks cite the page or row they came from. Selected datasheets scope retrieval;
        empty selection uses all.
      </div>
    </aside>
  );
}

function Header() {
  return (
    <div className="flex items-center gap-2.5 px-4 pb-2 pt-4">
      <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-brand/30 to-brand-2/30 text-brand">
        ◈
      </div>
      <div>
        <div className="text-sm font-semibold leading-tight text-slate-100">Studio</div>
        <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-slate-500">
          ColPali · grounded
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  right,
  children,
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="border-t border-ink-600/70 px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-mono text-[10px] uppercase tracking-[0.16em] text-slate-500">
          {title}
        </h3>
        {right}
      </div>
      {children}
    </div>
  );
}

function Chip({
  children,
  icon,
  title,
}: {
  children: ReactNode;
  icon: string;
  title?: string;
}) {
  return (
    <div
      title={title}
      className="flex items-center gap-2 rounded-lg bg-ink-700/50 px-2.5 py-1.5 text-xs text-slate-300"
    >
      <span className="text-slate-500">{icon}</span>
      <span className="min-w-0 flex-1 truncate">{children}</span>
    </div>
  );
}
