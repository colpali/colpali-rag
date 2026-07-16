import { useRef, useState, type ReactNode } from "react";
import { Button, Callout, Checkbox, Icon, Tag } from "@blueprintjs/core";
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
    <aside className="flex h-full w-[300px] shrink-0 flex-col border-r border-ink-600 bg-ink-900/60">
      <Header />

      {status && (
        <div className="flex items-center gap-2 px-4 pb-3 text-xs text-slate-400">
          <span
            className={`h-2 w-2 rounded-full ${status.mode === "llm" ? "bg-emerald-400" : "bg-brand"}`}
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
            <Button
              variant="minimal"
              small
              onClick={() => onSelectAll(!allOn)}
              className="!font-mono !text-[10px] !uppercase !tracking-wide"
            >
              {allOn ? "pick" : "all"}
            </Button>
          )
        }
      >
        {docs.length === 0 ? (
          <Callout intent="primary" icon="info-sign" compact className="!bg-ink-800/70 text-xs">
            {indexNote || "No datasheets indexed yet."}
            <code className="mt-1.5 block w-fit rounded bg-ink-950 px-1.5 py-0.5 text-[11px] text-brand-2">
              colpali-rag index ./pdfs
            </code>
          </Callout>
        ) : (
          <ul className="flex flex-col">
            {docs.map((d) => {
              const on = allOn || selected.includes(d.doc);
              return (
                <li key={d.doc}>
                  <Checkbox
                    checked={on}
                    onChange={() => onToggle(d.doc)}
                    className="!mb-0 rounded px-1 py-0.5 hover:bg-ink-700/50"
                    labelElement={
                      <span className="ml-1 inline-flex w-[212px] items-center justify-between gap-2 align-middle">
                        <span className="truncate text-[13px] text-slate-200" title={d.doc}>
                          {d.doc}
                        </span>
                        <Tag minimal round className="shrink-0 !font-mono !text-[10px]">
                          {d.pages}p
                        </Tag>
                      </span>
                    }
                  />
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
          className={`flex cursor-pointer flex-col items-center gap-1 rounded-lg border border-dashed px-3 py-4 text-center text-xs transition ${
            drag ? "border-brand bg-brand/10 text-brand" : "border-ink-600 text-slate-500 hover:border-brand/60"
          }`}
        >
          <Icon icon="upload" size={16} className={drag ? "text-brand" : "text-slate-500"} />
          {uploading ? (
            "uploading…"
          ) : (
            <>
              <span className="text-slate-300">Drop CSV · Excel · notes</span>
              <span className="text-slate-500">or click to browse</span>
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
            <UploadChip key={t.name} icon="th" title={`${t.rows} rows × ${t.columns.length} cols`}>
              {t.name}
            </UploadChip>
          ))}
          {session?.notes.map((n) => (
            <UploadChip key={n.name} icon="annotation" title={`${n.chars} chars`}>
              {n.name}
            </UploadChip>
          ))}
        </div>
      </Section>

      <div className="mt-auto px-4 py-3 text-[10px] leading-relaxed text-slate-500">
        Blocks cite the page or row they came from. Selected datasheets scope retrieval;
        empty selection uses all.
      </div>
    </aside>
  );
}

function Header() {
  return (
    <div className="flex items-center gap-2.5 px-4 pb-2 pt-4">
      <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-brand/30 to-brand-2/25 text-brand">
        <Icon icon="diagram-tree" size={16} />
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
    <div className="border-t border-ink-600/60 px-4 py-3">
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

function UploadChip({
  children,
  icon,
  title,
}: {
  children: ReactNode;
  icon: "th" | "annotation";
  title?: string;
}) {
  return (
    <div
      title={title}
      className="flex items-center gap-2 rounded-lg bg-ink-700/50 px-2.5 py-1.5 text-xs text-slate-300"
    >
      <Icon icon={icon} size={12} className="text-slate-500" />
      <span className="min-w-0 flex-1 truncate">{children}</span>
    </div>
  );
}
