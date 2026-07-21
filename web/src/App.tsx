import { useCallback, useEffect, useState } from "react";
import { Button, ButtonGroup, Icon, Tag } from "@blueprintjs/core";
import { SourcesPanel } from "./components/SourcesPanel";
import { ChatPanel, type Msg } from "./components/ChatPanel";
import { DiagramCanvas } from "./components/DiagramCanvas";
import { api, type SessionState } from "./api";
import type { DocInfo, Spec, StudioStatus } from "./types";

export default function App() {
  const [status, setStatus] = useState<StudioStatus | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [indexNote, setIndexNote] = useState<string | undefined>();
  const [selected, setSelected] = useState<string[]>([]); // [] => all datasheets
  const [session, setSession] = useState<SessionState | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [spec, setSpec] = useState<Spec | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSources, setShowSources] = useState(true);
  const [showChat, setShowChat] = useState(true);

  useEffect(() => {
    api.status().then(setStatus).catch(() => setStatus(null));
    api.newSession().then(setSessionId).catch((e) => setError(String(e.message ?? e)));
    api
      .sources()
      .then((d) => {
        setDocs(d.docs);
        setIndexNote(d.note);
      })
      .catch(() => {});
  }, []);

  const allIds = docs.map((d) => d.doc);
  const toggleDoc = useCallback(
    (doc: string) =>
      setSelected((prev) => {
        const base = prev.length === 0 ? allIds : prev;
        const next = base.includes(doc) ? base.filter((x) => x !== doc) : [...base, doc];
        return next.length === 0 ? [] : next;
      }),
    [allIds],
  );
  const selectAll = useCallback((all: boolean) => setSelected(all ? [] : allIds), [allIds]);

  const send = useCallback(
    async (text: string) => {
      if (!sessionId || loading) return;
      setMessages((m) => [...m, { role: "user", text }]);
      setLoading(true);
      setError(null);
      try {
        const res = await api.diagram(sessionId, text, selected);
        setSpec(res.spec);
        setMessages((m) => [
          ...m,
          { role: "assistant", text: res.spec.reasoning || "Here is the diagram.", spec: res.spec },
        ]);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setMessages((m) => [...m, { role: "assistant", text: msg }]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId, selected, loading],
  );

  const upload = useCallback(
    async (files: File[]) => {
      if (!sessionId || !files.length) return;
      setUploading(true);
      setError(null);
      try {
        const r = await api.upload(sessionId, files);
        setSession(r.session);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setUploading(false);
      }
    },
    [sessionId],
  );

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden">
      <TopBar
        status={status}
        showSources={showSources}
        showChat={showChat}
        onToggleSources={() => setShowSources((v) => !v)}
        onToggleChat={() => setShowChat((v) => !v)}
      />
      <div className="flex min-h-0 flex-1">
        {showSources && (
          <SourcesPanel
            status={status}
            docs={docs}
            indexNote={indexNote}
            selected={selected}
            onToggle={toggleDoc}
            onSelectAll={selectAll}
            session={session}
            onUpload={upload}
            uploading={uploading}
          />
        )}
        {showChat && (
          <ChatPanel messages={messages} onSend={send} loading={loading} mode={status?.mode ?? "demo"} />
        )}
        <main className="relative min-w-0 flex-1">
          <DiagramCanvas spec={spec} sessionId={sessionId} loading={loading} />
          {error && <ErrorToast msg={error} onClose={() => setError(null)} />}
        </main>
      </div>
    </div>
  );
}

function TopBar({
  status,
  showSources,
  showChat,
  onToggleSources,
  onToggleChat,
}: {
  status: StudioStatus | null;
  showSources: boolean;
  showChat: boolean;
  onToggleSources: () => void;
  onToggleChat: () => void;
}) {
  const live = status?.mode === "llm";
  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-ink-600 bg-ink-950/80 px-3 backdrop-blur">
      <div className="grid h-7 w-7 place-items-center rounded-md bg-gradient-to-br from-brand/45 to-brand-deep/40 text-white">
        <Icon icon="graph" size={15} />
      </div>
      <div className="text-[13px] font-semibold tracking-tight text-slate-100">ColPali Studio</div>
      <span className="hidden font-mono text-[10px] uppercase tracking-[0.18em] text-slate-500 sm:inline">
        grounded diagrams
      </span>
      <div className="flex-1" />
      {status && (
        <Tag
          round
          minimal
          intent={live ? "success" : "primary"}
          icon={live ? "tick-circle" : "info-sign"}
          className="!text-[11px]"
        >
          {live ? "model connected" : "demo mode"}
          {status.pages ? ` · ${status.pages} pages` : ""}
        </Tag>
      )}
      <ButtonGroup>
        <Button
          icon="database"
          active={showSources}
          variant={showSources ? "solid" : "minimal"}
          intent={showSources ? "primary" : "none"}
          onClick={onToggleSources}
          title="Toggle sources panel"
        />
        <Button
          icon="chat"
          active={showChat}
          variant={showChat ? "solid" : "minimal"}
          intent={showChat ? "primary" : "none"}
          onClick={onToggleChat}
          title="Toggle chat panel"
        />
      </ButtonGroup>
    </header>
  );
}

function ErrorToast({ msg, onClose }: { msg: string; onClose: () => void }) {
  return (
    <div className="absolute bottom-4 left-1/2 z-30 flex max-w-[80%] -translate-x-1/2 items-start gap-2 rounded-lg border border-rose-500/40 bg-rose-950/90 px-3 py-2 text-xs text-rose-100 shadow-xl">
      <Icon icon="error" size={14} className="mt-0.5 shrink-0 text-rose-300" />
      <span className="break-words">{msg}</span>
      <Button variant="minimal" small icon="cross" onClick={onClose} className="!mt-[-3px]" />
    </div>
  );
}
