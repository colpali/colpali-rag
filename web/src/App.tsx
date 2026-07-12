import { useCallback, useEffect, useState } from "react";
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
        setMessages((m) => [...m, { role: "assistant", text: "⚠ " + msg }]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId, selected, loading],
  );

  const upload = useCallback(
    async (file: File) => {
      if (!sessionId) return;
      setUploading(true);
      setError(null);
      try {
        const r = await api.upload(sessionId, file);
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
    <div className="flex h-screen w-screen overflow-hidden">
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
      <ChatPanel
        messages={messages}
        onSend={send}
        loading={loading}
        mode={status?.mode ?? "demo"}
      />
      <main className="relative min-w-0 flex-1">
        <DiagramCanvas spec={spec} sessionId={sessionId} loading={loading} />
        {error && (
          <div className="absolute bottom-4 left-1/2 z-20 -translate-x-1/2 rounded-lg border border-rose-500/40 bg-rose-500/15 px-3 py-1.5 text-xs text-rose-200">
            {error}
          </div>
        )}
      </main>
    </div>
  );
}
