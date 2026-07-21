import { useEffect, useRef, useState } from "react";
import { Button, Card, Icon, TextArea } from "@blueprintjs/core";
import type { Spec } from "../types";

export interface Msg {
  role: "user" | "assistant";
  text: string;
  spec?: Spec;
}

const EXAMPLES = [
  "Map how the parts in these sources connect",
  "Show the flow from input to output across the selected sources",
  "Group the components into subsystems and link them",
];

export function ChatPanel({
  messages,
  onSend,
  loading,
  mode,
}: {
  messages: Msg[];
  onSend: (text: string) => void;
  loading: boolean;
  mode: "llm" | "demo";
}) {
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = () => {
    const t = text.trim();
    if (!t || loading) return;
    onSend(t);
    setText("");
  };

  return (
    <section className="flex h-full w-[340px] shrink-0 flex-col border-r border-ink-600 bg-ink-950/40">
      <div className="flex items-center justify-between border-b border-ink-600/60 px-4 py-3">
        <h3 className="font-mono text-[10px] uppercase tracking-[0.16em] text-slate-500">
          Conversation
        </h3>
        <span className="font-mono text-[10px] text-slate-600">
          {mode === "llm" ? "reads sources" : "text-only demo"}
        </span>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && !loading && (
          <div className="space-y-2">
            <p className="text-sm text-slate-400">Describe what you want. Try:</p>
            {EXAMPLES.map((e) => (
              <Card
                key={e}
                interactive
                compact
                onClick={() => onSend(e)}
                className="!bg-ink-800/60 !p-3 text-[13px] text-slate-300 transition hover:!bg-ink-700/70 hover:text-slate-100"
              >
                {e}
              </Card>
            ))}
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="fadeup flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-brand/20 px-3.5 py-2 text-[13px] text-slate-50">
                {m.text}
              </div>
            </div>
          ) : (
            <AssistantMsg key={i} msg={m} />
          ),
        )}

        {loading && (
          <div className="thinking flex items-center gap-1 px-1 font-mono text-xs text-slate-500">
            reading + designing<span>.</span>
            <span>.</span>
            <span>.</span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="border-t border-ink-600/60 p-3">
        <div className="flex items-end gap-2">
          <TextArea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            fill
            autoResize
            rows={1}
            placeholder="Describe what you want, e.g. how the parts connect"
            className="!max-h-32 !min-h-[38px] text-[13px]"
          />
          <Button
            intent="primary"
            icon="arrow-up"
            onClick={submit}
            disabled={loading || !text.trim()}
            title="Send (Enter)"
            className="!rounded-lg"
          />
        </div>
      </div>
    </section>
  );
}

function AssistantMsg({ msg }: { msg: Msg }) {
  const spec = msg.spec;
  const sourcesUsed = spec ? dedupeSources(spec) : [];
  return (
    <div className="fadeup space-y-2">
      <div className="rounded-2xl rounded-bl-sm border border-ink-600 bg-ink-800/70 px-3.5 py-2.5">
        <p className="text-[13px] leading-relaxed text-slate-200">{msg.text}</p>
        {spec && spec.assumptions.length > 0 && (
          <ul className="mt-2 space-y-1 border-t border-ink-600/60 pt-2">
            {spec.assumptions.map((a, i) => (
              <li key={i} className="flex gap-1.5 text-[11px] text-slate-400">
                <span className="text-slate-600">·</span>
                {a}
              </li>
            ))}
          </ul>
        )}
      </div>
      {sourcesUsed.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 px-1">
          <span className="font-mono text-[10px] uppercase tracking-wide text-slate-600">
            grounded in
          </span>
          {sourcesUsed.map((s) => (
            <span
              key={s.id}
              className="inline-flex items-center gap-1 rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400"
            >
              <Icon icon={s.kind === "table" ? "th" : s.kind === "note" ? "annotation" : "document"} size={10} />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function dedupeSources(spec: Spec) {
  const seen = new Map<string, { id: string; kind: string; label: string }>();
  for (const b of spec.blocks) for (const c of b.citations) seen.set(c.id, c);
  for (const e of spec.connections) for (const c of e.citations) seen.set(c.id, c);
  return [...seen.values()].slice(0, 8);
}
