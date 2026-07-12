import { Handle, Position, type NodeProps } from "@xyflow/react";
import { KIND_COLOR, type FlowNode } from "../lib/flow";

export function BlockNode({ data, selected }: NodeProps<FlowNode>) {
  const b = data.block;
  const color = KIND_COLOR[b.kind] ?? KIND_COLOR.component;
  return (
    <div
      className="rounded-xl border bg-ink-800/95 px-3.5 py-2.5 backdrop-blur transition-shadow"
      style={{
        borderColor: selected ? color : "#1e293b",
        borderLeft: `3px solid ${color}`,
        minWidth: 168,
        maxWidth: 220,
        boxShadow: selected ? `0 0 0 1px ${color}, 0 10px 30px -12px ${color}` : undefined,
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: color, border: "none", width: 8, height: 8 }} />
      <div
        className="mb-1 font-mono text-[9px] uppercase tracking-[0.14em]"
        style={{ color }}
      >
        {b.kind}
      </div>
      <div className="text-[13px] font-semibold leading-snug text-slate-100">{b.label}</div>
      {b.citations.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {b.citations.slice(0, 3).map((c) => (
            <span
              key={c.id}
              title={c.label}
              className="max-w-[150px] truncate rounded bg-ink-900/80 px-1.5 py-0.5 font-mono text-[9px] text-slate-400"
            >
              {c.kind === "page" ? "▤ " : c.kind === "table" ? "▦ " : "✎ "}
              {c.label}
            </span>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ background: color, border: "none", width: 8, height: 8 }} />
    </div>
  );
}
