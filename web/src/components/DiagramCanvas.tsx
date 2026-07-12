import { useCallback, useEffect, useMemo, useRef, type ReactNode } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type NodeTypes,
  type ReactFlowInstance,
} from "@xyflow/react";
import { BlockNode } from "./BlockNode";
import { KIND_COLOR, specToFlow, type BlockNodeData, type FlowNode } from "../lib/flow";
import { api } from "../api";
import type { Spec } from "../types";

export function DiagramCanvas({
  spec,
  sessionId,
  loading,
}: {
  spec: Spec | null;
  sessionId: string | null;
  loading: boolean;
}) {
  const nodeTypes = useMemo(() => ({ block: BlockNode }) as NodeTypes, []);
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const inst = useRef<ReactFlowInstance<FlowNode, Edge> | null>(null);

  useEffect(() => {
    if (!spec) return;
    const { nodes: n, edges: e } = specToFlow(spec);
    setNodes(n);
    setEdges(e);
    const t = setTimeout(() => inst.current?.fitView({ padding: 0.2, duration: 400 }), 60);
    return () => clearTimeout(t);
  }, [spec, setNodes, setEdges]);

  const onInit = useCallback((i: ReactFlowInstance<FlowNode, Edge>) => {
    inst.current = i;
  }, []);

  const warn =
    spec && (spec.hallucinated_citations.length > 0 || spec.dropped_connections > 0);

  return (
    <div className="relative h-full w-full">
      {spec && (
        <div className="pointer-events-none absolute left-4 top-4 z-10 flex flex-col gap-2">
          <div className="pointer-events-auto flex items-center gap-2">
            <h2 className="max-w-[42vw] truncate text-lg font-semibold text-slate-100">
              {spec.title}
            </h2>
            <Badge tone={spec.mode.startsWith("demo") ? "demo" : "live"}>
              {spec.mode.startsWith("demo") ? "demo" : "grounded"}
            </Badge>
            {warn && (
              <Badge tone="warn">
                {spec.dropped_connections > 0 && `${spec.dropped_connections} dropped `}
                {spec.hallucinated_citations.length > 0 &&
                  `${spec.hallucinated_citations.length} bad cite`}
              </Badge>
            )}
          </div>
        </div>
      )}

      {spec && sessionId && (
        <div className="absolute right-4 top-4 z-10 flex gap-1.5">
          <ExportBtn href={api.exportUrl(sessionId, "mermaid")}>Mermaid</ExportBtn>
          <ExportBtn href={api.exportUrl(sessionId, "drawio")}>.drawio</ExportBtn>
        </div>
      )}

      {!spec && !loading && <EmptyState />}
      {loading && !spec && <EmptyState thinking />}

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onInit={onInit}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        className="bg-transparent"
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1e293b" />
        <Controls showInteractive={false} />
        <MiniMap
          pannable
          zoomable
          nodeColor={(n) => {
            const b = (n.data as BlockNodeData | undefined)?.block;
            return b ? KIND_COLOR[b.kind] ?? "#334155" : "#334155";
          }}
          maskColor="rgba(11,16,32,0.75)"
        />
      </ReactFlow>
    </div>
  );
}

function Badge({ children, tone }: { children: ReactNode; tone: "demo" | "live" | "warn" }) {
  const map = {
    live: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    demo: "bg-sky-500/15 text-sky-300 border-sky-500/30",
    warn: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  } as const;
  return (
    <span
      className={`rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide ${map[tone]}`}
    >
      {children}
    </span>
  );
}

function ExportBtn({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      className="rounded-lg border border-ink-600 bg-ink-800/80 px-2.5 py-1 text-xs text-slate-300 backdrop-blur transition hover:border-brand hover:text-brand"
    >
      ↓ {children}
    </a>
  );
}

function EmptyState({ thinking }: { thinking?: boolean }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-[5] grid place-items-center">
      <div className="text-center">
        <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-2xl border border-ink-600 bg-ink-800/60">
          <span className="text-2xl">◇</span>
        </div>
        {thinking ? (
          <p className="thinking font-mono text-sm text-slate-400">
            designing<span>.</span>
            <span>.</span>
            <span>.</span>
          </p>
        ) : (
          <p className="max-w-xs text-sm text-slate-500">
            Describe what you want — the model reads your selected sources and builds a
            cited, structured result here.
          </p>
        )}
      </div>
    </div>
  );
}
