import { useCallback, useEffect, useMemo, useRef } from "react";
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
import { AnchorButton, ButtonGroup, Callout, Icon, Tag } from "@blueprintjs/core";
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

  const warn = spec && (spec.hallucinated_citations.length > 0 || spec.dropped_connections > 0);

  return (
    <div className="relative h-full w-full">
      {spec && (
        <div className="pointer-events-none absolute left-4 top-4 z-10 flex items-center gap-2">
          <h2 className="pointer-events-auto max-w-[42vw] truncate text-lg font-semibold text-slate-100">
            {spec.title}
          </h2>
          <Tag
            round
            intent={spec.mode.startsWith("demo") ? "primary" : "success"}
            className="pointer-events-auto !font-mono !text-[10px] !uppercase !tracking-wide"
          >
            {spec.mode.startsWith("demo") ? "demo" : "grounded"}
          </Tag>
          {warn && (
            <Tag round intent="warning" className="pointer-events-auto !font-mono !text-[10px]">
              {spec.dropped_connections > 0 && `${spec.dropped_connections} dropped `}
              {spec.hallucinated_citations.length > 0 &&
                `${spec.hallucinated_citations.length} bad cite`}
            </Tag>
          )}
        </div>
      )}

      {spec && sessionId && (
        <div className="absolute right-4 top-4 z-10">
          <ButtonGroup>
            <ExportBtn sid={sessionId} fmt="json" icon="code" label="JSON" />
            <ExportBtn sid={sessionId} fmt="summary" icon="align-left" label="Summary" />
            <ExportBtn sid={sessionId} fmt="mermaid" icon="flows" label="Mermaid" />
            <ExportBtn sid={sessionId} fmt="drawio" icon="diagram-tree" label=".drawio" />
          </ButtonGroup>
        </div>
      )}

      {spec && <SpecBanner spec={spec} />}

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
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1e3d5a" />
        <Controls showInteractive={false} />
        <MiniMap
          pannable
          zoomable
          nodeColor={(n) => {
            const b = (n.data as BlockNodeData | undefined)?.block;
            return b ? KIND_COLOR[b.kind] ?? "#2b5175" : "#2b5175";
          }}
          maskColor="rgba(10,26,43,0.75)"
        />
      </ReactFlow>
    </div>
  );
}

function ExportBtn({
  sid,
  fmt,
  icon,
  label,
}: {
  sid: string;
  fmt: "drawio" | "mermaid" | "json" | "summary";
  icon: "code" | "align-left" | "flows" | "diagram-tree";
  label: string;
}) {
  return (
    <AnchorButton href={api.exportUrl(sid, fmt)} variant="outlined" small icon={<Icon icon={icon} size={12} />}>
      {label}
    </AnchorButton>
  );
}

function SpecBanner({ spec }: { spec: Spec }) {
  const fallback = spec.mode === "demo-fallback"; // model was called but failed
  const demo = spec.mode === "demo"; // no model configured at all
  if (!fallback && !demo) return null;
  const reasons = (spec.errors ?? []).slice(0, 3);
  return (
    <div className="pointer-events-none absolute bottom-4 left-1/2 z-20 w-[min(600px,92%)] -translate-x-1/2">
      <Callout
        intent={fallback ? "danger" : "primary"}
        icon={fallback ? "warning-sign" : "info-sign"}
        title={
          fallback
            ? "The answer model failed — this is a rough keyword sketch, not a real diagram"
            : "No answer model configured — showing a rough demo"
        }
        className="pointer-events-auto !bg-ink-800/95 text-xs shadow-2xl"
      >
        {fallback ? (
          <>
            The model was called but rejected the request. Fix the cause below, then regenerate:
            {reasons.length > 0 && (
              <ul className="mt-1.5 space-y-0.5 break-words font-mono text-[11px] text-rose-200">
                {reasons.map((e, i) => (
                  <li key={i}>• {e}</li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <>
            Set <code>VLM_BASE_URL</code> to a vision model so it actually reads your pages and
            Excels.
          </>
        )}
      </Callout>
    </div>
  );
}

function EmptyState({ thinking }: { thinking?: boolean }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-[5] grid place-items-center">
      <div className="text-center">
        <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-2xl border border-ink-600 bg-ink-800/60 text-brand">
          <Icon icon="graph" size={26} />
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
