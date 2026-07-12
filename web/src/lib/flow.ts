import { MarkerType, type Edge, type Node } from "@xyflow/react";
import type { Block, Kind, Spec } from "../types";

export const KIND_COLOR: Record<Kind, string> = {
  component: "#38bdf8",
  system: "#818cf8",
  process: "#2dd4bf",
  io: "#fbbf24",
  store: "#a78bfa",
  external: "#fb7185",
  actor: "#4ade80",
};

export const EDGE_COLOR: Record<string, string> = {
  data: "#38bdf8",
  signal: "#2dd4bf",
  control: "#818cf8",
  power: "#fbbf24",
  bus: "#a78bfa",
  dependency: "#94a3b8",
};

export type BlockNodeData = { block: Block };
export type FlowNode = Node<BlockNodeData, "block">;

const COL_W = 260;
const ROW_H = 130;

/** Longest-path layering so edges mostly point left→right; rows stack within a layer. */
export function specToFlow(spec: Spec): { nodes: FlowNode[]; edges: Edge[] } {
  const ids = spec.blocks.map((b) => b.id);
  const layer: Record<string, number> = {};
  ids.forEach((i) => (layer[i] = 0));
  for (let k = 0; k < ids.length; k++) {
    for (const e of spec.connections) {
      if (layer[e.from] != null && layer[e.to] != null) {
        layer[e.to] = Math.max(layer[e.to], layer[e.from] + 1);
      }
    }
  }
  const rowOf: Record<number, number> = {};
  const nodes: FlowNode[] = spec.blocks.map((b) => {
    const lx = layer[b.id] ?? 0;
    const row = rowOf[lx] ?? 0;
    rowOf[lx] = row + 1;
    return {
      id: b.id,
      type: "block",
      position: { x: lx * COL_W, y: row * ROW_H },
      data: { block: b },
    };
  });

  const edges: Edge[] = spec.connections.map((e, i) => {
    const color = EDGE_COLOR[e.kind] ?? EDGE_COLOR.data;
    const heavy = e.kind === "power" || e.kind === "bus";
    return {
      id: `e${i}`,
      source: e.from,
      target: e.to,
      label: e.label || undefined,
      animated: e.kind === "data" || e.kind === "signal",
      style: {
        stroke: color,
        strokeWidth: heavy ? 2.5 : 1.5,
        strokeDasharray: e.kind === "control" || e.kind === "dependency" ? "6 4" : undefined,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 18, height: 18 },
      labelStyle: { fill: "#cbd5e1", fontSize: 11, fontWeight: 500 },
      labelBgStyle: { fill: "#0f172a", fillOpacity: 0.9 },
      labelBgPadding: [4, 2],
      labelBgBorderRadius: 4,
    };
  });

  return { nodes, edges };
}
