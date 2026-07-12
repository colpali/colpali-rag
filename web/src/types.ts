export type Kind =
  | "component"
  | "system"
  | "process"
  | "io"
  | "store"
  | "external"
  | "actor";

export interface Source {
  id: string;
  kind: string; // "page" | "table" | "note"
  label: string;
  doc?: string;
  page?: number;
  ref?: string;
  score?: number;
}

export interface Block {
  id: string;
  label: string;
  kind: Kind;
  group: string | null;
  cites: string[];
  citations: Source[];
}

export interface Conn {
  from: string;
  to: string;
  label: string;
  kind: string; // "data" | "control" | "signal" | "power" | "bus" | "dependency"
  cites: string[];
  citations: Source[];
}

export interface GroupT {
  id: string;
  label: string;
}

export interface Spec {
  title: string;
  reasoning: string;
  assumptions: string[];
  groups: GroupT[];
  blocks: Block[];
  connections: Conn[];
  structured: boolean;
  mode: string;
  hallucinated_citations: number[];
  dropped_connections: number;
}

export interface DiagramResponse {
  session_id: string;
  spec: Spec;
  sources: Source[];
}

export interface DocInfo {
  doc: string;
  pages: number;
}

export interface StudioStatus {
  index: boolean;
  pages: number;
  llm: boolean;
  mode: "llm" | "demo";
  index_error?: string | null;
}
