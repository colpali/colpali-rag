import type { DiagramResponse, DocInfo, Source, StudioStatus } from "./types";

async function J<T>(r: Response): Promise<T> {
  if (r.ok) return r.json() as Promise<T>;
  let detail = r.statusText;
  try {
    const e = await r.json();
    detail = e.detail || detail;
  } catch {
    /* non-JSON error body */
  }
  throw new Error(detail);
}

const form = (obj: Record<string, string>) => {
  const f = new FormData();
  for (const [k, v] of Object.entries(obj)) f.append(k, v);
  return f;
};

export const api = {
  status: () => fetch("/api/studio/status").then(J<StudioStatus>),

  newSession: () =>
    fetch("/api/studio/session", { method: "POST" })
      .then(J<{ session_id: string }>)
      .then((d) => d.session_id),

  sources: () =>
    fetch("/api/studio/sources").then(J<{ docs: DocInfo[]; pages: number; note?: string }>),

  upload: (sessionId: string, file: File) => {
    const f = new FormData();
    f.append("session_id", sessionId);
    f.append("file", file);
    return fetch("/api/studio/upload", { method: "POST", body: f }).then(
      J<{ status: string; session: SessionState }>,
    );
  },

  diagram: (sessionId: string, message: string, docs: string[], topK = 6) =>
    fetch("/api/studio/generate", {
      method: "POST",
      body: form({
        session_id: sessionId,
        message,
        docs: docs.join(","),
        top_k: String(topK),
      }),
    }).then(J<DiagramResponse>),

  exportUrl: (sessionId: string, fmt: "drawio" | "mermaid" | "json" | "summary") =>
    `/api/studio/export?session_id=${sessionId}&fmt=${fmt}`,

  imageUrl: (pageId: string) =>
    `/api/studio/image?page_id=${encodeURIComponent(pageId)}`,
};

export interface SessionState {
  session_id: string;
  selected_docs: string[];
  tables: { name: string; rows: number; columns: string[]; sheet: string | null }[];
  notes: { name: string; chars: number }[];
  history: { request: string; title: string | null }[];
}

export type { Source };
