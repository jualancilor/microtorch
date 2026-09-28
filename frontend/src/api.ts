const BASE: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface ModelConfig {
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_embd: number;
  n_head: number;
}

export interface ArchChild {
  label: string;
  detail: string;
  params: number;
}

export interface ArchNode {
  id: string;
  label: string;
  detail: string;
  params: number;
  kind: "io" | "embedding" | "block" | "norm" | "head";
  children?: ArchChild[];
}

export interface Architecture {
  config: ModelConfig;
  total_params: number;
  nodes: ArchNode[];
}

export interface ParamsResponse {
  total: number;
  layers: { name: string; params: number }[];
}

export interface StatusResponse {
  trained: boolean;
  checkpoint: string | null;
  corpus: string;
  total_params: number;
  config: ModelConfig;
}

export interface TokenizeResponse {
  ids: number[];
  tokens: string[];
  vocab_size: number;
}

export interface GenCandidate {
  token_id: number;
  token: string;
  prob: number;
}

export interface GenStep {
  token_id: number;
  token: string;
  prob: number;
  top_k: GenCandidate[];
}

export interface GenerateResponse {
  prompt: string;
  output: string;
  full_text: string;
  steps: GenStep[];
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number;
}

export interface AttentionResponse {
  tokens: string[];
  n_layer: number;
  n_head: number;
  /** [layer][head][query position][key position] */
  weights: number[][][][];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = JSON.stringify((body as { detail: unknown }).detail);
      }
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

function post<T>(path: string, body: object): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  architecture: (): Promise<Architecture> => request("/architecture"),
  params: (): Promise<ParamsResponse> => request("/params"),
  status: (): Promise<StatusResponse> => request("/status"),
  tokenize: (text: string): Promise<TokenizeResponse> => post("/tokenize", { text }),
  generate: (req: GenerateRequest): Promise<GenerateResponse> => post("/generate", req),
  attention: (text: string): Promise<AttentionResponse> => post("/attention", { text }),
};

/** Render whitespace/control chars visibly in token chips. */
export function displayToken(t: string): string {
  return t.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");
}
