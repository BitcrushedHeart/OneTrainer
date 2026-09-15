import { API_BASE, request } from "./request";

export type TextAlignmentPhase = "idle" | "scanning" | "scanned" | "building" | "built" | "error" | "cancelled";

export interface TextAlignmentScanRequest {
  source_folder: string;
  output_folder: string;
  recursive: boolean;
  keep_absent: boolean;
  keep_verbatim: boolean;
  min_boxed_words: number;
  min_agreement: number;
  limit: number;
  ocr_device: string;
  detector_device: string;
}

export interface TextAlignmentStatus {
  ok: boolean;
  phase: TextAlignmentPhase;
  running: boolean;
  error: string | null;
  progress_done: number;
  progress_total: number;
  current: string;
  /** Per-reason counts of the last scan: keep, no_metadata, no_text_span, ... */
  counts: Record<string, number>;
  kept: number;
  scanned: number;
  build: { written: number; skipped: number; cancelled: boolean } | null;
  output_folder: string | null;
}

export interface TextAlignmentCandidate {
  source_image: string;
  sha256: string;
  prompt: string;
  text: string;
  words: string[];
  ocr_reading: string | null;
  text_score: number | null;
  verdicts: string[];
  boxed_words: number;
  reward_reading: string | null;
  reader_agreement: number | null;
  reason: string;
  keep: boolean;
}

export interface TextAlignmentResults {
  ok: boolean;
  total: number;
  items: TextAlignmentCandidate[];
}

export interface WildcardRequest {
  count: number;
  seed: number;
  name: string;
  directory: string;
  min_words: number;
  max_words: number;
}

export interface WildcardResponse {
  ok: boolean;
  error?: string;
  path?: string;
  count?: number;
  sample?: string[];
}

export interface OkResponse {
  ok: boolean;
  error?: string;
}

export const textAlignmentApi = {
  imageUrl: (path: string) => `${API_BASE}/text-alignment/image?path=${encodeURIComponent(path)}`,
  scan: (body: TextAlignmentScanRequest) =>
    request<OkResponse>("/text-alignment/scan", { method: "POST", body: JSON.stringify(body) }),
  build: () => request<OkResponse>("/text-alignment/build", { method: "POST" }),
  cancel: () => request<OkResponse>("/text-alignment/cancel", { method: "POST" }),
  status: () => request<TextAlignmentStatus>("/text-alignment/status"),
  results: (offset = 0, limit = 500, onlyKept = true) =>
    request<TextAlignmentResults>(`/text-alignment/results?offset=${offset}&limit=${limit}&only_kept=${onlyKept}`),
  wildcard: (body: WildcardRequest) =>
    request<WildcardResponse>("/text-alignment/wildcard", { method: "POST", body: JSON.stringify(body) }),
};
