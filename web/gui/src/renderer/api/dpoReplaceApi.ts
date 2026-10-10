import { request } from "./request";

export interface ReplaceSettings {
  concept_dir: string;
  generated_folder: string;
  base_url: string;
  token: string;
  images_per_pair: number;
}

export interface ReplaceGroup {
  session_id: string;
  group_id: string;
  caption: string;
  best_id: string | null;
  size: [number, number];
  images: Array<{ id: string; url: string; current_rejected?: boolean }>;
}

export interface ReplaceStatus {
  ok: boolean;
  active: boolean;
  session_id?: string;
  concept_dir?: string;
  generated_folder?: string;
  base_url?: string;
  running?: boolean;
  paused?: boolean;
  closing?: boolean;
  complete?: boolean;
  total?: number;
  counts?: Record<"queued" | "generating" | "ready" | "applied" | "skipped" | "failed", number>;
  params?: Record<string, unknown>;
  images_per_pair?: number;
  images?: number;
  error?: string | null;
  errors?: Array<{ group_id: string; pair_id: number; error: string }>;
  can_undo?: boolean;
}

export interface ReplaceResult {
  ok: boolean;
  error?: string;
  group?: ReplaceGroup | null;
  requires_reversal_confirmation?: boolean;
  /** How many pairs a save wrote. */
  pairs?: number;
}

/** One pair built from a review group: the first replaces the reviewed pair, the rest are added. */
export interface ReplacePair {
  chosen_id: string;
  rejected_id: string;
}

/** Fresh images the backend generates per pair at most. */
export const MAX_IMAGES_PER_PAIR = 10;

const post = <T = ReplaceResult>(route: string, body: unknown) =>
  request<T>(`/dpo/replace/${route}`, { method: "POST", body: JSON.stringify(body) });
const ids = (group: ReplaceGroup) => ({ session_id: group.session_id, group_id: group.group_id });

export const dpoReplaceApi = {
  concepts: () => request<{ ok: boolean; concepts: string[]; warning?: string }>("/dpo/replace/concepts"),
  start: (settings: ReplaceSettings, newPass = false) =>
    post<ReplaceStatus>("start", { ...settings, new_pass: newPass }),
  restore: (settings: ReplaceSettings) => post<ReplaceStatus>("restore", settings),
  status: () => request<ReplaceStatus>("/dpo/replace/status"),
  current: () => request<ReplaceResult>("/dpo/replace/current"),
  pause: (sessionId: string) => post("pause", { session_id: sessionId }),
  resume: (sessionId: string, token: string) => post("resume", { session_id: sessionId, token }),
  close: (sessionId: string) => post("close", { session_id: sessionId }),
  best: (group: ReplaceGroup, bestId: string | null) => post("best", { ...ids(group), best_id: bestId }),
  apply: (group: ReplaceGroup, worstId: string, overrideReversal = false) =>
    post("apply", { ...ids(group), best_id: group.best_id, worst_id: worstId, override_reversal: overrideReversal }),
  applyPairs: (group: ReplaceGroup, pairs: ReplacePair[], overrideReversal = false) =>
    post("apply-pairs", { ...ids(group), pairs, override_reversal: overrideReversal }),
  skip: (sessionId: string, groupId: string) => post("skip", { session_id: sessionId, group_id: groupId }),
  retry: (sessionId: string, groupId: string) => post("retry", { session_id: sessionId, group_id: groupId }),
  undo: (sessionId: string) => post("undo", { session_id: sessionId }),
};
