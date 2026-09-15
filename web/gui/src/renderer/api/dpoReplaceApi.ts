import { request } from "./request";

/** Presentation order of the ranked pair list. */
export type ReplaceOrder = "stale_first" | "oldest" | "least_realistic" | "path";

export interface ReplaceProvenance {
  model: string;
  loras: string[];
  loraweights: number[];
  generated_at: number | null;
}

export interface ReplaceStatus {
  ok: boolean;
  active: boolean;
  phase: "idle" | "scanning" | "ready" | "error";
  progress_done: number;
  progress_total: number;
  scan_finished: boolean;
  error: string | null;
  warning: string | null;
  pairs: number;
  /** Pairs whose caption was corrected after export (both sides answer an old prompt). */
  stale_captions: number;
  order: ReplaceOrder;
  allow_promote: boolean;
  running: boolean;
  run_phase: "idle" | "generating" | "done" | "error" | "cancelled";
  params: Record<string, unknown>;
  param_source: string;
  planned: number;
  done: number;
  failed: number;
  images: number;
  run_error: string | null;
  auto_apply: boolean;
  ready: number;
  has_proposal: boolean;
  shown: number;
  applied: number;
  auto_applied: number;
  promoted: number;
  kept: number;
  unscored: number;
  can_undo: boolean;
  /** "keep" or a "chosen<-x rejected<-y" label. */
  undo_action: string | null;
}

export interface ReplacePair {
  pair_id: number;
  chosen_path: string;
  rejected_path: string;
  chosen_file: string;
  rejected_file: string;
  /** The caption the trainer reads (chosen/<pair>.txt). */
  caption: string;
  caption_source: "pair" | "manifest";
  caption_changed: boolean;
  manifest_prompt: string;
  chosen_size: [number, number] | null;
  rejected_size: [number, number] | null;
  chosen_provenance: ReplaceProvenance;
  rejected_provenance: ReplaceProvenance;
  /** Model + adapters that made the rejected side, as a label. */
  rejected_model: string;
  rejected_mtime: number | null;
  chosen_score: number | null;
  rejected_score: number | null;
  replacements: number;
  stale_caption: boolean;
}

/** One image of a pair's selection group. */
export interface ReplaceGroupImage {
  path: string;
  /** Where it sits now: the old chosen, the old rejected, or a fresh generation. */
  role: "chosen" | "rejected" | "fresh";
  size: [number, number] | null;
  score: number | null;
}

export interface ReplaceProposal extends ReplacePair {
  /** Old chosen first, old rejected second, then the fresh images. */
  images: ReplaceGroupImage[];
  fresh_count: number;
  /** Pre-selection from the realism scores; the reviewer may change it. */
  suggested_chosen: string;
  suggested_rejected: string;
  suggestion_is_keep: boolean;
  scored: boolean;
  reason: string;
  /** Promotion is off for the session: only the rejected side is picked. */
  chosen_locked: boolean;
  new_provenance: ReplaceProvenance;
  new_model: string;
  index: number;
  queued: number;
}

export interface ReplaceModelGroup {
  label: string;
  pairs: number;
  oldest: number | null;
  newest: number | null;
}

export const dpoReplaceApi = {
  concepts: () => request<{ ok: boolean; concepts: string[]; warning?: string }>("/dpo/replace/concepts"),

  start: (
    conceptDir: string,
    generatedFolder: string,
    baseUrl: string,
    token: string,
    order: ReplaceOrder,
    allowPromote: boolean,
  ) =>
    request<{ ok: boolean; pairs?: number; pruned?: number; error?: string }>("/dpo/replace/start", {
      method: "POST",
      body: JSON.stringify({
        concept_dir: conceptDir,
        generated_folder: generatedFolder,
        base_url: baseUrl,
        token,
        order,
        allow_promote: allowPromote,
      }),
    }),

  status: () => request<ReplaceStatus>("/dpo/replace/status"),

  pairs: (offset = 0, limit = 200) =>
    request<{
      ok: boolean;
      waiting?: boolean;
      pairs: ReplacePair[];
      total: number;
      offset?: number;
      models: ReplaceModelGroup[];
      error?: string;
    }>(`/dpo/replace/pairs?offset=${offset}&limit=${limit}`),

  // The settings the next batch would use: the user's last generation.
  params: () =>
    request<{ ok: boolean; params?: Record<string, unknown>; model?: string; found?: boolean; error?: string }>(
      "/dpo/replace/params",
    ),

  generate: (count: number, imagesPerPair: number, concurrency: number, autoApply: boolean, pairIds?: number[]) =>
    request<{ ok: boolean; planned?: number; model?: string; params?: Record<string, unknown>; error?: string }>(
      "/dpo/replace/generate",
      {
        method: "POST",
        body: JSON.stringify({
          count,
          pair_ids: pairIds ?? null,
          images_per_pair: imagesPerPair,
          concurrency,
          auto_apply: autoApply,
        }),
      },
    ),

  next: () =>
    request<{ ok: boolean; proposal?: ReplaceProposal; waiting?: boolean; done?: boolean; error?: string }>(
      "/dpo/replace/next",
      { method: "POST" },
    ),

  current: () => request<{ ok: boolean; proposal?: ReplaceProposal | null; error?: string }>("/dpo/replace/current"),

  // Make two images of the group the pair's sides; omitted = the suggestion.
  apply: (chosenPath?: string, rejectedPath?: string) =>
    request<{ ok: boolean; action?: string; pair_id?: number; error?: string }>("/dpo/replace/apply", {
      method: "POST",
      body: JSON.stringify({ chosen_path: chosenPath ?? null, rejected_path: rejectedPath ?? null }),
    }),

  keep: () => request<{ ok: boolean; pair_id?: number; error?: string }>("/dpo/replace/keep", { method: "POST" }),

  undo: () =>
    request<{ ok: boolean; undone?: string; proposal?: ReplaceProposal; error?: string }>("/dpo/replace/undo", {
      method: "POST",
    }),

  cancel: () => request<{ ok: boolean }>("/dpo/replace/cancel", { method: "POST" }),
};
