import { API_BASE, request } from "./request";

/** The four review sliders; strength 0, noise 0, hue 0 and saturation 1 are the identity. */
export interface NightDarkenAdjust {
  strength: number;
  noise: number;
  hue: number;
  saturation: number;
}

export const NEUTRAL_ADJUST: NightDarkenAdjust = { strength: 0, noise: 0, hue: 0, saturation: 1 };

export interface NightDarkenDefaults {
  phrases: string[];
  thresholds: { median: number; mean: number };
  fraction: number;
  adjust: NightDarkenAdjust;
  hue_range: number;
  saturation_max: number;
  night_edits_dir_name: string;
  curve: {
    exposure_stops: number;
    gamma_lift: number;
    highlight_knee: number;
    highlight_keep: number;
    noise_max_sigma: number;
    noise_floor: number;
    noise_chroma: number;
  };
  bright_pixel_luma: number;
}

export interface NightDarkenJob {
  kind: "scan" | "commit";
  phase: string;
  done: number;
  total: number;
  error: string | null;
  running: boolean;
  elapsed: number;
  started_at: number;
  finished_at: number | null;
  pairs_planned?: number;
  night_edits_dir?: string | null;
  commit_id?: string | null;
}

export interface NightDarkenScan {
  at: number;
  phrases: string[];
  thresholds: { median: number; mean: number };
  counts: { images: number; caption_matches: number; already_dark: number; unreadable: number; candidates: number };
  already_dark: string[];
  unreadable: string[];
}

export interface NightDarkenCommit {
  id: string;
  at: number;
  items: number;
  pairs: number;
  night_edits_dir: string | null;
  fraction: number;
  restored: boolean;
  restored_at: number | null;
  restore_missing: string[];
  error: string | null;
  mean_delta_e: number | null;
}

export interface NightDarkenCounts {
  candidates: number;
  accepted: number;
  accepted_uncommitted: number;
  skipped: number;
  undecided: number;
}

export interface NightDarkenState {
  ok: boolean;
  error?: string;
  defaults: NightDarkenDefaults;
  job: NightDarkenJob | null;
  concept_dir: string | null;
  ledger_dir?: string;
  backup_dir?: string;
  default_night_edits_dir: string | null;
  scan: NightDarkenScan | null;
  counts: NightDarkenCounts | null;
  commits: NightDarkenCommit[];
}

export interface NightDarkenDecision {
  state: "accepted" | "skipped";
  at: number;
  adjust?: NightDarkenAdjust;
  strength?: number;
  delta_e?: number;
  delta_luma?: number;
  committed?: string;
}

export interface NightDarkenCandidate {
  rel: string;
  stem: string;
  split: string;
  phrases: string[];
  stats: { median: number; mean: number; p90: number; bright_fraction: number; dark_fraction: number };
  size: [number, number];
  has_caption: boolean;
  has_rejected: boolean;
  decision: NightDarkenDecision | null;
}

interface Ok {
  ok: boolean;
  error?: string;
}

export const dpoNightDarkenApi = {
  open: (conceptDir: string) =>
    request<NightDarkenState>("/dpo/night-darken/open", {
      method: "POST",
      body: JSON.stringify({ concept_dir: conceptDir }),
    }),

  state: () => request<NightDarkenState>("/dpo/night-darken/state"),

  scan: (thresholds: { median: number; mean: number }, phrases: string[]) =>
    request<Ok>("/dpo/night-darken/scan", {
      method: "POST",
      body: JSON.stringify({ thresholds, phrases }),
    }),

  status: () => request<{ ok: boolean; job: NightDarkenJob | null }>("/dpo/night-darken/status"),

  cancel: () => request<{ ok: boolean; cancelled: boolean }>("/dpo/night-darken/cancel", { method: "POST" }),

  candidates: () =>
    request<{ ok: boolean; error?: string; candidates: NightDarkenCandidate[] }>("/dpo/night-darken/candidates"),

  caption: (rel: string) =>
    request<{ ok: boolean; caption?: string }>(`/dpo/night-darken/caption?rel=${encodeURIComponent(rel)}`),

  decide: (rel: string, state: "accepted" | "skipped", adjust: NightDarkenAdjust | null) =>
    request<Ok & { decision?: NightDarkenDecision }>("/dpo/night-darken/decide", {
      method: "POST",
      body: JSON.stringify({ rel, state, adjust }),
    }),

  undo: (rel: string) =>
    request<Ok & { removed?: boolean }>("/dpo/night-darken/undo", {
      method: "POST",
      body: JSON.stringify({ rel }),
    }),

  commit: (nightEditsDir: string, fraction: number) =>
    request<Ok & { pending?: number; pairs_planned?: number; night_edits_dir?: string | null }>(
      "/dpo/night-darken/commit",
      {
        method: "POST",
        body: JSON.stringify({ night_edits_dir: nightEditsDir, fraction }),
      },
    ),

  restore: (commitId: string) =>
    request<Ok & { restored?: number; pairs_removed?: number; missing?: string[] }>("/dpo/night-darken/restore", {
      method: "POST",
      body: JSON.stringify({ commit_id: commitId }),
    }),

  previewUrl: (rel: string, adjust: NightDarkenAdjust) =>
    `${API_BASE}/dpo/night-darken/preview?rel=${encodeURIComponent(rel)}&strength=${adjust.strength.toFixed(3)}` +
    `&noise=${adjust.noise.toFixed(3)}&hue=${adjust.hue.toFixed(1)}&saturation=${adjust.saturation.toFixed(3)}`,
};
