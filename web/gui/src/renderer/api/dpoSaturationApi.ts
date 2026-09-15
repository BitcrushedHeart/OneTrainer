import { API_BASE, request } from "./request";

/** State of one concept's saturation table, as written by the scorer. */
export interface SaturationTableInfo {
  pairs: number;
  scored: number;
  draws: number;
  origin: string | null;
  current: string | null;
  current_scored_at: string | null;
  num_train_timesteps: number | null;
}

export interface SaturationConcept {
  name: string;
  path: string;
  chosen_dir: string;
  rejected_dir: string;
  /** Per-concept beta override; 0 = inherit the global beta. */
  beta_override: number;
  retired: number;
  table: SaturationTableInfo | null;
}

export interface SaturationConceptsResponse {
  ok: boolean;
  concepts: SaturationConcept[];
  global_beta?: number;
  default_origin?: string;
  adapter?: string;
  cache_dir?: string;
  warning?: string;
}

export type SaturationPhase = "idle" | "running" | "done" | "error" | "cancelled";

export interface SaturationStatus {
  ok: boolean;
  running: boolean;
  phase: SaturationPhase;
  /** Free-text phase of the child process: loading model, origin pass, current pass... */
  progress_phase: string;
  done: number;
  total: number;
  error: string | null;
  log: string[];
  started_at: number | null;
  elapsed: number;
  request: Record<string, unknown>;
}

export interface SaturationRow {
  key: string;
  chosen: string | null;
  rejected: string | null;
  draws: number;
  mean_margin: number;
  /** beta * mean margin: nats of preference already absorbed. */
  mean_logit: number;
  /** Expected gradient weight sigmoid(-beta * margin). */
  mean_weight: number;
  min_weight: number;
  max_weight: number;
  /** Mean weight per timestep quartile (low noise first); null when unsampled. */
  bucket_weight: Array<number | null>;
  bucket_margin: Array<number | null>;
  /** Fraction of draws where the current model prefers chosen over origin. */
  wins: number;
  saturated: boolean;
}

export interface SaturationSummary {
  ok: boolean;
  error?: string;
  beta: number;
  dead_weight: number;
  require_all_buckets: boolean;
  min_draws: number;
  buckets: number;
  pairs: number;
  pairs_in_table: number;
  draws_per_pair: number;
  saturated: number;
  saturated_keys: string[];
  wins_mean: number | null;
  logit_percentiles: Record<string, number>;
  weight_percentiles: Record<string, number>;
  bucket_counts: number[];
  bucket_dead: number[];
  utility: { total_weight: number; pairs: number; share: Record<string, number> };
  beta_sweep: Array<{ multiplier: number; beta: number; dead: number; pairs: number }>;
  most_saturated: SaturationRow[];
  most_negative: SaturationRow[];
  concept_path: string;
  retired: number;
  origin: string | null;
  current: string | null;
  current_scored_at: string | null;
  logit_histogram: { edges: number[]; counts: number[] };
  utility_curve: Array<{ pairs: number; share: number }>;
  saturated_rows: SaturationRow[];
  saturated_truncated: number;
}

export interface SaturationPairDetail {
  ok: boolean;
  error?: string;
  key: string;
  chosen: string | null;
  rejected: string | null;
  beta: number;
  draws: Array<{ t: number; t_frac: number; margin: number; logit: number; weight: number }>;
  retired: boolean;
}

export interface RetiredPair {
  key: string;
  reason: string;
  retired_at: string | null;
  chosen: string | null;
  rejected: string | null;
}

export interface SaturationRule {
  beta: number | null;
  deadWeight: number;
  requireAllBuckets: boolean;
  minDraws: number;
}

export interface StartScoringParams {
  current: string;
  origin: string;
  adapter: string;
  draws: number;
  timesteps: string;
  seeds: number;
  cacheDir: string;
  forceOrigin: boolean;
  concepts: string[];
}

function ruleBody(conceptPath: string, rule: SaturationRule) {
  return {
    concept_path: conceptPath,
    beta: rule.beta,
    dead_weight: rule.deadWeight,
    require_all_buckets: rule.requireAllBuckets,
    min_draws: rule.minDraws,
  };
}

export const dpoSaturationApi = {
  concepts: () => request<SaturationConceptsResponse>("/dpo/saturation/concepts"),

  start: (p: StartScoringParams) =>
    request<{ ok: boolean; error?: string }>("/dpo/saturation/start", {
      method: "POST",
      body: JSON.stringify({
        current: p.current,
        origin: p.origin,
        adapter: p.adapter,
        draws: p.draws,
        timesteps: p.timesteps,
        seeds: p.seeds,
        cache_dir: p.cacheDir,
        force_origin: p.forceOrigin,
        concepts: p.concepts,
      }),
    }),

  status: () => request<SaturationStatus>("/dpo/saturation/status"),

  cancel: () => request<{ ok: boolean; cancelled: boolean }>("/dpo/saturation/cancel", { method: "POST" }),

  summary: (conceptPath: string, rule: SaturationRule, tails = 24) =>
    request<SaturationSummary>("/dpo/saturation/summary", {
      method: "POST",
      body: JSON.stringify({ ...ruleBody(conceptPath, rule), tails }),
    }),

  pair: (conceptPath: string, key: string, beta: number | null) =>
    request<SaturationPairDetail>("/dpo/saturation/pair", {
      method: "POST",
      body: JSON.stringify({ concept_path: conceptPath, key, beta }),
    }),

  retireKeys: (conceptPath: string, keys: string[], reason: string) =>
    request<{ ok: boolean; error?: string; moved: string[]; retired: number }>("/dpo/saturation/retire", {
      method: "POST",
      body: JSON.stringify({ concept_path: conceptPath, keys, reason }),
    }),

  retireSaturated: (conceptPath: string, rule: SaturationRule, dryRun: boolean) =>
    request<{ ok: boolean; error?: string; moved: string[]; dry_run: boolean; retired: number }>(
      "/dpo/saturation/retire",
      { method: "POST", body: JSON.stringify({ ...ruleBody(conceptPath, rule), keys: null, dry_run: dryRun }) },
    ),

  restore: (conceptPath: string, keys: string[] | null) =>
    request<{ ok: boolean; error?: string; restored: string[]; retired: number }>("/dpo/saturation/restore", {
      method: "POST",
      body: JSON.stringify({ concept_path: conceptPath, keys }),
    }),

  retired: (conceptPath: string) =>
    request<{ ok: boolean; error?: string; pairs: RetiredPair[] }>(
      `/dpo/saturation/retired?concept_path=${encodeURIComponent(conceptPath)}`,
    ),

  imageUrl: (path: string) => `${API_BASE}/dpo/saturation/image?path=${encodeURIComponent(path)}`,
};
