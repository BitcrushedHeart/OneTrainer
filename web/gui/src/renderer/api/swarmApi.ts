import { request } from "./request";

/** One planned or completed generation. */
export interface SwarmResult {
  caption: string;
  real_path: string | null;
  size: [number, number];
  images: number;
  caption_type: string;
  ok?: boolean;
  error?: string;
  paths?: string[];
  planned_only?: boolean;
}

/** A queued run, as shown in the queue list. */
export interface SwarmJobSummary {
  keywords: string;
  captions: number;
  images_per_caption: number;
  dry_run: boolean;
}

export interface SwarmStatus {
  ok: boolean;
  running: boolean;
  phase:
    | "idle"
    | "starting"
    | "indexing"
    | "scanning"
    | "planning"
    | "generating"
    | "done"
    | "error"
    | "cancelling"
    | "cancelled";
  /** Photos still without a usable generation, after the keyword filter. */
  pool: number;
  planned: number;
  done: number;
  images: number;
  failed: number;
  skipped_tag_style: number;
  skipped_short: number;
  /** The parameters the run actually used, and where they came from. */
  params: Record<string, unknown>;
  param_source: string;
  /** Which queued run is executing, and what is still waiting behind it. */
  job_index: number;
  job_total: number;
  current_job: SwarmJobSummary | null;
  queued: SwarmJobSummary[];
  error: string | null;
  dry_run: boolean;
  results: SwarmResult[];
}

/** One LoRA applied to every generation in a run. */
export interface SwarmLora {
  name: string;
  weight: number;
}

export interface SwarmGenerateOptions {
  real_folder: string;
  generated_folder: string;
  output_dir: string;
  model?: string;
  base_url?: string;
  token?: string;
  captions?: number;
  images_per_caption?: number;
  keywords?: string;
  exclude_tag_style?: boolean;
  min_caption_words?: number;
  steps?: number | null;
  cfg_scale?: number | null;
  negative_prompt?: string;
  loras?: SwarmLora[];
  /** Where the generation settings come from. */
  param_source?: "last" | "preset" | "manual";
  preset_name?: string;
  /** Requests kept in flight so the GPU never waits for the next one. */
  concurrency?: number;
  dry_run?: boolean;
}

export const swarmApi = {
  // Confirms SwarmUI answers and lists the models it can load.
  check: (baseUrl: string, token = "") =>
    request<{
      ok: boolean;
      version?: string;
      user_id?: string;
      models?: string[];
      /** Every LoRA SwarmUI can apply. */
      loras?: string[];
      /** Checkpoints it currently has loaded — a sensible default model. */
      loaded?: string[];
      /** Saved SwarmUI presets, usable as a settings source. */
      presets?: string[];
      error?: string;
    }>("/swarm/check", {
      method: "POST",
      body: JSON.stringify({ base_url: baseUrl, token }),
    }),

  // Starts a background run; poll status(). dry_run plans without generating.
  generate: (options: SwarmGenerateOptions) =>
    request<{ ok: boolean; error?: string }>("/swarm/generate", {
      method: "POST",
      body: JSON.stringify(options),
    }),

  status: () => request<SwarmStatus>("/swarm/status"),

  // Stops the current run and drops everything queued behind it.
  cancel: () => request<{ ok: boolean; dropped?: number }>("/swarm/cancel", { method: "POST" }),

  // Drops the waiting runs; the one in flight finishes.
  clearQueue: () => request<{ ok: boolean; dropped?: number }>("/swarm/queue/clear", { method: "POST" }),

  // What "reuse my last generation" would apply, without generating.
  lastParams: (generatedFolder: string) =>
    request<{ ok: boolean; params?: Record<string, unknown>; error?: string }>(
      `/swarm/last-params?generated_folder=${encodeURIComponent(generatedFolder)}`,
    ),

  // The trainer bucket sizes generations are requested at.
  buckets: () => request<{ ok: boolean; buckets: Array<[number, number]> }>("/swarm/buckets"),
};
