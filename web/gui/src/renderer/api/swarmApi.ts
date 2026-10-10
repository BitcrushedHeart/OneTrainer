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
  /** Set when this is a fresh generation of a black (failed) image. */
  requeued_from?: string | null;
  /** Outputs of this job that came back black and were retired to .black. */
  black?: number;
}

/** A queued run, as shown in the queue list. */
export interface SwarmJobSummary {
  /** An ordinary caption run, or a requeue of black images. */
  mode: "generate" | "requeue";
  keywords: string;
  captions: number;
  images_per_caption: number;
  /** Requeue only: how many images are resent. */
  images: number;
  dry_run: boolean;
}

/** One black image found in the generated tree. */
export interface SwarmBlackImage {
  path: string;
  prompt: string;
  model: string;
  seed: number | string | null;
  size: [number, number] | null;
  /** Without a SwarmUI parameter block the image can be retired but not resent. */
  has_params: boolean;
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
  /** Black images found at the start of a run and retired to .black. */
  black_retired: number;
  /** In-run black outputs that were retired and retried with a fresh seed. */
  black_retried: number;
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
  /** Bulk: one queued run per entry ("a+b" = both required); replaces `keywords`. */
  keyword_jobs?: string[];
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
  /** Retire and regenerate black images before the new captions; retry in-run black outputs. */
  auto_requeue_black?: boolean;
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
      /** Detection models usable as a `<segment:yolo-NAME>` target. */
      yolo?: string[];
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

  // Black images in the generated tree: generations that failed inside
  // SwarmUI's backend but still claim their caption.
  scanBlack: (generatedFolder: string) =>
    request<{ ok: boolean; count?: number; images?: SwarmBlackImage[]; error?: string }>("/swarm/black/scan", {
      method: "POST",
      body: JSON.stringify({ generated_folder: generatedFolder }),
    }),

  // Moves the black images to .black and queues a fresh generation of each
  // from its own recorded parameters. Empty paths = every black image.
  requeueBlack: (options: {
    generated_folder: string;
    base_url?: string;
    token?: string;
    model?: string;
    concurrency?: number;
    paths?: string[];
  }) =>
    request<{
      ok: boolean;
      queued?: boolean;
      found?: number;
      moved?: number;
      requeued?: number;
      unresendable?: Array<{ path: string; error: string }>;
      error?: string;
    }>("/swarm/black/requeue", {
      method: "POST",
      body: JSON.stringify(options),
    }),
};
