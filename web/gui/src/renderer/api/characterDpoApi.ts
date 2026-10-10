import { type OffPolicyCandidate } from "./dpoApi";
import { API_BASE, request } from "./request";

// Backend routes live under /api/dpo/character (see the Character DPO router).

export interface CharacterSplitCounts {
  train: number;
  val: number;
  unsplit: number;
  missing_masks: number;
}

export interface CharacterLora {
  path: string | null;
  kind: "best" | "second" | null;
  /** Name the LoRA is (or would be) published under in SwarmUI. */
  swarm_name: string | null;
  /** Which of the LoRA folders it was found in (null when none). */
  dir?: string | null;
}

export interface Character {
  name: string;
  path: string;
  train_count: number;
  val_count: number;
  generated_count: number;
  offpolicy: CharacterSplitCounts;
  aivai: CharacterSplitCounts;
  lora: CharacterLora;
  /** Queue entry id when this character already has one, else null. */
  queue_entry: string | null;
  finalized: boolean;
  concepts_file: string | null;
}

export interface CharacterListResponse {
  root: string;
  models_dir: string;
  characters: Character[];
}

export interface PublishLoraResponse {
  ok: boolean;
  path?: string;
  /** The LoRA already lives in the SwarmUI folder — nothing was copied. */
  unchanged?: boolean;
  error?: string;
}

export interface ImportCandidate {
  key: string;
  concept_dir: string;
  concept_name: string;
  pair_id: string;
  kind: "offpolicy" | "aivai";
  match: "name" | "lora" | "both";
  prompt: string;
  caption: string;
  chosen_path: string;
  rejected_path: string;
  already_imported: boolean;
}

export interface ImportScanResponse {
  candidates: ImportCandidate[];
}

export interface ImportApplyResponse {
  ok: boolean;
  imported: number;
  skipped: number;
  errors: string[];
}

export interface FinalizeStartResponse {
  ok: boolean;
  job_id?: string;
  error?: string;
}

export interface FinalizeStatus {
  running: boolean;
  stage: string;
  done: number;
  total: number;
  message: string;
  error: string | null;
  result: Record<string, unknown> | null;
}

export type QueueBuildMode = "preview" | "add" | "export";

export interface QueueBuildEntry {
  name: string;
  overrides: Record<string, unknown>;
  warnings: string[];
}

export interface QueueBuildResponse {
  entries: QueueBuildEntry[];
  errors: Array<{ name: string; error: string }>;
  added: number;
}

/** One character's row of a `POST /dpo/character/generate` call. */
export interface GenerateJob {
  name: string;
  ok: boolean;
  error?: string;
  /** Position in the SwarmUI run queue (1 = running next / now). */
  position?: number;
  /** The SwarmUI LoRA name the run was queued with. */
  swarm_lora?: string;
}

export interface GenerateResponse {
  ok: boolean;
  jobs: GenerateJob[];
  error?: string;
}

export interface PrepareStartResponse {
  ok: boolean;
  job_id?: string;
  error?: string;
}

/** Per-character outcome of a prepare (batch finalize) run, as the UI shows it. */
export interface PrepareResult {
  name: string;
  ok: boolean;
  skipped?: boolean;
  error?: string | null;
  counts?: Record<string, unknown>;
  message?: string;
}

/**
 * What the backend stores per character: the finalize result
 * (`{offpolicy: {train, val}, aivai: {train, val}, masks_generated, masks_copied, concepts_file}`),
 * `{error}` or `{skipped: true}`.
 */
export type PrepareRawResult = Record<string, unknown>;

export interface PrepareStatusRaw {
  running: boolean;
  /** Character being finalized right now (null/"" when idle). */
  current: string | null;
  done: number;
  total: number;
  stage: string;
  message: string;
  error: string | null;
  results: Record<string, PrepareRawResult>;
}

export interface PrepareStatus extends Omit<PrepareStatusRaw, "results" | "current"> {
  current: string;
  results: PrepareResult[];
}

function flattenCounts(raw: PrepareRawResult): Record<string, unknown> {
  const counts: Record<string, unknown> = {};
  for (const folder of ["offpolicy", "aivai"]) {
    const split = raw[folder];
    if (split && typeof split === "object") {
      for (const [k, v] of Object.entries(split as Record<string, unknown>)) counts[`${folder} ${k}`] = v;
    }
  }
  for (const key of ["masks_generated", "masks_copied"]) {
    if (raw[key] !== undefined) counts[key.replace("_", " ")] = raw[key];
  }
  return counts;
}

/** Turns the backend's name-keyed results map into the list the UI renders. */
export function normalizePrepareStatus(raw: PrepareStatusRaw): PrepareStatus {
  const results: PrepareResult[] = Object.entries(raw.results ?? {}).map(([name, r]) => {
    if (r.skipped) return { name, ok: true, skipped: true, message: "already finalized" };
    if (typeof r.error === "string") return { name, ok: false, error: r.error };
    return { name, ok: true, counts: flattenCounts(r) };
  });
  return { ...raw, current: raw.current ?? "", results };
}

/**
 * The off-policy candidate as the embedded picker sees it: every generation
 * of the photo plus which one is the rejected side right now.
 */
export interface OffPolicyPickCandidate extends OffPolicyCandidate {
  generated_paths: string[];
  rejected_index: number;
}

export interface OffPolicyStartBody {
  generated_folder: string;
  real_folder: string;
  output_dir: string;
  aspect_tolerance: number;
  max_upscale: number;
  base_resolution: number;
  order: "random" | "path";
}

const BASE = "/dpo/character";

function query(params: Record<string, string | undefined>): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "") qs.set(k, v);
  }
  const s = qs.toString();
  return s ? `?${s}` : "";
}

export const characterDpoApi = {
  list: (opts: { root: string; modelsDir?: string; swarmUrl?: string; swarmToken?: string }) =>
    request<CharacterListResponse>(
      `${BASE}/list${query({
        root: opts.root,
        models_dir: opts.modelsDir,
        swarm_url: opts.swarmUrl,
        swarm_token: opts.swarmToken,
      })}`,
    ),

  publishLora: (body: { name: string; root: string; models_dir?: string; swarm_lora_dir: string; force?: boolean }) =>
    request<PublishLoraResponse>(`${BASE}/lora/publish`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  importScan: (body: { name: string; root: string; concept_files: string[]; models_dir?: string }) =>
    request<ImportScanResponse>(`${BASE}/import/scan`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  importApply: (body: { name: string; root: string; picks: Array<{ concept_dir: string; pair_id: string }> }) =>
    request<ImportApplyResponse>(`${BASE}/import/apply`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  finalize: (body: { name: string; root: string; val_pct: number; mask_model: string; dpo_resolution?: string }) =>
    request<FinalizeStartResponse>(`${BASE}/finalize`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  finalizeStatus: (name: string, root: string) =>
    request<FinalizeStatus>(`${BASE}/finalize/status${query({ name, root })}`),

  queueBuild: (body: {
    names: string[];
    root: string;
    models_dir?: string;
    add_on_weight: number;
    /** Starting DPO beta; the entry enables adaptive beta (bounded x1/4..x4 of this). */
    dpo_beta?: number;
    mode: QueueBuildMode;
  }) =>
    request<QueueBuildResponse>(`${BASE}/queue/build`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // ---- "Select characters and go" --------------------------------------

  /** Queues one SwarmUI run per character (its own LoRA, all its train captions). */
  generate: (body: {
    names: string[];
    root: string;
    models_dir?: string;
    swarm_url?: string;
    swarm_token?: string;
    swarm_lora_dir: string;
    param_folder: string;
    images_per_caption: number;
    /** 0 = every caption that still needs a generation. */
    captions: number;
  }) =>
    request<GenerateResponse>(`${BASE}/generate`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** How many train captions a character has (what a generate run would cover). */
  trainCaptions: (name: string, root: string) =>
    request<{ count: number }>(`${BASE}/train-captions${query({ name, root })}`),

  /** Finalizes several characters in one background job; 409 while one runs. */
  prepare: (body: {
    names: string[];
    root: string;
    models_dir?: string;
    val_pct: number;
    mask_model: string;
    /** Resolution cap for the DPO concepts (adaptive ladder snaps each pair below it). */
    dpo_resolution?: string;
  }) =>
    request<PrepareStartResponse>(`${BASE}/prepare`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  prepareStatus: async () => normalizePrepareStatus(await request<PrepareStatusRaw>(`${BASE}/prepare/status`)),

  /**
   * What "reuse my last generation" would apply from the user's Swarm output
   * folder, with the adapters stripped (each character brings its own LoRA).
   * `source_file` names the newest image the params were read from.
   */
  swarmLastParams: (generatedFolder: string) =>
    request<{ ok: boolean; params?: Record<string, unknown>; source_file?: string | null; error?: string }>(
      `/swarm/last-params${query({ generated_folder: generatedFolder, drop_loras: "true" })}`,
    ),

  // The off-policy session calls the embedded picker needs beyond
  // `dpoOffPolicyApi` (which has no `order` and no `choose`).
  offPolicyStart: (body: OffPolicyStartBody) =>
    request<{
      ok: boolean;
      existing_pairs?: number;
      cache_imported?: number;
      permanently_skipped?: number;
      error?: string;
    }>("/dpo/offpolicy/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Makes one of the current photo's generations the rejected side. */
  offPolicyChoose: (path: string) =>
    request<{ ok: boolean; candidate?: OffPolicyPickCandidate; error?: string }>("/dpo/offpolicy/choose", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  imageUrl: (path: string) => `${API_BASE}${BASE}/image?path=${encodeURIComponent(path)}`,
};
