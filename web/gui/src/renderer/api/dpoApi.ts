import { API_BASE, request } from "./request";

/**
 * Where the realism head runs. "auto" prefers CUDA; "cpu" pins it to the
 * processor and system RAM, which is slower but leaves the GPU alone; "off"
 * disables scoring entirely, so the model is never loaded and groups keep
 * their source order.
 */
export type RealismDevice = "auto" | "cpu" | "off";

/** One proposition of the caption, and how strongly the image supports it. */
export interface ScoreProposition {
  text: string;
  /** Vocabulary axis ("shot_type", "vertical_angle", ..., "age_model", "text_ocr") or null for a free phrase. */
  axis: string | null;
  label: string | null;
  prob: number;
  estimated_age?: number;
  p_none?: number;
  /** For "text_ocr": what the OCR reader saw for this span, null when nothing resembled it. */
  read?: string | null;
  /** For "text_ocr": "verbatim", "wrong" or "absent". `prob` is the policy score (absent sits above wrong). */
  verdict?: "verbatim" | "wrong" | "absent";
  /** For "text_ocr": the raw similarity of the reading, for display. */
  ratio?: number;
}

/**
 * Every scorer's verdict on one image of the current group. `null` where a
 * scorer had nothing to say (no model, or — for age and text — a caption that
 * claims no age / mentions no on-image text). `realism` is P(AI-generated),
 * lower is better; the rest are higher-is-better.
 */
export interface ScoreBreakdown {
  realism: number | null;
  adherence: number | null;
  age: number | null;
  estimated_age: number | null;
  /** OCR match (0–1) against the text the caption asks for; the worst span is `text_min`. */
  text: number | null;
  text_min?: number | null;
  /** The reader's full transcription of the image, for the tooltip. */
  ocr_text?: string | null;
  quality: number | null;
  axis_mean: number | null;
  phrase_mean: number | null;
  /** Weighted z under the server's default weights; the UI recomputes it from the sliders. */
  combined: number | null;
  /** 1-based position under the gate-then-rank rule. */
  gate_rank: number;
  propositions: ScoreProposition[];
}

/** Presentation order of scanned groups; the realism orders rank on P(AI). */
export type GroupOrder = "random" | "least_realistic" | "most_realistic" | "largest_gap";

export interface PairCheckResult {
  total_matched: number;
  total_chosen_stray: number;
  total_rejected_stray: number;
  format_stats: Record<string, number>;
  multiline_captions: number;
  pairs: Array<{
    chosen_path: string;
    rejected_path: string;
    matched: number;
    chosen_stray: number;
    rejected_stray: number;
  }>;
}

export interface ReviewPair {
  chosen_path: string;
  rejected_path: string;
  caption: string;
  prompt_key: string;
}

export interface CaptionMismatch {
  concept_pair_index: number;
  key: string;
  chosen_image: string;
  rejected_image: string;
  chosen_caption_path: string | null;
  rejected_caption_path: string | null;
  chosen_caption: string;
  rejected_caption: string;
}

export interface SessionStatus {
  active: boolean;
  scan_count: number;
  scan_total: number;
  hash_count: number;
  cache_hits: number;
  groups_queued: number;
  groups_shown: number;
  worker_finished: boolean;
  has_group: boolean;
  total_pairs: number;
}

export interface GroupData {
  prompt: string;
  aspectratio: string;
  images: string[];
  group_index: number;
  total_groups: number;
  pairs_done: number;
  pairs_target: number;
  mode?: "selection" | "swiss" | "triage";
}

export type SwissScores = Record<string, { score: number; elo: number }>;

export interface SwissStateResponse {
  ok: boolean;
  finished?: boolean;
  match?: [string, string] | null;
  round?: number;
  total_rounds?: number;
  matches_played?: number;
  matches_total?: number;
  scores?: SwissScores;
  error?: string;
}

export interface SwissRankingResponse {
  ok: boolean;
  order?: string[];
  scores?: SwissScores;
  max_pairs?: number;
  default_pairs?: number;
  error?: string;
}

export interface NextGroupResponse {
  group?: GroupData;
  done?: boolean;
  waiting?: boolean;
  total_pairs?: number;
}

export interface SelectResponse {
  ok: boolean;
  error?: string;
  phase?: string;
  best?: string;
  // pair_pending: user picked best+worst but backend held the write so the UI
  // can offer Confirm / Pick More / Cancel. Resolve with confirmPair/cancelPair.
  pair_pending?: boolean;
  // pair_created: backend auto-committed (final pair in the group — no Cancel
  // option would be meaningful, matches Ctk).
  pair_created?: boolean;
  chosen?: string;
  rejected?: string;
  continue_group?: boolean;
  remaining_images?: string[];
  pairs_done?: number;
}

export interface ConfirmPairResponse {
  ok: boolean;
  error?: string;
  committed?: boolean;
  continue_group?: boolean;
  remaining_images?: string[];
  pairs_done?: number;
  discarded?: number;
}

export interface CancelPairResponse {
  ok: boolean;
  error?: string;
  phase?: string;
  best?: string;
  remaining_images?: string[];
}

export interface BucketAnalysisRow {
  h: number;
  w: number;
  count: number;
  drops: number;
  add: number;
  remove: number;
  aspect_label: string;
}

export interface BucketAnalysisTarget {
  target: number;
  total_pairs: number;
  total_drops: number;
  total_add: number;
  total_remove: number;
  buckets: BucketAnalysisRow[];
}

export interface RepairRejectedResult {
  groups_processed: number;
  groups_skipped_single: number;
  pairs_repaired: number;
  pairs_total: number;
}

export interface BucketAnalysisResult {
  concept_path: string;
  batch_size: number;
  quantization: number;
  scanned: number;
  unreadable: number;
  targets: BucketAnalysisTarget[];
}

/** One `<segment:…>` fix for Fix & Regenerate. */
export interface RegenFix {
  target: string;
  creativity: number;
  threshold: number;
  sub_prompt: string;
}

/** SwarmUI's "Segment Refining" group by API id; the server coerces values. */
export type RegenSegmentParams = Record<string, string | number | boolean>;

export interface RegenPreviewItem {
  path: string;
  ok: boolean;
  error: string | null;
  warnings: string[];
  original_prompt: string;
  prompt: string;
  seed: number | null;
  model: string;
  width: number | null;
  height: number | null;
  steps: number | null;
  cfgscale: number | null;
  sampler: string | null;
  scheduler: string | null;
  loras: string[];
  negativeprompt: string;
}

export interface RegenPreviewResponse {
  ok: boolean;
  error?: string;
  group_index?: number;
  items?: RegenPreviewItem[];
  segment_params?: RegenSegmentParams;
}

export interface RegenItemStatus {
  status: "queued" | "running" | "done" | "failed" | "dropped" | "skipped";
  output: string | null;
  error: string | null;
  prompt: string;
}

export interface RegenStatus {
  ok: boolean;
  running: boolean;
  phase: "idle" | "generating" | "done" | "error" | "cancelled";
  group_index: number;
  planned: number;
  done: number;
  failed: number;
  items: Record<string, RegenItemStatus>;
  added: string[];
  dropped: string[];
  error: string | null;
  warning: string | null;
}

export interface RegenStartOptions {
  paths: string[];
  fixes: RegenFix[];
  segment_params: RegenSegmentParams;
  model_override: string;
  base_url: string;
  token: string;
  generated_folder: string;
  concurrency: number;
}

export const dpoApi = {
  checkPairs: () =>
    request<{ ok: boolean; result?: PairCheckResult; error?: string }>("/dpo/check-pairs", {
      method: "POST",
    }),

  removeStrays: () =>
    request<{ ok: boolean; removed?: number }>("/dpo/remove-strays", {
      method: "POST",
    }),

  reviewPairs: () => request<{ ok: boolean; pairs?: ReviewPair[]; error?: string }>("/dpo/review"),

  removePair: (chosenPath: string | null, rejectedPath: string | null) =>
    request<{ ok: boolean }>("/dpo/remove-pair", {
      method: "POST",
      body: JSON.stringify({ chosen_path: chosenPath, rejected_path: rejectedPath }),
    }),

  fixCaptions: () =>
    request<{ ok: boolean; fixed?: number }>("/dpo/fix-captions", {
      method: "POST",
    }),

  checkCaptionMismatches: () =>
    request<{ ok: boolean; mismatches?: CaptionMismatch[]; error?: string }>("/dpo/check-caption-mismatches", {
      method: "POST",
    }),

  correctAllToChosen: () =>
    request<{ ok: boolean; corrected?: number; error?: string }>("/dpo/correct-all-to-chosen", {
      method: "POST",
    }),

  applyCaption: (chosenImage: string, rejectedImage: string, caption: string) =>
    request<{ ok: boolean; error?: string }>("/dpo/apply-caption", {
      method: "POST",
      body: JSON.stringify({
        chosen_image: chosenImage,
        rejected_image: rejectedImage,
        caption,
      }),
    }),

  // Curation session
  startSession: (
    sourceFolder: string,
    outputDir: string,
    pairsPerGroup = 1,
    mode: "selection" | "swiss" | "triage" = "selection",
    groupOrder: GroupOrder = "most_realistic",
    captionFilter = "",
    minImagesPerGroup = 2,
    loraFilter = "",
  ) =>
    request<{ ok: boolean; existing_pairs?: number; pruned?: number; error?: string }>("/dpo/session/start", {
      method: "POST",
      body: JSON.stringify({
        source_folder: sourceFolder,
        output_dir: outputDir,
        pairs_per_group: pairsPerGroup,
        mode,
        group_order: groupOrder,
        caption_filter: captionFilter,
        lora_filter: loraFilter,
        min_images_per_group: minImagesPerGroup,
      }),
    }),

  sessionStatus: () => request<SessionStatus>("/dpo/session/status"),

  nextGroup: () => request<NextGroupResponse>("/dpo/session/next-group", { method: "POST" }),

  selectImage: (path: string) =>
    request<SelectResponse>("/dpo/session/select", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  // Fix & Regenerate: what would be resent for each selected image (composed
  // prompt, seed, model, size), and which images carry no parameter block.
  regenPreview: (paths: string[], fixes: RegenFix[], segmentParams: RegenSegmentParams, modelOverride = "") =>
    request<RegenPreviewResponse>("/dpo/session/regen/preview", {
      method: "POST",
      body: JSON.stringify({ paths, fixes, segment_params: segmentParams, model_override: modelOverride }),
    }),

  // Starts the background run; poll regenStatus. Results are appended to the
  // current group as they land.
  regenStart: (options: RegenStartOptions) =>
    request<{ ok: boolean; error?: string; planned?: number; group_index?: number }>("/dpo/session/regen/start", {
      method: "POST",
      body: JSON.stringify(options),
    }),

  regenStatus: () => request<RegenStatus>("/dpo/session/regen/status"),

  regenCancel: () => request<{ ok: boolean }>("/dpo/session/regen/cancel", { method: "POST" }),

  // discardRest commits the pair and then retires the whole group from the
  // source tree (paired sources → .chosen/.rejected, leftovers → .discard).
  confirmPair: (continueScoring: boolean, discardRest = false) =>
    request<ConfirmPairResponse>("/dpo/session/confirm-pair", {
      method: "POST",
      body: JSON.stringify({ continue_scoring: continueScoring, discard_rest: discardRest }),
    }),

  cancelPair: () => request<CancelPairResponse>("/dpo/session/cancel-pair", { method: "POST" }),

  skipGroup: () => request<{ ok: boolean }>("/dpo/session/skip-group", { method: "POST" }),

  // Triage mode: all voting/pairing happens client-side; this commits the
  // whole group's pairs in one batch and releases the group.
  // discardRest retires the whole group from the source tree after export:
  // paired sources move to .chosen/.rejected, leftovers to .discard (all
  // dot-pruned so the group can't reappear in a later session).
  commitTriagePairs: (pairs: Array<{ chosen: string; rejected: string }>, discardRest = false) =>
    request<{ ok: boolean; pairs_done?: number; discarded?: number; error?: string }>("/dpo/session/commit-pairs", {
      method: "POST",
      body: JSON.stringify({ pairs, discard_rest: discardRest }),
    }),

  // Discard the whole current group without saving any pair — moves every
  // remaining source image into a dot-prefixed .discard subfolder.
  discardGroup: () =>
    request<{ ok: boolean; discarded?: number; error?: string }>("/dpo/session/discard-group", { method: "POST" }),

  triageAlign: (good: string[], bad: string[]) =>
    request<{
      ok: boolean;
      pairs?: Array<{ chosen: string; rejected: string }>;
      chosen_pool?: string[];
      rejected_pool?: string[];
      error?: string;
    }>("/dpo/session/triage-align", {
      method: "POST",
      body: JSON.stringify({ good, bad }),
    }),

  // P(AI-generated) per image, used to sort a group most-photorealistic-first.
  // Backed by the session scan cache, so a group costs the model once and is
  // free on every revisit.
  realismScores: (paths: string[]) =>
    request<{
      ok: boolean;
      scores?: Record<string, number>;
      // Set when scoring is turned off: an empty result that is not a failure.
      disabled?: boolean;
      error?: string;
    }>("/dpo/session/realism-scores", {
      method: "POST",
      body: JSON.stringify({ paths }),
    }),

  // Every scorer's verdict per image of the current group — realism, prompt
  // adherence with its propositions, age agreement, on-image text (OCR),
  // scalar quality — for the per-tile breakdown. Scorers fail one at a time
  // (see `errors`).
  scoreBreakdown: (paths: string[]) =>
    request<{
      ok: boolean;
      scores?: Record<string, ScoreBreakdown>;
      errors?: Record<string, string>;
      default_weights?: Record<string, number>;
      disabled?: boolean;
      error?: string;
    }>("/dpo/session/score-breakdown", {
      method: "POST",
      body: JSON.stringify({ paths }),
    }),

  // Score the whole source tree on every axis once, ahead of curation, so
  // Selection never loads a model. Returns immediately — poll
  // scoreBackfillStatus. Re-running is cheap: anything already cached under
  // the current model versions is a hit, so a second pass only costs newly
  // added images.
  startScoreBackfill: () =>
    request<{ ok: boolean; total?: number; error?: string }>("/dpo/session/score-backfill", {
      method: "POST",
    }),

  scoreBackfillStatus: () =>
    request<{
      ok: boolean;
      running: boolean;
      done: number;
      total: number;
      // Images each axis has covered; the caption decides whether the age,
      // adherence and text axes apply to an image at all.
      axes?: Record<string, number>;
      // Per-axis load failures — that axis is skipped, the others still run.
      errors?: Record<string, string>;
      finished: boolean;
      device?: RealismDevice;
      error?: string | null;
    }>("/dpo/session/score-backfill/status"),

  stopScoreBackfill: () => request<{ ok: boolean }>("/dpo/session/score-backfill/stop", { method: "POST" }),

  // Where the realism head runs. "cpu" keeps it off the GPU entirely, for when
  // training or generation is using the VRAM; "auto" takes CUDA when it can.
  // Not session-scoped — readable and settable before curation starts.
  realismDevice: () => request<{ ok: boolean; device: RealismDevice }>("/dpo/realism/device"),

  setRealismDevice: (device: RealismDevice) =>
    request<{ ok: boolean; device: RealismDevice; error?: string }>("/dpo/realism/device", {
      method: "POST",
      body: JSON.stringify({ device }),
    }),

  finalizeSession: (valPercentage = 0) =>
    request<{ ok: boolean; total_pairs?: number; train_count?: number; val_count?: number }>("/dpo/session/finalize", {
      method: "POST",
      body: JSON.stringify({ val_percentage: valPercentage }),
    }),

  cancelSession: () => request<{ ok: boolean }>("/dpo/session/cancel", { method: "POST" }),

  // Swiss tournament mode
  swissState: () => request<SwissStateResponse>("/dpo/session/swiss-state"),

  swissVote: (a: string, b: string, winner: "a" | "b" | "tie") =>
    request<SwissStateResponse>("/dpo/session/swiss-vote", {
      method: "POST",
      body: JSON.stringify({ a, b, winner }),
    }),

  swissFinishEarly: () => request<SwissRankingResponse>("/dpo/session/swiss-finish-early", { method: "POST" }),

  swissRanking: () => request<SwissRankingResponse>("/dpo/session/swiss-ranking"),

  swissReorder: (order: string[]) =>
    request<{ ok: boolean; error?: string }>("/dpo/session/swiss-reorder", {
      method: "POST",
      body: JSON.stringify({ order }),
    }),

  // discardRest additionally retires the group from the source tree after the
  // export: exported sources → .chosen/.rejected, mid-ranked leftovers →
  // .discard (all dot-pruned so the group can't reappear).
  swissExport: (pairCount: number, discardRest = false) =>
    request<{ ok: boolean; exported?: number; pairs_done?: number; discarded?: number; error?: string }>(
      "/dpo/session/swiss-export",
      {
        method: "POST",
        body: JSON.stringify({ pair_count: pairCount, discard_rest: discardRest }),
      },
    ),

  bucketAnalysis: (concept_path: string, batch_size: number, target_resolutions: number[], quantization: number) =>
    request<{ ok: boolean; result?: BucketAnalysisResult; error?: string }>("/dpo/bucket-analysis", {
      method: "POST",
      body: JSON.stringify({ concept_path, batch_size, target_resolutions, quantization }),
    }),

  // Starts a background re-pair job; poll repairStatus for progress + result.
  repairRejected: (qualityFloor = 0.85) =>
    request<{ ok: boolean; started?: boolean; running?: boolean; error?: string }>("/dpo/repair-rejected", {
      method: "POST",
      body: JSON.stringify({ quality_floor: qualityFloor }),
    }),

  repairStatus: () =>
    request<{
      ok: boolean;
      running: boolean;
      phase?: string;
      images_done?: number;
      images_total?: number;
      groups_done?: number;
      groups_total?: number;
      pairs_repaired?: number;
      summary?: RepairRejectedResult | null;
      error?: string | null;
    }>("/dpo/repair-status"),

  // Absolute against API_BASE: under Electron the page is file://, so a bare
  // "/api/..." would resolve to file:///api/... and render a broken image.
  imageUrl: (path: string) => `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`,
};

// ---- Off-policy DPO builder: real photo (chosen) vs own generation (rejected) ----

export interface OffPolicyStatus {
  active: boolean;
  phase: "idle" | "indexing" | "scanning" | "matching" | "scoring" | "ready" | "error";
  progress_done: number;
  progress_total: number;
  worker_finished: boolean;
  error: string | null;
  warning: string | null;
  // Candidates whose generations had no cached realism score (random pick, shown UNSCORED).
  unscored: number;
  photos: number;
  generated: number;
  excluded_captions: number;
  // Photos that matched a generation on caption and aspect but were smaller
  // than every one of them (a smaller photo would be upscaled relative to the
  // generation, teaching a resampling cue instead of realism).
  too_small: number;
  cache_imported: number;
  // Files listed so far while indexing (the walk phase has no other progress).
  listed: number;
  can_undo: boolean;
  undo_action: "confirm" | "skip" | "skip_forever" | "caption" | null;
  candidates: number;
  ready: number;
  shown: number;
  confirmed: number;
  skipped: number;
  // Caption lines removed this session (photo kept).
  captions_removed: number;
  // Photos permanently skipped this session, and the size of the saved list.
  skipped_forever: number;
  permanently_skipped: number;
  total_pairs: number;
  has_candidate: boolean;
}

export interface OffPolicyCandidate {
  real_path: string;
  caption_path: string;
  real_size: [number, number];
  caption: string;
  caption_lines: string[];
  generated_count: number;
  // How many of this photo's generations have been shown (re-roll history).
  shown_count: number;
  rejected_path: string | null;
  rejected_size: [number, number] | null;
  rejected_score: number | null;
  // P(AI) of the real photo itself, when scored — a photo the head trusts
  // less than the generation deserves a second look before it becomes chosen.
  real_score: number | null;
  scored: boolean;
  aspect_mismatch?: number | null;
  aspect_bucket: string;
  index: number;
  total: number;
}

export interface OffPolicySuggestion {
  real_path: string;
  caption: string;
  caption_lines: string[];
  real_size: [number, number];
  /** Trainer bucket the photo trains in. */
  aspect_bucket: string;
  /** Nearest ratio the generator can actually produce ("3:2", "9:16", ...). */
  generate_aspect: string;
  generate_size: [number, number];
  /** Aspect gap left between the photo and that generatable ratio. */
  aspect_delta: number | null;
  /** False when that gap exceeds the session's aspect tolerance. */
  within_tolerance: boolean;
}

/** Per trainer bucket: photos still to generate for, and the one size to batch them at. */
export type OffPolicyBuckets = Record<string, { photos: number; generate_size: [number, number] }>;

export interface OffPolicyPhotoBrief {
  real_path: string;
  real_size: [number, number];
  aspect_bucket: string;
  generate_aspect: string;
  generate_size: [number, number];
  aspect_delta: number | null;
  caption_lines: string[];
  lines_needing_generation: string[];
  paired: boolean;
  skipped: boolean;
  has_usable_generation: boolean;
  too_small_for_suggested_size: boolean;
}

export const dpoOffPolicyApi = {
  start: (
    generatedFolder: string,
    realFolder: string,
    outputDir: string,
    aspectTolerance = 0.15,
    baseResolution = 1024,
    maxUpscale = 1.1,
  ) =>
    request<{
      ok: boolean;
      existing_pairs?: number;
      pruned?: number;
      cache_imported?: number;
      permanently_skipped?: number;
      error?: string;
    }>("/dpo/offpolicy/start", {
      method: "POST",
      body: JSON.stringify({
        generated_folder: generatedFolder,
        real_folder: realFolder,
        output_dir: outputDir,
        aspect_tolerance: aspectTolerance,
        max_upscale: maxUpscale,
        base_resolution: baseResolution,
      }),
    }),

  status: () => request<OffPolicyStatus>("/dpo/offpolicy/status"),

  next: () =>
    request<{ ok: boolean; candidate?: OffPolicyCandidate; waiting?: boolean; done?: boolean; error?: string }>(
      "/dpo/offpolicy/next",
      { method: "POST" },
    ),

  // The candidate currently on screen (null when none), for a reloaded UI.
  current: () =>
    request<{ ok: boolean; candidate?: OffPolicyCandidate | null; error?: string }>("/dpo/offpolicy/current"),

  confirm: () =>
    request<{ ok: boolean; total_pairs?: number; error?: string }>("/dpo/offpolicy/confirm", { method: "POST" }),

  skip: () => request<{ ok: boolean; error?: string }>("/dpo/offpolicy/skip", { method: "POST" }),

  // Never offer this photo again for this output folder (saved to disk).
  skipForever: () =>
    request<{ ok: boolean; permanently_skipped?: number; error?: string }>("/dpo/offpolicy/skip-forever", {
      method: "POST",
    }),

  // Swap the rejected generation for another of the same photo's.
  reroll: () =>
    request<{ ok: boolean; candidate?: OffPolicyCandidate; error?: string }>("/dpo/offpolicy/reroll", {
      method: "POST",
    }),

  // Reverses the last confirm / skip / delete and returns that candidate.
  undo: () =>
    request<{ ok: boolean; undone?: string; candidate?: OffPolicyCandidate; error?: string }>("/dpo/offpolicy/undo", {
      method: "POST",
    }),

  // Removes the caption line on screen from the photo's .txt (the photo and
  // its other lines stay; an emptied .txt is deleted). Undoable in-session.
  removeCaption: () =>
    request<{ ok: boolean; removed_lines?: number; removed_file?: boolean; remaining_lines?: number; error?: string }>(
      "/dpo/offpolicy/remove-caption",
      { method: "POST" },
    ),

  // Random captions that have no usable generation yet, with the size to
  // generate each at; optionally only photos in one trainer bucket ("3:2").
  suggest: (aspectBucket: string | null = null, count = 1, keywords = "") =>
    request<{
      ok: boolean;
      suggestions?: OffPolicySuggestion[];
      remaining?: number;
      aspect_bucket?: string | null;
      buckets?: OffPolicyBuckets;
      waiting?: boolean;
      error?: string;
    }>("/dpo/offpolicy/suggest", {
      method: "POST",
      body: JSON.stringify({ aspect_bucket: aspectBucket, count, keywords }),
    }),

  buckets: (keywords = "") =>
    request<{ ok: boolean; buckets?: OffPolicyBuckets; remaining?: number; waiting?: boolean; error?: string }>(
      `/dpo/offpolicy/buckets${keywords ? `?keywords=${encodeURIComponent(keywords)}` : ""}`,
    ),

  // Photos whose file name matches, with the size to generate at. Matched
  // against the in-memory index by string only — never resolved on disk.
  lookup: (filename: string) =>
    request<{
      ok: boolean;
      matches?: OffPolicyPhotoBrief[];
      generations_scanned?: boolean;
      waiting?: boolean;
      error?: string;
    }>("/dpo/offpolicy/lookup", {
      method: "POST",
      body: JSON.stringify({ filename }),
    }),

  cancel: () => request<{ ok: boolean }>("/dpo/offpolicy/cancel", { method: "POST" }),
};
