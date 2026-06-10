import { request } from "./request";

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

export interface BucketAnalysisResult {
  concept_path: string;
  batch_size: number;
  quantization: number;
  scanned: number;
  unreadable: number;
  targets: BucketAnalysisTarget[];
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
  ) =>
    request<{ ok: boolean; existing_pairs?: number; pruned?: number; error?: string }>("/dpo/session/start", {
      method: "POST",
      body: JSON.stringify({
        source_folder: sourceFolder,
        output_dir: outputDir,
        pairs_per_group: pairsPerGroup,
        mode,
      }),
    }),

  sessionStatus: () => request<SessionStatus>("/dpo/session/status"),

  nextGroup: () => request<NextGroupResponse>("/dpo/session/next-group", { method: "POST" }),

  selectImage: (path: string) =>
    request<SelectResponse>("/dpo/session/select", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  confirmPair: (continueScoring: boolean) =>
    request<ConfirmPairResponse>("/dpo/session/confirm-pair", {
      method: "POST",
      body: JSON.stringify({ continue_scoring: continueScoring }),
    }),

  cancelPair: () => request<CancelPairResponse>("/dpo/session/cancel-pair", { method: "POST" }),

  skipGroup: () => request<{ ok: boolean }>("/dpo/session/skip-group", { method: "POST" }),

  // Triage mode: all voting/pairing happens client-side; this commits the
  // whole group's pairs in one batch and releases the group.
  commitTriagePairs: (pairs: Array<{ chosen: string; rejected: string }>) =>
    request<{ ok: boolean; pairs_done?: number; error?: string }>("/dpo/session/commit-pairs", {
      method: "POST",
      body: JSON.stringify({ pairs }),
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

  swissExport: (pairCount: number) =>
    request<{ ok: boolean; exported?: number; pairs_done?: number; error?: string }>("/dpo/session/swiss-export", {
      method: "POST",
      body: JSON.stringify({ pair_count: pairCount }),
    }),

  bucketAnalysis: (concept_path: string, batch_size: number, target_resolutions: number[], quantization: number) =>
    request<{ ok: boolean; result?: BucketAnalysisResult; error?: string }>("/dpo/bucket-analysis", {
      method: "POST",
      body: JSON.stringify({ concept_path, batch_size, target_resolutions, quantization }),
    }),

  imageUrl: (path: string) => `/api/dpo/session/image?path=${encodeURIComponent(path)}`,
};
