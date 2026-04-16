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

export interface SessionStatus {
  active: boolean;
  scan_count: number;
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
  pair_created?: boolean;
  chosen?: string;
  rejected?: string;
  continue_group?: boolean;
  remaining_images?: string[];
  pairs_done?: number;
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

  reviewPairs: () =>
    request<{ ok: boolean; pairs?: ReviewPair[]; error?: string }>("/dpo/review"),

  removePair: (chosenPath: string | null, rejectedPath: string | null) =>
    request<{ ok: boolean }>("/dpo/remove-pair", {
      method: "POST",
      body: JSON.stringify({ chosen_path: chosenPath, rejected_path: rejectedPath }),
    }),

  fixCaptions: () =>
    request<{ ok: boolean; fixed?: number }>("/dpo/fix-captions", {
      method: "POST",
    }),

  // Curation session
  startSession: (sourceFolder: string, outputDir: string, pairsPerGroup = 1) =>
    request<{ ok: boolean; existing_pairs?: number; pruned?: number; error?: string }>(
      "/dpo/session/start",
      { method: "POST", body: JSON.stringify({ source_folder: sourceFolder, output_dir: outputDir, pairs_per_group: pairsPerGroup }) },
    ),

  sessionStatus: () => request<SessionStatus>("/dpo/session/status"),

  nextGroup: () =>
    request<NextGroupResponse>("/dpo/session/next-group", { method: "POST" }),

  selectImage: (path: string) =>
    request<SelectResponse>("/dpo/session/select", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  skipGroup: () =>
    request<{ ok: boolean }>("/dpo/session/skip-group", { method: "POST" }),

  finalizeSession: (valPercentage = 0) =>
    request<{ ok: boolean; total_pairs?: number; train_count?: number; val_count?: number }>(
      "/dpo/session/finalize",
      { method: "POST", body: JSON.stringify({ val_percentage: valPercentage }) },
    ),

  cancelSession: () =>
    request<{ ok: boolean }>("/dpo/session/cancel", { method: "POST" }),

  imageUrl: (path: string) => `/api/dpo/session/image?path=${encodeURIComponent(path)}`,
};
