import { request } from "./request";

export type TriageStatus =
  | "tagged"
  | "queued"
  | "generating"
  | "ready"
  | "fixed"
  | "kept"
  | "skipped"
  | "failed"
  | "deleted";

/** One fix on one image, as the backend's FixModel takes it. */
export interface TriageFixSpec {
  id: string;
  target: string;
  creativity?: number;
  threshold?: number;
  instruction: string;
  include_caption?: boolean;
}

/** A row of the session's category table (the number keys). */
export interface TriageCategory {
  id: string;
  key: string;
  label: string;
  target: string;
  instruction: string;
  include_caption: boolean;
  creativity: number;
  threshold: number;
  /** Detection ladder: when the YOLO target finds nothing, retry at low_threshold, then as this CLIPSeg word ("" = none). */
  fallback_target: string;
  low_threshold: number;
}

export interface TriageItem {
  stem: string;
  pair_id: number | null;
  chosen_path: string;
  caption: string;
  caption_source: "pair" | "manifest" | "unconditional";
  width: number | null;
  height: number | null;
  status: TriageStatus | null;
  detected: string | null;
  fixes: TriageFixSpec[];
  error: string | null;
  attempt: number;
  /** Repair round: 1 = first pass; higher after "fix more" on an accepted fix. */
  round: number;
  /** Epoch seconds of the current decision (fixed / kept / ...), null while undecided. */
  decided_at: number | null;
  /** The chosen image before any round repaired it; null when never fixed. */
  original_path: string | null;
  /** A fix pair was written (or updated) for this image in some round. */
  has_fix_pair: boolean;
  /** Second round: null = not taken up yet, "open" = taken up (tagged, queued or decided since), "skipped". */
  second_pass: "open" | "skipped" | null;
  /** What an image taken up by the second round was before (fixed / skipped / kept). */
  second_from: TriageStatus | null;
  /** Queued here but held back: the other session (by name) repairs it first. */
  waiting_on: string | null;
  /** Other sessions that have this image queued or in progress. */
  claimed_by: string[];
}

export interface TriageCounts {
  tagged: number;
  queued: number;
  generating: number;
  ready: number;
  fixed: number;
  kept: number;
  skipped: number;
  failed: number;
  deleted: number;
  undecided: number;
  total: number;
}

export interface TriageFixStatus {
  ok: boolean;
  session: string;
  name: string;
  /** Other running sessions on the same dataset. */
  peers: string[];
  /** Queued images held back while another session repairs them. */
  waiting: number;
  active: boolean;
  /** Sending held: confirms queue up but nothing new goes to SwarmUI until released. */
  held: boolean;
  phase: "idle" | "running" | "error";
  concept_dir: string | null;
  fix_pair_dir: string;
  shuffle_seed: number | null;
  options_per_image: number;
  has_detector: boolean;
  counts: TriageCounts;
  in_flight: number;
  can_undo: boolean;
  undo_label: string | null;
  error: string | null;
  warning: string | null;
}

export interface TriageCandidate {
  k: number;
  path: string | null;
  seed: number | null;
  mode: "segment" | "edit" | "mask" | null;
  no_change: boolean;
  diff: number | null;
  /** Which rung of the detection ladder produced it: 0 = as configured, 1 = low threshold, 2 = CLIPSeg word. */
  rung?: 0 | 1 | 2;
  fallback?: "" | "yolo low" | "clipseg";
  /** Fraction (0..1) of pixels that moved. */
  changed?: number | null;
  error: string | null;
}

export interface ReadyItem {
  stem: string;
  pair_id: number | null;
  caption: string;
  original_path: string;
  width: number | null;
  height: number | null;
  fixes: TriageFixSpec[];
  attempt: number;
  seed: number | null;
  candidates: TriageCandidate[];
  all_no_change: boolean;
  ready_at: number | null;
  /** The painted Quick Mask, when the item was queued with one. */
  mask: string | null;
}

export interface InspectResult {
  ok: boolean;
  error?: string;
  total?: number;
  has_ledger?: boolean;
  counts?: TriageCounts;
  shuffle_seed?: number | null;
  settings?: Record<string, unknown>;
}

export interface TriageStartRequest {
  concept_dir: string;
  base_url: string;
  token: string;
  model: string;
  steps: number;
  cfgscale: number;
  sampler: string;
  scheduler: string;
  sigmashift: number | null;
  /** SwarmUI "Color Correction Behavior": None / Uniform / Linear / Linear2. */
  color_correction: string;
  /** SwarmUI "Crop Reference Strength" (local patch, Flux.2 / Klein), 0–1. */
  crop_reference: number;
  /** LoRAs / adapters on top of the fixer model, named as SwarmUI lists them. */
  loras: Array<{ name: string; weight: number }>;
  options_per_image: number;
  concurrency: number;
  segment_params: Record<string, string | number | boolean>;
  mask_params: Record<string, number>;
  categories: TriageCategory[];
  prompt_template: string;
  fix_pair_dir: string;
  shuffle_seed: number | null;
  detector: string;
  /** Start with sending held. */
  hold: boolean;
  /** The session's tab name, used in the other sessions' messages. */
  name: string;
}

/** A session the backend knows (running or paused). */
export interface TriageSessionInfo {
  id: string;
  name: string;
  active: boolean;
  held: boolean;
  concept_dir: string | null;
  model: string;
  counts: TriageCounts | null;
}

/** The main session keeps the original ledger folder; the others are named. */
export const MAIN_TRIAGE_SESSION = "main";

interface Ack {
  ok: boolean;
  error?: string;
}

/** Another session's fixes cover the same area: nothing was queued. */
interface Conflicts {
  conflicts?: Record<string, string>;
  overlap?: boolean;
}

/** The API of one Triage & Fix session. */
export function triageFixApi(session: string) {
  const q = (path: string) => `${path}${path.includes("?") ? "&" : "?"}session=${encodeURIComponent(session)}`;
  return {
    concepts: () => request<{ ok: boolean; concepts: string[]; warning?: string }>(q("/dpo/triage-fix/concepts")),

    inspect: (conceptDir: string) =>
      request<InspectResult>(q(`/dpo/triage-fix/inspect?concept_dir=${encodeURIComponent(conceptDir)}`)),

    start: (body: TriageStartRequest) =>
      request<{
        ok: boolean;
        total?: number;
        undecided?: number;
        ready?: number;
        resumed?: boolean;
        shuffle_seed?: number;
        held?: boolean;
        error?: string;
      }>(q("/dpo/triage-fix/start"), { method: "POST", body: JSON.stringify(body) }),

    status: () => request<TriageFixStatus>(q("/dpo/triage-fix/status")),

    // Hold: nothing new goes to SwarmUI (images it already has finish); release sends the queue.
    hold: (held: boolean) =>
      request<Ack & { held?: boolean; withdrawn?: number }>(q("/dpo/triage-fix/hold"), {
        method: "POST",
        body: JSON.stringify({ held }),
      }),

    items: () =>
      request<{ ok: boolean; items?: TriageItem[]; categories?: TriageCategory[]; error?: string }>(
        q("/dpo/triage-fix/items"),
      ),

    confirm: (stem: string, fixes: TriageFixSpec[], secondRound = false) =>
      request<Ack & Conflicts & { status?: TriageStatus }>(q("/dpo/triage-fix/confirm"), {
        method: "POST",
        body: JSON.stringify({ stem, fixes, second_round: secondRound }),
      }),

    // Quick Mask: a painted region plus an instruction, sent straight off.
    confirmMask: (
      stem: string,
      mask: string,
      instruction: string,
      creativity?: number,
      secondRound = false,
      colorCorrection?: string,
      cropReference?: number,
    ) =>
      request<Ack & Conflicts & { status?: TriageStatus }>(q("/dpo/triage-fix/confirm-mask"), {
        method: "POST",
        body: JSON.stringify({
          stem,
          mask,
          instruction,
          creativity: creativity ?? null,
          second_round: secondRound,
          color_correction: colorCorrection ?? null,
          crop_reference: cropReference ?? null,
        }),
      }),

    confirmAll: (mode: "default" | "detected", fixes: TriageFixSpec[] = []) =>
      request<Ack & Conflicts & { queued?: number }>(q("/dpo/triage-fix/confirm-all"), {
        method: "POST",
        body: JSON.stringify({ mode, fixes }),
      }),

    skip: (stem: string) => request<Ack>(q("/dpo/triage-fix/skip"), { method: "POST", body: JSON.stringify({ stem }) }),

    // Triage grid view: the same, for a selection. Decided images are refused, not taken up.
    confirmManyFirst: (stems: string[], fixes: TriageFixSpec[]) =>
      request<Ack & Conflicts & { queued?: string[]; refused?: string[] }>(q("/dpo/triage-fix/confirm-many"), {
        method: "POST",
        body: JSON.stringify({ stems, fixes }),
      }),
    skipMany: (stems: string[]) =>
      request<Ack & { skipped?: string[]; refused?: string[] }>(q("/dpo/triage-fix/skip-many"), {
        method: "POST",
        body: JSON.stringify({ stems }),
      }),

    undoTriage: (stem: string) =>
      request<Ack & { status?: TriageStatus | null }>(q("/dpo/triage-fix/undo-triage"), {
        method: "POST",
        body: JSON.stringify({ stem }),
      }),

    nextReady: () =>
      request<Ack & { item?: ReadyItem; waiting?: boolean; held?: boolean; done?: boolean }>(
        q("/dpo/triage-fix/next-ready"),
        {
          method: "POST",
        },
      ),

    pick: (stem: string, k: number) =>
      request<Ack & { backup?: string; fix_pair_id?: number | null }>(q("/dpo/triage-fix/pick"), {
        method: "POST",
        body: JSON.stringify({ stem, k }),
      }),

    keep: (stem: string) => request<Ack>(q("/dpo/triage-fix/keep"), { method: "POST", body: JSON.stringify({ stem }) }),

    // A re-roll is a new seed only; `lower` also turns the creativity down a notch.
    requeue: (stem: string, fixes?: TriageFixSpec[], lower = false) =>
      request<Ack>(q("/dpo/triage-fix/requeue"), {
        method: "POST",
        body: JSON.stringify({ stem, fixes: fixes ?? null, lower }),
      }),

    // Send a ready / failed / kept image back to triage with its fixes intact.
    retriage: (stem: string) =>
      request<Ack & { status?: TriageStatus; round?: number }>(q("/dpo/triage-fix/retriage"), {
        method: "POST",
        body: JSON.stringify({ stem }),
      }),

    // Remove the pair from the dataset (files move to .triage_fix/deleted; undoable).
    // Recovery: write the fix pairs of accepted repairs made without a fix-pair folder.
    addFixPairs: (stems: string[]) =>
      request<Ack & { added?: string[]; skipped?: string[]; errors?: string[] }>(q("/dpo/triage-fix/add-fix-pairs"), {
        method: "POST",
        body: JSON.stringify({ stems }),
      }),
    // Second round: fixed, skipped and kept images taken up again.
    confirmMany: (stems: string[], fixes: TriageFixSpec[]) =>
      request<Ack & Conflicts & { queued?: string[]; refused?: string[] }>(q("/dpo/triage-fix/second/confirm-many"), {
        method: "POST",
        body: JSON.stringify({ stems, fixes }),
      }),
    skipSecond: (stems: string[]) =>
      request<Ack & { skipped?: string[]; refused?: string[] }>(q("/dpo/triage-fix/second/skip"), {
        method: "POST",
        body: JSON.stringify({ stems }),
      }),
    clearSecond: () => request<Ack & { cleared?: number }>(q("/dpo/triage-fix/second/clear"), { method: "POST" }),
    undoSecond: (stem: string) =>
      request<Ack & { status?: TriageStatus | null }>(q("/dpo/triage-fix/second/undo"), {
        method: "POST",
        body: JSON.stringify({ stem }),
      }),
    deletePair: (stem: string) =>
      request<Ack & { folder?: string }>(q("/dpo/triage-fix/delete-pair"), {
        method: "POST",
        body: JSON.stringify({ stem }),
      }),

    undoDecision: () =>
      request<Ack & { undone?: "fixed" | "kept" | "deleted"; stem?: string }>(q("/dpo/triage-fix/undo-decision"), {
        method: "POST",
      }),

    cancel: () => request<Ack>(q("/dpo/triage-fix/cancel"), { method: "POST" }),
  };
}

export type TriageFixApi = ReturnType<typeof triageFixApi>;

/** The main session. */
export const dpoTriageFixApi = triageFixApi(MAIN_TRIAGE_SESSION);

export const triageSessionsApi = {
  list: () => request<{ ok: boolean; sessions: TriageSessionInfo[] }>("/dpo/triage-fix/sessions"),
};
