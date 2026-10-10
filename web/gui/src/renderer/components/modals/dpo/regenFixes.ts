import type { RegenFix, RegenItemStatus, RegenSegmentParams } from "@/api/dpoApi";

/**
 * Fix & Regenerate: the dialog's rows, its persisted settings and the
 * bookkeeping the session hook needs. Pure so the pieces that decide what
 * gets sent to SwarmUI are testable without the dialog.
 */

/** One fix row in the dialog: a preset or a user-added target. */
export interface FixRow {
  id: string;
  label: string;
  /** CLIPSeg text, `a|b` for one merged mask, or a `yolo-<model>` name. */
  target: string;
  enabled: boolean;
  /** SwarmUI inpaint strength for the region (its own default is 0.6). */
  creativity: number;
  /** Mask cutoff (SwarmUI default 0.5, 0.25 for yolo targets). */
  threshold: number;
  /** Text for the refined region instead of the global prompt. Changes the image's group on a rescan. */
  subPrompt: string;
  custom: boolean;
}

export const DEFAULT_CREATIVITY = 0.95;
export const DEFAULT_THRESHOLD = 0.3;

function preset(id: string, label: string, target: string): FixRow {
  return {
    id,
    label,
    target,
    enabled: false,
    creativity: DEFAULT_CREATIVITY,
    threshold: DEFAULT_THRESHOLD,
    subPrompt: "",
    custom: false,
  };
}

/** The built-in fixes, in dialog order. Targets are editable. */
export const PRESET_ROWS: FixRow[] = [
  preset("faces", "Fix Faces", "face"),
  preset("background_people", "Fix Background People", "person in the background"),
  preset("text", "Fix Text", "text"),
  preset("hands_feet", "Fix Hands/Feet", "hands|feet"),
  preset("limbs", "Fix Limbs/Body", "arms|legs|torso"),
];

export function customRow(): FixRow {
  return {
    id: `custom-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    label: "Fix …",
    target: "",
    enabled: true,
    creativity: DEFAULT_CREATIVITY,
    threshold: DEFAULT_THRESHOLD,
    subPrompt: "",
    custom: true,
  };
}

function clamp01(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(0, Math.min(1, value));
}

/** The enabled rows with a target, as the request wants them. */
export function rowsToFixes(rows: FixRow[]): RegenFix[] {
  return rows
    .filter((row) => row.enabled && row.target.trim().length > 0)
    .map((row) => ({
      target: row.target.trim(),
      creativity: clamp01(row.creativity, DEFAULT_CREATIVITY),
      threshold: clamp01(row.threshold, DEFAULT_THRESHOLD),
      sub_prompt: row.subPrompt.trim(),
    }));
}

/** Non-blank segment settings; the server coerces and clamps. */
export function segmentParamsFromForm(form: Record<string, string | boolean>): RegenSegmentParams {
  const out: RegenSegmentParams = {};
  for (const [key, value] of Object.entries(form)) {
    if (value === "" || value === undefined || value === null) continue;
    if (typeof value === "boolean") {
      if (value) out[key] = true;
    } else {
      out[key] = value;
    }
  }
  return out;
}

/** Append without duplicates, keeping the existing order. */
export function mergeAdded(existing: string[], added: string[]): string[] {
  if (added.length === 0) return existing;
  const seen = new Set(existing);
  const out = [...existing];
  for (const path of added) {
    if (!seen.has(path)) {
      seen.add(path);
      out.push(path);
    }
  }
  return out;
}

/** Restore preset rows that a stored settings blob is missing (new presets). */
export function reconcileRows(stored: FixRow[] | undefined): FixRow[] {
  if (!stored || stored.length === 0) return PRESET_ROWS.map((row) => ({ ...row }));
  const byId = new Map(stored.map((row) => [row.id, row]));
  const rows: FixRow[] = PRESET_ROWS.map((row) => {
    const saved = byId.get(row.id);
    return saved ? { ...row, ...saved, custom: false, label: row.label } : { ...row };
  });
  for (const row of stored) {
    if (row.custom) rows.push({ ...row });
  }
  return rows;
}

export interface RegenSettings {
  baseUrl: string;
  token: string;
  generated: string;
  rows: FixRow[];
  segment: Record<string, string | boolean>;
  modelOverride: string;
  concurrency: number;
}

export const REGEN_STORAGE_KEY = "ot.dpoRegen";
const SWARM_KEY = "ot.swarmGenerate";
const OFFPOLICY_KEY = "ot.offpolicyDpo.folders";

export function defaultRegenSettings(): RegenSettings {
  return {
    baseUrl: "http://localhost:7801",
    token: "",
    generated: "",
    rows: PRESET_ROWS.map((row) => ({ ...row })),
    segment: {},
    modelOverride: "",
    concurrency: 2,
  };
}

export function loadRegenSettings(): RegenSettings {
  const settings = defaultRegenSettings();
  try {
    const shared = window.localStorage.getItem(OFFPOLICY_KEY);
    if (shared) {
      const folders = JSON.parse(shared) as { generated?: string };
      if (folders.generated) settings.generated = folders.generated;
    }
    const swarm = window.localStorage.getItem(SWARM_KEY);
    if (swarm) {
      const s = JSON.parse(swarm) as { baseUrl?: string; token?: string; generated?: string };
      if (s.baseUrl) settings.baseUrl = s.baseUrl;
      if (s.token) settings.token = s.token;
      if (s.generated) settings.generated = s.generated;
    }
    const own = window.localStorage.getItem(REGEN_STORAGE_KEY);
    if (own) {
      const parsed = JSON.parse(own) as Partial<RegenSettings>;
      Object.assign(settings, parsed);
      settings.rows = reconcileRows(parsed.rows);
      settings.segment = parsed.segment ?? {};
    }
  } catch {
    /* storage unavailable — defaults are fine */
  }
  return settings;
}

export function saveRegenSettings(settings: RegenSettings): void {
  try {
    window.localStorage.setItem(REGEN_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* storage unavailable */
  }
}

/** What the session hook sends when the dialog's Generate is pressed. */
export interface RegenStartRequest {
  paths: string[];
  fixes: RegenFix[];
  segmentParams: RegenSegmentParams;
  modelOverride: string;
  baseUrl: string;
  token: string;
  generatedFolder: string;
  concurrency: number;
}

/** The run as the grid paints it: per-source status, what was added. */
export interface RegenUiState {
  running: boolean;
  phase: string;
  planned: number;
  done: number;
  failed: number;
  items: Record<string, RegenItemStatus>;
  added: string[];
  dropped: string[];
  error: string | null;
  warning: string | null;
  groupIndex: number;
}

export const EMPTY_REGEN: RegenUiState = {
  running: false,
  phase: "idle",
  planned: 0,
  done: 0,
  failed: 0,
  items: {},
  added: [],
  dropped: [],
  error: null,
  warning: null,
  groupIndex: 0,
};
