import { AlertCircle, Check, X } from "lucide-react";

import { type Character } from "@/api/characterDpoApi";

export const SECTION = "rounded border border-[var(--color-border-subtle)] p-3 flex flex-col gap-2";
export const SECTION_TITLE = "text-sm font-semibold text-[var(--color-on-surface)]";
export const HINT = "text-xs text-[var(--color-on-surface-secondary)]";

export type ChipTone = "neutral" | "ok" | "warn" | "bad" | "info";

const CHIP_TONE: Record<ChipTone, string> = {
  neutral: "text-[var(--color-on-surface-secondary)]",
  ok: "text-[var(--color-success-500)]",
  warn: "text-[var(--color-warning-500,#eab308)]",
  bad: "text-[var(--color-error-500)]",
  info: "text-[var(--color-cobalt-600)]",
};

export function Chip({
  tone = "neutral",
  title,
  children,
}: {
  tone?: ChipTone;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold tabular-nums whitespace-nowrap bg-[var(--color-surface-container)] ${CHIP_TONE[tone]}`}
    >
      {children}
    </span>
  );
}

export function Notice({ kind, text, onClose }: { kind: "error" | "info"; text: string; onClose: () => void }) {
  const color = kind === "error" ? "text-[var(--color-error-500)]" : "text-[var(--color-on-surface)]";
  return (
    <div className={`flex items-start gap-2 text-sm ${color} bg-[var(--color-surface-container)] rounded px-3 py-2`}>
      {kind === "error" ? (
        <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
      ) : (
        <Check className="w-4 h-4 mt-0.5 shrink-0" />
      )}
      <span className="flex-1 whitespace-pre-wrap break-words">{text}</span>
      <button type="button" onClick={onClose} className="shrink-0 opacity-70 hover:opacity-100" aria-label="Dismiss">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

/** A determinate (or, with total 0, indeterminate) progress bar in the modal's own style. */
export function Bar({ done, total }: { done: number; total: number }) {
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  return (
    <div className="h-2 rounded bg-[var(--color-border-subtle)] overflow-hidden">
      <div
        className={`h-full bg-[var(--color-cobalt-600)] transition-all ${total === 0 ? "skeleton" : ""}`}
        style={{ width: total > 0 ? `${pct}%` : "100%" }}
      />
    </div>
  );
}

export function joinPath(base: string, ...parts: string[]): string {
  const sep = base.includes("\\") && !base.includes("/") ? "\\" : "/";
  let out = base.replace(/[\\/]+$/, "");
  for (const p of parts) out += sep + p;
  return out;
}

export function loraStem(c: Character): string {
  if (c.lora.path) {
    const base = c.lora.path.split(/[\\/]/).pop() ?? "";
    return base.replace(/\.[^.]+$/, "");
  }
  return c.lora.swarm_name ?? "";
}

export function splitLabel(s: Character["offpolicy"]): string {
  return `${s.train}/${s.val}${s.unsplit > 0 ? ` (+${s.unsplit})` : ""}`;
}

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function baseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

export function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** The Off-policy DPO Builder's persisted knobs, reused for the embedded picker. */
export interface OffPolicyDefaults {
  tolerance: number;
  baseResolution: number;
  upscalePct: number;
  order: "random" | "path";
}

const OFFPOLICY_KEY = "ot.offpolicyDpo.folders";

export function loadOffPolicyDefaults(): OffPolicyDefaults {
  const fallback: OffPolicyDefaults = { tolerance: 0.15, baseResolution: 1024, upscalePct: 10, order: "random" };
  try {
    const raw = window.localStorage.getItem(OFFPOLICY_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<OffPolicyDefaults>;
    return {
      tolerance: Number(parsed.tolerance) || fallback.tolerance,
      baseResolution: Number(parsed.baseResolution) || fallback.baseResolution,
      upscalePct: Number.isFinite(Number(parsed.upscalePct)) ? Number(parsed.upscalePct) : fallback.upscalePct,
      order: parsed.order === "path" ? "path" : "random",
    };
  } catch {
    return fallback;
  }
}
