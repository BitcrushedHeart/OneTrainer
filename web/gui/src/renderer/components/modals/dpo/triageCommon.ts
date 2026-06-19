import { API_BASE } from "@/api/request";

export type Verdict = "good" | "bad" | "skip";

export const GOOD_COLOR = "#22c55e";
export const BAD_COLOR = "#ef4444";
export const SKIP_COLOR = "#9ca3af";

export function basename(p: string): string {
  return p.split(/[/\\]/).pop() ?? p;
}

export function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

export function verdictColor(v: Verdict | undefined): string {
  if (v === "good") return GOOD_COLOR;
  if (v === "bad") return BAD_COLOR;
  if (v === "skip") return SKIP_COLOR;
  return "var(--color-border-subtle)";
}

export function verdictLabel(v: Verdict): string {
  if (v === "good") return "Good";
  if (v === "bad") return "Bad";
  return "Skip";
}
