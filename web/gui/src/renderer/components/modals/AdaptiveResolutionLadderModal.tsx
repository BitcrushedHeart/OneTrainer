import { Check, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/shared";
import { useBoundField } from "@/hooks/fieldBinding";

import { ModalBase } from "./ModalBase";

const DEFAULT_LADDER = "1024,768,512,256,128";

const PRESETS = [
  { label: "Standard", value: DEFAULT_LADDER },
  { label: "Power of two", value: "1024,512,256,128" },
  { label: "Fine steps", value: "1024,896,768,640,512,384,256,128" },
];

function parseLadder(value: string): { values: number[]; error?: string } {
  const parts = value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) return { values: [], error: "Add at least one snap resolution." };
  if (parts.some((part) => !/^\d+$/.test(part))) {
    return { values: [], error: "Use positive whole numbers separated by commas." };
  }
  const values = [...new Set(parts.map(Number))];
  if (values.some((item) => item <= 0)) {
    return { values: [], error: "Every snap resolution must be greater than zero." };
  }
  return { values: values.sort((a, b) => b - a) };
}

function parseMaximum(spec: string | undefined): number | null {
  const value = String(spec ?? "").trim();
  if (!value || (value.toLowerCase().includes("x") && !value.includes(","))) return null;
  const candidates = value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((item) => Number.isInteger(item) && item > 0);
  return candidates.length > 0 ? Math.max(...candidates) : null;
}

export interface AdaptiveResolutionLadderModalProps {
  open: boolean;
  onClose: () => void;
}

export function AdaptiveResolutionLadderModal({ open, onClose }: AdaptiveResolutionLadderModalProps) {
  const [storedLadder, setStoredLadder] = useBoundField<string>("adaptive_resolution_ladder");
  const [resolution] = useBoundField<string>("resolution");
  const [draft, setDraft] = useState(DEFAULT_LADDER);

  useEffect(() => {
    if (open) setDraft(storedLadder || DEFAULT_LADDER);
  }, [open, storedLadder]);

  const parsed = useMemo(() => parseLadder(draft), [draft]);
  const maximum = useMemo(() => parseMaximum(resolution), [resolution]);
  const activeTiers = useMemo(() => {
    if (parsed.error || maximum === null) return [];
    return [...new Set([maximum, ...parsed.values.filter((tier) => tier <= maximum)])].sort((a, b) => b - a);
  }, [maximum, parsed]);

  const save = () => {
    if (parsed.error) return;
    setStoredLadder(parsed.values.join(","));
    onClose();
  };

  return (
    <ModalBase open={open} onClose={onClose} title="Adaptive Resolution Ladder" size="md">
      <div className="flex flex-col gap-5">
        <div className="relative overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] p-4">
          <div aria-hidden="true" className="absolute inset-y-0 left-0 w-1 bg-[var(--color-cobalt-600)]" />
          <p className="text-sm font-semibold text-[var(--color-on-surface)]">Highest fit, never above the ceiling</p>
          <p className="mt-1 text-xs leading-5 text-[var(--color-on-surface-secondary)]">
            Your training resolution is always included as the maximum. Tiers above it are ignored; the smallest active
            tier becomes the intentional fallback for very small images.
          </p>
        </div>

        <div>
          <label
            htmlFor="adaptive-resolution-ladder"
            className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-[var(--color-on-surface-secondary)]"
          >
            Snap resolutions
          </label>
          <input
            id="adaptive-resolution-ladder"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            aria-invalid={!!parsed.error}
            placeholder={DEFAULT_LADDER}
            className="w-full rounded-[var(--radius-sm)] border border-[var(--color-border-subtle)] bg-[var(--color-surface)] px-3 py-2 font-mono text-sm text-[var(--color-on-surface)] outline-none transition focus:border-[var(--color-cobalt-600)] focus:ring-2 focus:ring-[color-mix(in_srgb,var(--color-cobalt-600)_25%,transparent)]"
          />
          {parsed.error ? (
            <p role="alert" className="mt-2 text-xs text-[var(--color-error-500)]">
              {parsed.error}
            </p>
          ) : (
            <p className="mt-2 text-xs text-[var(--color-on-surface-secondary)]">
              Comma-separated scalar resolutions. Duplicates are removed; model bucket quantization may adjust
              non-aligned values.
            </p>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => setDraft(preset.value)}
              className="rounded-full border border-[var(--color-border-subtle)] bg-transparent px-3 py-1.5 text-xs font-medium text-[var(--color-on-surface-secondary)] transition hover:border-[var(--color-cobalt-600)] hover:text-[var(--color-on-surface)]"
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div className="rounded-[var(--radius-md)] border border-[var(--color-border-subtle)] p-4">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--color-on-surface-secondary)]">
            Active ladder preview
          </p>
          {maximum === null ? (
            <p className="text-sm text-[var(--color-on-surface-secondary)]">
              Fixed WxH resolutions bypass adaptive snapping.
            </p>
          ) : parsed.error ? (
            <p className="text-sm text-[var(--color-on-surface-secondary)]">Correct the ladder to preview it.</p>
          ) : (
            <div className="flex flex-wrap items-center gap-2" aria-label="Active snap ladder">
              {activeTiers.map((tier, index) => (
                <div key={tier} className="contents">
                  {index > 0 && <span className="text-[var(--color-on-surface-secondary)]">→</span>}
                  <span
                    className={`inline-flex min-w-14 items-center justify-center rounded-full px-3 py-1.5 font-mono text-sm font-semibold ${
                      index === 0
                        ? "bg-[var(--color-cobalt-600)] text-white"
                        : "bg-[var(--color-surface-raised)] text-[var(--color-on-surface)]"
                    }`}
                  >
                    {tier}
                  </span>
                </div>
              ))}
            </div>
          )}
          {activeTiers.length > 0 && (
            <p className="mt-3 flex items-center gap-1.5 text-xs text-[var(--color-on-surface-secondary)]">
              <Check className="h-3.5 w-3.5 text-[var(--color-cobalt-600)]" />
              Fallback tier: {activeTiers.at(-1)} px
            </p>
          )}
        </div>

        <div className="flex justify-between border-t border-[var(--color-border-subtle)] pt-4">
          <Button variant="ghost" size="sm" onClick={() => setDraft(DEFAULT_LADDER)}>
            <RotateCcw className="h-4 w-4" /> Reset default
          </Button>
          <div className="flex gap-3">
            <Button variant="secondary" size="sm" onClick={onClose}>
              Cancel
            </Button>
            <Button size="sm" onClick={save} disabled={!!parsed.error} aria-label="Save Ladder">
              Save Ladder
            </Button>
          </div>
        </div>
      </div>
    </ModalBase>
  );
}
