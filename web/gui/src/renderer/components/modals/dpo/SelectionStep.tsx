import { Ban, Cpu, Loader2, SkipForward, Sparkles, Trash2, X, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { API_BASE } from "@/api/request";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "./PreviewOverlay";
import { PromptExpander } from "./PromptExpander";
import type { RealismDevice, SelectionStepProps } from "./types";

// The scoring control cycles rather than opening a menu: three positions is
// short enough to click through, and it is the same setting the setup screen
// exposes as a dropdown before the session starts.
const SCORING_CYCLE: Record<RealismDevice, RealismDevice> = { auto: "cpu", cpu: "off", off: "auto" };

const SCORING_LABEL: Record<RealismDevice, string> = { auto: "GPU", cpu: "CPU", off: "Scoring off" };

const SCORING_TITLE: Record<RealismDevice, string> = {
  auto: "Scoring on the GPU when one is available. Click to run on the CPU instead, leaving VRAM to training.",
  cpu: "Scoring on the CPU — slower, but it leaves the GPU free. Click to turn scoring off entirely.",
  off: "Scoring is off: no model, no sort, no badges. Click to score on the GPU again.",
};

function basename(path: string | null): string {
  if (!path) return "";
  return path.split(/[/\\]/).pop() ?? path;
}

function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

function parseAspectRatio(ar: string): { ratio: number; css: string } | null {
  const m = /^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/.exec(ar.trim());
  if (!m) return null;
  const w = parseFloat(m[1]);
  const h = parseFloat(m[2]);
  if (!w || !h) return null;
  return { ratio: w / h, css: `${m[1]} / ${m[2]}` };
}

function columnsForRatio(ratio: number | null, count: number): number {
  // Pick a grid column count that lets wide images breathe while keeping tall
  // images at a reasonable size. Never exceed the image count.
  let cols: number;
  if (ratio === null) cols = 4;
  else if (ratio >= 1.7) cols = 2;
  else if (ratio >= 1.15) cols = 3;
  else if (ratio >= 0.85) cols = 4;
  else cols = 5;
  return Math.max(1, Math.min(cols, count));
}

// Colour ramp for the realism badge: green = the model finds it convincing,
// red = it does not. Deliberately coarse — the number is a hint, the ordering
// is the real signal.
function realismColor(pAi: number): string {
  if (pAi < 0.5) return "#22c55e";
  if (pAi < 0.8) return "#eab308";
  if (pAi < 0.95) return "#f97316";
  return "#ef4444";
}

export function SelectionStep({
  group,
  onPick,
  onAcceptPair,
  onDismissPair,
  onSkipGroup,
  onDiscardGroup,
  onCancel,
  phase,
  bestImage,
  remainingImages,
  pairsDone,
  showAcceptDialog,
  pendingBest,
  pendingWorst,
  realismScores,
  realismError,
  backfill,
  onStartBackfill,
  onStopBackfill,
  // Optional on the props so tests and modes that do not wire scoring still
  // render; "auto" is what the server starts on.
  realismDevice = "auto",
  onSetRealismDevice,
}: SelectionStepProps) {
  const [previewPath, setPreviewPath] = useState<string | null>(null);

  // Close preview if the step changes such that the path is no longer in the list.
  useEffect(() => {
    if (previewPath && !remainingImages.includes(previewPath)) {
      setPreviewPath(null);
    }
  }, [previewPath, remainingImages]);

  // Esc inside the accept-pair dialog is Cancel — discards the pending pair
  // (no file written) and returns to the worst-pick UI. Matches Ctk's Cancel
  // branch of askyesnocancel. Capture-phase stopPropagation so ModalBase's
  // document-level Escape handler doesn't also close the whole modal.
  useEffect(() => {
    if (!showAcceptDialog) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      onDismissPair();
    };
    window.addEventListener("keydown", handleKey, true);
    return () => window.removeEventListener("keydown", handleKey, true);
  }, [showAcceptDialog, onDismissPair]);

  const handleThumbClick = (path: string) => {
    setPreviewPath(path);
  };

  const handleThumbContextMenu = (e: React.MouseEvent, path: string) => {
    e.preventDefault();
    void onPick(path);
  };

  const banner =
    phase === "best"
      ? {
          text: `Pair ${pairsDone + 1}/${group.pairs_target}. Click an image to preview, right-click to select as BEST`,
          color: "#22c55e",
          bg: "rgba(34,197,94,0.1)",
        }
      : {
          text: `Best: ${basename(bestImage)}. Click to preview, right-click to select WORST`,
          color: "#ef4444",
          bg: "rgba(239,68,68,0.1)",
        };

  // Phase "best": need ≥2 images to form a pair. Phase "worst": a best is
  // already chosen, so ≥1 remaining image is enough to complete the pair —
  // matches Ctk, which renders the grid minus `selected_best` regardless of
  // count. Using ≥2 here broke the 2-image-group case: after picking best,
  // exactly 1 image remains and the UI falsely reported "not enough images".
  const minRemaining = phase === "worst" ? 1 : 2;
  const hasEnoughImages = remainingImages.length >= minRemaining;

  // Most photorealistic first — the lowest P(AI) is the strongest DPO winner
  // candidate. Unscored images keep their source position relative to each
  // other (Array.sort is stable), so a partially-scored or unavailable model
  // degrades to the original order rather than to a scrambled one.
  const scored = realismScores ?? {};
  const displayImages =
    Object.keys(scored).length > 0
      ? [...remainingImages].sort((a, b) => {
          const sa = scored[a];
          const sb = scored[b];
          if (sa === undefined && sb === undefined) return 0;
          if (sa === undefined) return 1;
          if (sb === undefined) return -1;
          return sa - sb;
        })
      : remainingImages;
  const ar = parseAspectRatio(group.aspectratio);
  const arStyle: React.CSSProperties | undefined = ar ? { aspectRatio: ar.css } : undefined;
  const gridCols = columnsForRatio(ar?.ratio ?? null, displayImages.length);

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      {/* Header row */}
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs text-[var(--color-on-surface-secondary)] flex items-center gap-3 flex-wrap">
          <span>
            Group {group.group_index} / {group.total_groups}
          </span>
          <span className="text-[var(--color-border-subtle)]">·</span>
          <span>AR: {group.aspectratio}</span>
          <span className="text-[var(--color-border-subtle)]">·</span>
          <span>
            Pair {pairsDone + 1}/{group.pairs_target}
          </span>
          {realismDevice === "off" ? (
            <>
              <span className="text-[var(--color-border-subtle)]">·</span>
              <span>scoring off — source order</span>
            </>
          ) : realismError ? (
            <>
              <span className="text-[var(--color-border-subtle)]">·</span>
              <span title={realismError} style={{ color: "#f97316" }}>
                realism scoring unavailable
              </span>
            </>
          ) : Object.keys(scored).length > 0 ? (
            <>
              <span className="text-[var(--color-border-subtle)]">·</span>
              <span style={{ color: "#22c55e" }}>sorted most photorealistic first</span>
            </>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {/* Score the whole source tree once so no group ever waits on the
              model. Hidden when scoring is unavailable — there would be nothing
              to backfill. */}
          {/* Move scoring off the GPU when something else needs the VRAM. The
              server applies it from the next call on, so a backfill in flight
              switches at its next chunk instead of restarting. */}
          {/* Stays visible with scoring off — it is the way back on, and the
              "unavailable" case is the only one where there is nothing to pick
              between. */}
          {onSetRealismDevice && (!realismError || realismDevice === "off") ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onSetRealismDevice(SCORING_CYCLE[realismDevice])}
              title={SCORING_TITLE[realismDevice]}
              style={
                realismDevice === "cpu"
                  ? { color: "#60a5fa" }
                  : realismDevice === "off"
                    ? { color: "#f97316" }
                    : undefined
              }
            >
              {realismDevice === "cpu" ? (
                <Cpu className="w-4 h-4" />
              ) : realismDevice === "off" ? (
                <Ban className="w-4 h-4" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              {SCORING_LABEL[realismDevice]}
            </Button>
          ) : null}
          {onStartBackfill && !realismError && realismDevice !== "off" ? (
            backfill?.running ? (
              <Button variant="ghost" size="sm" onClick={onStopBackfill} title="Stop after the current chunk">
                <Loader2 className="w-4 h-4 animate-spin" />
                {backfill.total > 0 ? `Scoring ${backfill.done}/${backfill.total}` : "Scoring…"}
              </Button>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                onClick={onStartBackfill}
                title={
                  backfill?.error ??
                  "Score every image in the source folder once, so each group is instant from then on"
                }
                style={backfill?.error ? { color: "#f97316" } : undefined}
              >
                <Sparkles className="w-4 h-4" />
                {backfill?.error ? "Backfill failed" : "Backfill Scores"}
              </Button>
            )
          ) : null}
          <Button variant="ghost" size="sm" onClick={onSkipGroup} style={{ background: "#8B4513", color: "#fff" }}>
            <SkipForward className="w-4 h-4" />
            Skip Group
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onDiscardGroup}
            style={{ background: "#7f1d1d", color: "#fff" }}
            title="Move every image still in this group to a .discard subfolder so it never reappears (pairs already saved are kept)"
          >
            <Trash2 className="w-4 h-4" />
            Discard Group
          </Button>
          <Button variant="ghost" size="sm" onClick={onCancel}>
            <X className="w-4 h-4" />
            Cancel
          </Button>
        </div>
      </div>

      {/* Prompt expander */}
      <PromptExpander prompt={group.prompt} />

      {/* Phase banner */}
      <div
        className="w-full text-center font-bold text-sm py-2 px-3 rounded"
        style={{ color: banner.color, background: banner.bg }}
      >
        {banner.text}
      </div>

      {/* Thumbnail grid */}
      {hasEnoughImages ? (
        <div
          className="grid gap-2 flex-1 min-h-0 overflow-y-auto auto-rows-min content-start"
          style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}
        >
          {displayImages.map((imgPath) => {
            const isBest = imgPath === bestImage;
            return (
              <button
                key={imgPath}
                type="button"
                onClick={() => handleThumbClick(imgPath)}
                onContextMenu={(e) => handleThumbContextMenu(e, imgPath)}
                className="group relative rounded border-2 overflow-hidden transition-all hover:shadow-lg hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-[var(--color-cobalt-600)] bg-black/30"
                style={{
                  borderColor: isBest ? "#22c55e" : "var(--color-border-subtle)",
                  ...arStyle,
                }}
                onMouseEnter={(e) => {
                  if (!isBest) e.currentTarget.style.borderColor = "var(--color-cobalt-600)";
                }}
                onMouseLeave={(e) => {
                  if (!isBest) e.currentTarget.style.borderColor = "var(--color-border-subtle)";
                }}
              >
                <img
                  src={imageUrl(imgPath)}
                  alt={basename(imgPath)}
                  loading="lazy"
                  draggable={false}
                  className={`${arStyle ? "object-contain w-full h-full" : "object-contain w-full h-40"} transition-[filter] group-hover:brightness-110`}
                />
                {isBest && (
                  <div
                    className="absolute top-1 left-1 text-[10px] font-bold px-2 py-0.5 rounded shadow"
                    style={{ background: "#22c55e", color: "#0b2c13" }}
                  >
                    BEST
                  </div>
                )}
                {scored[imgPath] !== undefined && (
                  <div
                    className="absolute top-1 right-1 text-[10px] font-bold px-2 py-0.5 rounded shadow"
                    style={{ background: realismColor(scored[imgPath]), color: "#0b0b0b" }}
                    title={`Model thinks this is ${(scored[imgPath] * 100).toFixed(1)}% AI-generated`}
                  >
                    {(scored[imgPath] * 100).toFixed(0)}% AI
                  </div>
                )}
                <div
                  className="absolute bottom-0 left-0 right-0 text-[10px] text-white/90 bg-black/60 px-1 py-0.5 truncate text-left font-mono"
                  title={basename(imgPath)}
                >
                  {basename(imgPath)}
                </div>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="text-sm text-[var(--color-on-surface-secondary)] py-8 text-center">
          Not enough images remaining in group
        </div>
      )}

      {/* Preview overlay */}
      {previewPath && (
        <PreviewOverlay
          path={previewPath}
          caption={basename(previewPath)}
          onClose={() => setPreviewPath(null)}
          onPick={async () => {
            const toPick = previewPath;
            await onPick(toPick);
            setPreviewPath(null);
          }}
        />
      )}

      {/* 3-way Accept Pair dialog */}
      {showAcceptDialog &&
        createPortal(
          <div
            className="fixed inset-0 bg-black/70 z-[55] flex items-center justify-center p-6"
            onClick={(e) => {
              // Backdrop click does nothing — user must choose a button.
              e.stopPropagation();
            }}
          >
            <div
              className="bg-[var(--color-surface-container)] border border-[var(--color-border-subtle)] rounded-lg p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-bold text-[var(--color-on-surface)] mb-4 text-center">Accept Pair</h3>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="flex flex-col items-center gap-2">
                  <div
                    className="rounded border-2 overflow-hidden w-full bg-black/30"
                    style={{ borderColor: "#22c55e", ...arStyle }}
                  >
                    {pendingBest && (
                      <img
                        src={imageUrl(pendingBest)}
                        alt="Chosen"
                        className={arStyle ? "object-contain w-full h-full" : "object-contain w-full h-48"}
                        draggable={false}
                      />
                    )}
                  </div>
                  <div className="text-xs font-semibold" style={{ color: "#22c55e" }}>
                    Chosen
                  </div>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <div
                    className="rounded border-2 overflow-hidden w-full bg-black/30"
                    style={{ borderColor: "#ef4444", ...arStyle }}
                  >
                    {pendingWorst && (
                      <img
                        src={imageUrl(pendingWorst)}
                        alt="Rejected"
                        className={arStyle ? "object-contain w-full h-full" : "object-contain w-full h-48"}
                        draggable={false}
                      />
                    )}
                  </div>
                  <div className="text-xs font-semibold" style={{ color: "#ef4444" }}>
                    Rejected
                  </div>
                </div>
              </div>
              <div className="text-xs text-[var(--color-on-surface-secondary)] text-center mb-4 font-mono break-all">
                Best: {basename(pendingBest)} / Worst: {basename(pendingWorst)}
              </div>
              <div className="text-[11px] text-[var(--color-on-surface-secondary)] text-center mb-2">
                Confirm = accept and move to next group · Pick More = accept and keep scoring this group · Save &
                Discard Rest = accept, then retire the whole group from the source tree · Cancel = discard this pair
                (nothing written)
              </div>
              <div className="flex items-center justify-center gap-2 flex-wrap">
                <Button variant="primary" size="sm" onClick={() => onAcceptPair(false)}>
                  Confirm
                </Button>
                <Button variant="secondary" size="sm" onClick={() => onAcceptPair(true)}>
                  Pick More
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onAcceptPair(false, true)}
                  style={{ background: "#7f1d1d", color: "#fff" }}
                  title="Save this pair, then move the paired sources to .chosen/.rejected and every remaining image to .discard so this group won't reappear"
                >
                  <Trash2 className="w-4 h-4" />
                  Save &amp; Discard Rest
                </Button>
                <Button variant="ghost" size="sm" onClick={onDismissPair}>
                  Cancel
                </Button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
