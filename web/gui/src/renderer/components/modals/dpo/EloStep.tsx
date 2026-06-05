import { ArrowDown, ArrowLeft, ArrowRight, CheckCircle2, Loader2, SkipForward, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { API_BASE } from "@/api/request";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "./PreviewOverlay";
import { PromptExpander } from "./PromptExpander";
import type { EloStepProps } from "./types";

const MIN_COMPARISONS_TO_ACCEPT = 5;

function basename(p: string): string {
  return p.split(/[/\\]/).pop() ?? p;
}

function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

function formatRating(value: number | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return value.toFixed(0);
}

function aspectRatioStyle(ar: string): React.CSSProperties | undefined {
  const m = /^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/.exec(ar.trim());
  if (!m) return undefined;
  return { aspectRatio: `${m[1]} / ${m[2]}` };
}

export function EloStep({
  group,
  pair,
  ratings,
  done,
  suggested,
  finished,
  onVote,
  onAccept,
  onSkipGroup,
  onCancel,
  pairsDone,
}: EloStepProps) {
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [showFinishDialog, setShowFinishDialog] = useState(false);

  const acceptThreshold = Math.min(MIN_COMPARISONS_TO_ACCEPT, suggested);
  const canAccept = done >= acceptThreshold;
  const showAcceptPanel = finished || showFinishDialog;

  // Clear preview if the currently previewed path is no longer part of the pair
  useEffect(() => {
    if (!previewPath) return;
    if (!pair || !pair.includes(previewPath)) {
      setPreviewPath(null);
    }
  }, [previewPath, pair]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      // Comparison-mode shortcuts (only when no preview is open and not in finish/dialog state)
      if (!showAcceptPanel && !previewPath && pair) {
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          void onVote("a");
          return;
        }
        if (e.key === "ArrowRight") {
          e.preventDefault();
          void onVote("b");
          return;
        }
        if (e.key === "ArrowDown") {
          e.preventDefault();
          void onVote("tie");
          return;
        }
        if (e.key === "Enter" && canAccept) {
          e.preventDefault();
          setShowFinishDialog(true);
          return;
        }
      }

      // Accept-panel shortcuts (only when no preview open)
      if (showAcceptPanel && !previewPath) {
        if (e.key === "Enter") {
          e.preventDefault();
          void onAccept(true);
          return;
        }
        if (e.key === "n" || e.key === "N") {
          e.preventDefault();
          void onAccept(false);
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [showAcceptPanel, previewPath, pair, canAccept, onVote, onAccept]);

  // Sort ratings for accept-panel display
  const { bestPath, worstPath, bestRating, worstRating } = useMemo(() => {
    const entries = Object.entries(ratings);
    if (entries.length === 0) {
      return { bestPath: null, worstPath: null, bestRating: 0, worstRating: 0 };
    }
    const sorted = [...entries].sort((a, b) => b[1] - a[1]);
    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    return {
      bestPath: first[0],
      worstPath: last[0],
      bestRating: first[1],
      worstRating: last[1],
    };
  }, [ratings]);

  const handleImageClick = (path: string) => {
    setPreviewPath(path);
  };

  const handleImageContextMenu = (e: React.MouseEvent, idx: number) => {
    e.preventDefault();
    void onVote(idx === 0 ? "a" : "b");
  };

  const handlePreviewPick = async () => {
    if (!pair || !previewPath) {
      setPreviewPath(null);
      return;
    }
    const idx = pair.indexOf(previewPath);
    setPreviewPath(null);
    if (idx >= 0) {
      await onVote(idx === 0 ? "a" : "b");
    }
  };

  const acceptButtonTitle = canAccept
    ? undefined
    : `Do at least ${acceptThreshold} comparisons before accepting (currently ${done}).`;

  const arStyle = aspectRatioStyle(group.aspectratio);

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
          <span className="tabular-nums">
            ELO comparisons {done}/{suggested}
          </span>
          <span className="text-[var(--color-border-subtle)]">·</span>
          <span className="tabular-nums">
            Pair {pairsDone + 1}/{group.pairs_target}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={onSkipGroup} style={{ background: "#8B4513", color: "#fff" }}>
            <SkipForward className="w-4 h-4" />
            Skip Group
          </Button>
          <Button variant="ghost" size="sm" onClick={onCancel}>
            <X className="w-4 h-4" />
            Cancel
          </Button>
        </div>
      </div>

      {/* Prompt expander */}
      <PromptExpander prompt={group.prompt} />

      {/* Main area: comparison vs. accept panel */}
      {showAcceptPanel ? (
        <div className="flex flex-col gap-4 p-4 rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] flex-1 min-h-0">
          <h3 className="text-lg font-bold text-[var(--color-on-surface)] text-center">Accept Pair</h3>
          <div className="grid grid-cols-2 gap-4 flex-1 min-h-0">
            <div className="flex flex-col items-center gap-2 min-h-0">
              <div
                className="rounded border-2 overflow-hidden w-full bg-black/30 flex-1 min-h-0"
                style={{ borderColor: "var(--color-success-500)", ...arStyle }}
              >
                {bestPath && (
                  <img src={imageUrl(bestPath)} alt="Best" className="object-contain w-full h-full" draggable={false} />
                )}
              </div>
              <div
                className="text-xs font-semibold flex items-center gap-2"
                style={{ color: "var(--color-success-500)" }}
              >
                <span>Best</span>
                <span className="tabular-nums">ELO {formatRating(bestRating)}</span>
              </div>
              <div
                className="text-[10px] font-mono text-[var(--color-on-surface-secondary)] break-all text-center"
                title={bestPath ?? ""}
              >
                {bestPath ? basename(bestPath) : ""}
              </div>
            </div>
            <div className="flex flex-col items-center gap-2 min-h-0">
              <div
                className="rounded border-2 overflow-hidden w-full bg-black/30 flex-1 min-h-0"
                style={{ borderColor: "var(--color-error-500)", ...arStyle }}
              >
                {worstPath && (
                  <img
                    src={imageUrl(worstPath)}
                    alt="Worst"
                    className="object-contain w-full h-full"
                    draggable={false}
                  />
                )}
              </div>
              <div
                className="text-xs font-semibold flex items-center gap-2"
                style={{ color: "var(--color-error-500)" }}
              >
                <span>Worst</span>
                <span className="tabular-nums">ELO {formatRating(worstRating)}</span>
              </div>
              <div
                className="text-[10px] font-mono text-[var(--color-on-surface-secondary)] break-all text-center"
                title={worstPath ?? ""}
              >
                {worstPath ? basename(worstPath) : ""}
              </div>
            </div>
          </div>
          <div className="text-xs text-[var(--color-on-surface-secondary)] text-center">
            Best: {bestPath ? basename(bestPath) : "—"} (ELO {formatRating(bestRating)}). Worst:{" "}
            {worstPath ? basename(worstPath) : "—"} (ELO {formatRating(worstRating)}). Pick how to proceed.
          </div>
          <div className="flex items-center justify-center gap-2 flex-wrap">
            <Button variant="primary" size="sm" onClick={() => void onAccept(true)}>
              Accept &amp; keep scoring
            </Button>
            <Button variant="primary" size="sm" onClick={() => void onAccept(false)}>
              Accept &amp; next group
            </Button>
            {showFinishDialog && !finished && (
              <Button variant="ghost" size="sm" onClick={() => setShowFinishDialog(false)}>
                Back to comparisons
              </Button>
            )}
          </div>
        </div>
      ) : pair ? (
        <>
          {/* Image comparison grid */}
          <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
            {pair.map((path, idx) => (
              <button
                key={path}
                type="button"
                onClick={() => handleImageClick(path)}
                onContextMenu={(e) => handleImageContextMenu(e, idx)}
                className="relative rounded border-2 overflow-hidden transition-all hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-[var(--color-cobalt-600)] min-h-0"
                style={{ borderColor: "var(--color-border-subtle)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--color-cobalt-600)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--color-border-subtle)";
                }}
              >
                <img
                  src={imageUrl(path)}
                  alt={basename(path)}
                  draggable={false}
                  className="w-full h-full object-contain bg-black/30"
                />
                <div
                  className="absolute bottom-0 left-0 right-0 flex items-center justify-between gap-2 text-[10px] text-white/90 bg-black/60 px-2 py-1 font-mono"
                  title={basename(path)}
                >
                  <span className="truncate">{basename(path)}</span>
                  <span className="tabular-nums shrink-0">ELO {formatRating(ratings[path])}</span>
                </div>
                <div
                  className="absolute top-2 left-2 text-[10px] font-bold px-2 py-0.5 rounded bg-black/60 text-white"
                  aria-hidden
                >
                  {idx === 0 ? "A" : "B"}
                </div>
              </button>
            ))}
          </div>

          {/* Vote buttons */}
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <Button variant="primary" size="md" onClick={() => void onVote("a")}>
              <ArrowLeft className="w-4 h-4" />A is Better (←)
            </Button>
            <Button variant="secondary" size="md" onClick={() => void onVote("tie")}>
              <ArrowDown className="w-4 h-4" />
              Tie / Skip (↓)
            </Button>
            <Button variant="primary" size="md" onClick={() => void onVote("b")}>
              B is Better (→)
              <ArrowRight className="w-4 h-4" />
            </Button>
          </div>

          {/* Accept controls row */}
          <div className="flex items-center justify-end">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowFinishDialog(true)}
              disabled={!canAccept}
              title={acceptButtonTitle}
            >
              <CheckCircle2 className="w-4 h-4" />
              Accept Pair
            </Button>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center py-16 text-[var(--color-on-surface-secondary)]">
          <Loader2 className="w-6 h-6 animate-spin" />
        </div>
      )}

      {/* Preview overlay */}
      {previewPath && (
        <PreviewOverlay
          path={previewPath}
          caption={basename(previewPath)}
          onClose={() => setPreviewPath(null)}
          onPick={handlePreviewPick}
        />
      )}
    </div>
  );
}
