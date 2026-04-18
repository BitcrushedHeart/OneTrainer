import { SkipForward, X } from "lucide-react";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { API_BASE } from "@/api/request";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "./PreviewOverlay";
import { PromptExpander } from "./PromptExpander";
import type { SelectionStepProps } from "./types";

function basename(path: string | null): string {
  if (!path) return "";
  return path.split(/[/\\]/).pop() ?? path;
}

function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

export function SelectionStep({
  group,
  onPick,
  onAcceptPair,
  onDismissPair,
  onSkipGroup,
  onCancel,
  phase,
  bestImage,
  remainingImages,
  pairsDone,
  showAcceptDialog,
  pendingBest,
  pendingWorst,
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

  const hasEnoughImages = remainingImages.length >= 2;

  return (
    <div className="flex flex-col gap-3">
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

      {/* Phase banner */}
      <div
        className="w-full text-center font-bold text-sm py-2 px-3 rounded"
        style={{ color: banner.color, background: banner.bg }}
      >
        {banner.text}
      </div>

      {/* Thumbnail grid */}
      {hasEnoughImages ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 max-h-[500px] overflow-y-auto">
          {remainingImages.map((imgPath) => {
            const isBest = imgPath === bestImage;
            return (
              <button
                key={imgPath}
                type="button"
                onClick={() => handleThumbClick(imgPath)}
                onContextMenu={(e) => handleThumbContextMenu(e, imgPath)}
                className="relative rounded border-2 overflow-hidden transition-all hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-[var(--color-cobalt-600)]"
                style={{
                  borderColor: isBest ? "#22c55e" : "var(--color-border-subtle)",
                }}
              >
                <img
                  src={imageUrl(imgPath)}
                  alt={basename(imgPath)}
                  loading="lazy"
                  draggable={false}
                  className="object-cover h-40 w-full"
                />
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
              className="bg-[var(--color-surface-container)] border border-[var(--color-border-subtle)] rounded-lg p-6 max-w-2xl w-full shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-bold text-[var(--color-on-surface)] mb-4 text-center">Accept Pair</h3>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="flex flex-col items-center gap-2">
                  <div className="rounded border-2 overflow-hidden w-full" style={{ borderColor: "#22c55e" }}>
                    {pendingBest && (
                      <img
                        src={imageUrl(pendingBest)}
                        alt="Chosen"
                        className="object-cover w-full h-48"
                        draggable={false}
                      />
                    )}
                  </div>
                  <div className="text-xs font-semibold" style={{ color: "#22c55e" }}>
                    Chosen
                  </div>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <div className="rounded border-2 overflow-hidden w-full" style={{ borderColor: "#ef4444" }}>
                    {pendingWorst && (
                      <img
                        src={imageUrl(pendingWorst)}
                        alt="Rejected"
                        className="object-cover w-full h-48"
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
                Confirm = accept and move to next group · Pick More = accept and keep scoring this
                group · Cancel = discard this pair (nothing written)
              </div>
              <div className="flex items-center justify-center gap-2 flex-wrap">
                <Button variant="primary" size="sm" onClick={() => onAcceptPair(false)}>
                  Confirm
                </Button>
                <Button variant="secondary" size="sm" onClick={() => onAcceptPair(true)}>
                  Pick More
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
