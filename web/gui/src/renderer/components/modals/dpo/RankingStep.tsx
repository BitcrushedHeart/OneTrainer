import { Download, SkipForward, X } from "lucide-react";
import { useMemo, useState } from "react";

import { API_BASE } from "@/api/request";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "./PreviewOverlay";
import { PromptExpander } from "./PromptExpander";
import type { RankingStepProps } from "./types";

function basename(p: string): string {
  return p.split(/[/\\]/).pop() ?? p;
}

function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

export function RankingStep({
  group,
  order,
  scores,
  maxPairs,
  pairCount,
  onPairCountChange,
  onSwap,
  onExport,
  onSkipGroup,
  onCancel,
  pairsDone,
}: RankingStepProps) {
  const [swapIndex, setSwapIndex] = useState<number | null>(null);
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  const clampedPairs = Math.max(0, Math.min(pairCount, maxPairs));

  // First k ranks export as chosen (green), last k as rejected (red);
  // mirrors outermost_pairs on the server: rank 1 vs n, rank 2 vs n-1, ...
  const roleByIndex = useMemo(() => {
    const roles = new Map<number, "chosen" | "rejected">();
    for (let i = 0; i < clampedPairs; i++) {
      roles.set(i, "chosen");
      roles.set(order.length - 1 - i, "rejected");
    }
    return roles;
  }, [clampedPairs, order.length]);

  const handleTileClick = (index: number) => {
    if (swapIndex === null) {
      setSwapIndex(index);
      return;
    }
    if (swapIndex === index) {
      setSwapIndex(null);
      return;
    }
    const a = swapIndex;
    setSwapIndex(null);
    void onSwap(a, index);
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      await onExport();
    } finally {
      setExporting(false);
    }
  };

  const borderColor = (index: number): string => {
    if (swapIndex === index) return "var(--color-cobalt-600)";
    const role = roleByIndex.get(index);
    if (role === "chosen") return "var(--color-success-500)";
    if (role === "rejected") return "var(--color-error-500)";
    return "var(--color-border-subtle)";
  };

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
          <span>Ranked review — click two images to swap their ranks</span>
          <span className="text-[var(--color-border-subtle)]">·</span>
          <span className="tabular-nums">{pairsDone} pairs exported</span>
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

      {/* Ranked grid */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="grid grid-cols-4 gap-3">
          {order.map((path, index) => {
            const role = roleByIndex.get(index);
            const score = scores[path];
            return (
              <button
                key={path}
                type="button"
                onClick={() => handleTileClick(index)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  setPreviewPath(path);
                }}
                className="relative rounded border-2 overflow-hidden transition-all hover:shadow-lg focus:outline-none aspect-square"
                style={{ borderColor: borderColor(index) }}
                title={`${basename(path)} — left-click to swap, right-click to preview`}
              >
                <img
                  src={imageUrl(path)}
                  alt={basename(path)}
                  draggable={false}
                  loading="lazy"
                  className="w-full h-full object-cover bg-black/30"
                />
                <div className="absolute top-1 left-1 text-[10px] font-bold px-2 py-0.5 rounded bg-black/70 text-white tabular-nums">
                  #{index + 1}
                </div>
                {role && (
                  <div
                    className="absolute top-1 right-1 text-[10px] font-bold px-2 py-0.5 rounded text-white"
                    style={{
                      background: role === "chosen" ? "var(--color-success-500)" : "var(--color-error-500)",
                    }}
                  >
                    {role === "chosen" ? "Chosen" : "Rejected"}
                  </div>
                )}
                <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between gap-2 text-[10px] text-white/90 bg-black/60 px-2 py-1 font-mono">
                  <span className="truncate">{basename(path)}</span>
                  {score && <span className="tabular-nums shrink-0">{score.score} pts</span>}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Export controls */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <label className="flex items-center gap-2 text-xs text-[var(--color-on-surface-secondary)]">
          Pairs to export
          <input
            type="number"
            min={0}
            max={maxPairs}
            value={clampedPairs}
            onChange={(e) => onPairCountChange(Number(e.target.value))}
            className="w-16 px-2 py-1 rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] text-[var(--color-on-surface)] tabular-nums"
          />
          <span className="tabular-nums">(max {maxPairs}; 0 skips the group)</span>
        </label>
        <Button variant="primary" size="md" onClick={() => void handleExport()} disabled={exporting}>
          <Download className="w-4 h-4" />
          {exporting ? "Exporting…" : clampedPairs === 0 ? "Skip & Next Group" : `Export ${clampedPairs} Pairs`}
        </Button>
      </div>

      {/* Preview overlay */}
      {previewPath && (
        <PreviewOverlay path={previewPath} caption={basename(previewPath)} onClose={() => setPreviewPath(null)} />
      )}
    </div>
  );
}
