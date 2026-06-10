import { ArrowDown, ArrowLeft, ArrowRight, Flag, Loader2, SkipForward, X } from "lucide-react";
import { useEffect, useState } from "react";

import { API_BASE } from "@/api/request";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "./PreviewOverlay";
import { PromptExpander } from "./PromptExpander";
import type { TournamentStepProps } from "./types";

function basename(p: string): string {
  return p.split(/[/\\]/).pop() ?? p;
}

function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

export function TournamentStep({
  group,
  match,
  round,
  totalRounds,
  matchesPlayed,
  matchesTotal,
  onVote,
  onFinishEarly,
  onSkipGroup,
  onCancel,
  pairsDone,
}: TournamentStepProps) {
  const [previewPath, setPreviewPath] = useState<string | null>(null);

  // Clear preview if the currently previewed path left the match
  useEffect(() => {
    if (!previewPath) return;
    if (!match || !match.includes(previewPath)) {
      setPreviewPath(null);
    }
  }, [previewPath, match]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (previewPath || !match) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        void onVote("a");
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        void onVote("b");
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        void onVote("tie");
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [previewPath, match, onVote]);

  const handlePreviewPick = async () => {
    if (!match || !previewPath) {
      setPreviewPath(null);
      return;
    }
    const idx = match.indexOf(previewPath);
    setPreviewPath(null);
    if (idx >= 0) {
      await onVote(idx === 0 ? "a" : "b");
    }
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
          <span className="tabular-nums">
            Round {round}/{totalRounds}
          </span>
          <span className="text-[var(--color-border-subtle)]">·</span>
          <span className="tabular-nums">
            Match {Math.min(matchesPlayed + 1, matchesTotal)}/{matchesTotal}
          </span>
          <span className="text-[var(--color-border-subtle)]">·</span>
          <span className="tabular-nums">{pairsDone} pairs exported</span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => void onFinishEarly()}>
            <Flag className="w-4 h-4" />
            Finish Early
          </Button>
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

      {match ? (
        <>
          {/* Image comparison grid */}
          <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
            {match.map((path, idx) => (
              <button
                key={path}
                type="button"
                onClick={() => setPreviewPath(path)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  void onVote(idx === 0 ? "a" : "b");
                }}
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
              Tie (↓)
            </Button>
            <Button variant="primary" size="md" onClick={() => void onVote("b")}>
              B is Better (→)
              <ArrowRight className="w-4 h-4" />
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
