import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useCallback, useEffect } from "react";
import { createPortal } from "react-dom";

import {
  BAD_COLOR,
  basename,
  GOOD_COLOR,
  imageUrl,
  SKIP_COLOR,
  type Verdict,
  verdictColor,
  verdictLabel,
} from "./triageCommon";

export interface TriageReviewOverlayProps {
  /** Images to step through; fixed at open time so navigation stays stable. */
  list: string[];
  pos: number;
  verdictFor: (path: string) => Verdict | undefined;
  onVerdict: (path: string, verdict: Verdict) => void;
  onNavigate: (pos: number) => void;
  onClose: () => void;
}

/**
 * Full-screen rescore view for triage's pairing phase: the same mouse and
 * keyboard controls as the voting screen (LMB/1 good, RMB/0 bad, MMB/Space
 * skip), plus arrow-key navigation. Re-verdicting advances to the next image
 * in the list and closes after the last one.
 */
export function TriageReviewOverlay({
  list,
  pos,
  verdictFor,
  onVerdict,
  onNavigate,
  onClose,
}: TriageReviewOverlayProps) {
  const path = list[pos];
  const current = verdictFor(path);

  const applyVerdict = useCallback(
    (v: Verdict) => {
      onVerdict(list[pos], v);
      if (pos < list.length - 1) onNavigate(pos + 1);
      else onClose();
    },
    [list, pos, onVerdict, onNavigate, onClose],
  );

  useEffect(() => {
    // Capture phase + stopPropagation so neither TriageStep's window-level
    // handler nor ModalBase's Escape listener react to keys meant for the
    // overlay (same trick as PreviewOverlay's Escape handling).
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowLeft") {
        if (pos > 0) onNavigate(pos - 1);
      } else if (e.key === "ArrowRight") {
        if (pos < list.length - 1) onNavigate(pos + 1);
      } else if (e.key === "1") {
        applyVerdict("good");
      } else if (e.key === "0") {
        applyVerdict("bad");
      } else if (e.key === " ") {
        applyVerdict("skip");
      } else {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("keydown", handleKey, true);
    return () => window.removeEventListener("keydown", handleKey, true);
  }, [pos, list.length, applyVerdict, onNavigate, onClose]);

  const handleImageClick = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    applyVerdict("good");
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    applyVerdict("bad");
  };

  const handleAuxClick = (e: React.MouseEvent) => {
    if (e.button !== 1) return;
    e.preventDefault();
    applyVerdict("skip");
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    // Suppress middle-click autoscroll so MMB is a clean "skip".
    if (e.button === 1) e.preventDefault();
  };

  const verdictPill = (v: Verdict, label: string, color: string, bg: string) => (
    <div
      role="button"
      tabIndex={-1}
      onClick={() => applyVerdict(v)}
      className={`px-3 py-1 rounded cursor-pointer font-semibold ${current === v ? "ring-2 ring-white/70" : ""}`}
      style={{ color, background: bg, border: `1px solid ${color}55` }}
    >
      {label}
    </div>
  );

  return createPortal(
    <div className="fixed inset-0 z-[60] flex flex-col" style={{ background: "rgba(0, 0, 0, 0.95)" }}>
      <div
        className="relative flex-1 min-h-0 flex items-center justify-center overflow-hidden cursor-pointer select-none"
        onClick={handleImageClick}
        onContextMenu={handleContextMenu}
        onAuxClick={handleAuxClick}
        onMouseDown={handleMouseDown}
      >
        <img
          key={path}
          src={imageUrl(path)}
          alt={basename(path)}
          className="w-full h-full object-contain"
          fetchPriority="high"
          draggable={false}
        />
        {current && (
          <div
            className="absolute top-3 left-3 px-2.5 py-1 rounded text-xs font-bold pointer-events-none"
            style={{
              background: "rgba(0, 0, 0, 0.7)",
              color: verdictColor(current),
              border: `1px solid ${verdictColor(current)}`,
            }}
          >
            {verdictLabel(current)}
          </div>
        )}
        {pos > 0 && (
          <div
            role="button"
            tabIndex={-1}
            className="absolute left-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/60 text-white hover:bg-black/85 cursor-pointer"
            title="Previous (←)"
            onClick={(e) => {
              e.stopPropagation();
              onNavigate(pos - 1);
            }}
          >
            <ChevronLeft className="w-6 h-6" />
          </div>
        )}
        {pos < list.length - 1 && (
          <div
            role="button"
            tabIndex={-1}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/60 text-white hover:bg-black/85 cursor-pointer"
            title="Next (→)"
            onClick={(e) => {
              e.stopPropagation();
              onNavigate(pos + 1);
            }}
          >
            <ChevronRight className="w-6 h-6" />
          </div>
        )}
        <div
          role="button"
          tabIndex={-1}
          className="absolute top-3 right-3 p-2 rounded-full bg-black/60 text-white hover:bg-black/85 cursor-pointer"
          title="Close (Esc)"
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
        >
          <X className="w-5 h-5" />
        </div>
      </div>

      <div className="px-4 py-2.5 bg-black/80 text-white text-xs flex items-center justify-between gap-3 flex-wrap">
        <span className="font-mono truncate">{basename(path)}</span>
        <div className="flex items-center gap-2">
          {verdictPill("good", "Good — LMB / 1", GOOD_COLOR, "rgba(34,197,94,0.15)")}
          {verdictPill("bad", "Bad — RMB / 0", BAD_COLOR, "rgba(239,68,68,0.15)")}
          {verdictPill("skip", "Skip — MMB / Space", SKIP_COLOR, "rgba(156,163,175,0.15)")}
        </div>
        <span className="text-white/60 tabular-nums">
          {pos + 1} / {list.length} · Esc to close
        </span>
      </div>
    </div>,
    document.body,
  );
}
