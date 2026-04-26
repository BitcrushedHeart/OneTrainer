import { ArrowLeft, ArrowRight, CheckCircle2, Loader2, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { dpoApi } from "@/api/dpoApi";
import { API_BASE } from "@/api/request";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "./PreviewOverlay";
import { PromptExpander } from "./PromptExpander";
import type { ReviewPair, ReviewStepProps } from "./types";

function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

function basename(path: string): string {
  return path.split(/[/\\]/).pop() ?? path;
}

function isMissing(pair: ReviewPair): boolean {
  return !pair.chosen_path || !pair.rejected_path;
}

export function ReviewStep({ onBack, onClose }: ReviewStepProps) {
  const [pairs, setPairs] = useState<ReviewPair[] | null>(null);
  const [initialCount, setInitialCount] = useState(0);
  const [index, setIndex] = useState(0);
  const [removed, setRemoved] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [orphanPromptVisible, setOrphanPromptVisible] = useState(false);
  const [orphans, setOrphans] = useState<ReviewPair[]>([]);
  const [orphanBusy, setOrphanBusy] = useState(false);

  // Refs to avoid stale closures inside the window keydown handler.
  const pairsRef = useRef<ReviewPair[] | null>(pairs);
  const indexRef = useRef(index);
  const previewRef = useRef(previewPath);
  const orphanRef = useRef(orphanPromptVisible);

  useEffect(() => {
    pairsRef.current = pairs;
  }, [pairs]);
  useEffect(() => {
    indexRef.current = index;
  }, [index]);
  useEffect(() => {
    previewRef.current = previewPath;
  }, [previewPath]);
  useEffect(() => {
    orphanRef.current = orphanPromptVisible;
  }, [orphanPromptVisible]);

  const load = useCallback(async () => {
    try {
      const res = await dpoApi.reviewPairs();
      if (!res.ok || !res.pairs) {
        setError(res.error ?? "Failed to load pairs");
        setPairs([]);
        return;
      }
      const all = res.pairs;
      setInitialCount(all.length);
      const missing = all.filter(isMissing);
      if (missing.length > 0) {
        setOrphans(missing);
        setOrphanPromptVisible(true);
        setPairs(all);
      } else {
        setPairs(all);
      }
      setIndex(0);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPairs([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const handleRemove = useCallback(async () => {
    const current = pairsRef.current;
    const i = indexRef.current;
    if (!current || i < 0 || i >= current.length) return;
    const p = current[i];
    try {
      await dpoApi.removePair(p.chosen_path, p.rejected_path);
      const next = [...current.slice(0, i), ...current.slice(i + 1)];
      setPairs(next);
      setRemoved((r) => r + 1);
      if (i >= next.length) {
        setIndex(Math.max(0, next.length - 1));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const goPrev = useCallback(() => {
    setIndex((i) => (i > 0 ? i - 1 : i));
  }, []);

  const goNext = useCallback(() => {
    const current = pairsRef.current;
    if (!current) return;
    setIndex((i) => (i < current.length - 1 ? i + 1 : i));
  }, []);

  // Keyboard shortcuts (only when no preview and no orphan prompt).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (previewRef.current !== null) return;
      if (orphanRef.current) return;
      const current = pairsRef.current;
      if (!current || current.length === 0) return;
      if (e.key === "ArrowLeft") {
        if (indexRef.current > 0) {
          e.preventDefault();
          goPrev();
        }
      } else if (e.key === "ArrowRight") {
        if (indexRef.current < current.length - 1) {
          e.preventDefault();
          goNext();
        }
      } else if (e.key === "Delete") {
        e.preventDefault();
        void handleRemove();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goPrev, goNext, handleRemove]);

  const handleConfirmOrphans = useCallback(async () => {
    setOrphanBusy(true);
    try {
      const current = pairsRef.current ?? [];
      const toRemove = orphans;
      for (const o of toRemove) {
        try {
          await dpoApi.removePair(o.chosen_path || null, o.rejected_path || null);
        } catch {
          // Continue regardless; orphan entries can be malformed.
        }
      }
      const filtered = current.filter((p) => !isMissing(p));
      setPairs(filtered);
      setRemoved((r) => r + toRemove.length);
      setOrphans([]);
      setOrphanPromptVisible(false);
      setIndex(0);
    } finally {
      setOrphanBusy(false);
    }
  }, [orphans]);

  const handleKeepOrphans = useCallback(() => {
    setOrphans([]);
    setOrphanPromptVisible(false);
  }, []);

  // ---------- Render: Loading ----------
  if (pairs === null) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3 text-[var(--color-on-surface-secondary)]">
        <Loader2 className="w-8 h-8 animate-spin" />
        <div className="text-sm">Loading pairs...</div>
      </div>
    );
  }

  // ---------- Render: Error ----------
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-4">
        <div className="text-sm text-[var(--color-error-500)] font-semibold">Error</div>
        <div className="text-xs text-[var(--color-on-surface-secondary)] max-w-md text-center">{error}</div>
        <Button variant="secondary" onClick={onBack}>
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
      </div>
    );
  }

  // ---------- Render: Empty / Summary ----------
  // An empty list on entry ("No pairs found.") is indistinguishable from
  // "finished all pairs". We present the Review Complete summary when any
  // removal happened, otherwise a plain empty-state.
  if (pairs.length === 0) {
    if (removed > 0) {
      return (
        <div className="flex flex-col items-center justify-center py-16 gap-4">
          <CheckCircle2 className="w-12 h-12 text-[var(--color-success-500)]" />
          <div className="text-xl font-semibold text-[var(--color-on-surface)]">Review Complete</div>
          <div className="text-sm text-[var(--color-on-surface)]">
            Kept: {Math.max(0, initialCount - removed)} pairs
          </div>
          <div className="text-sm text-[var(--color-error-500)]">Removed: {removed} pairs</div>
          <div className="flex gap-3 mt-2">
            <Button variant="primary" onClick={onBack}>
              Back to Start
            </Button>
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      );
    }
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-4">
        <div className="text-sm text-[var(--color-on-surface-secondary)]">No pairs found.</div>
        <Button variant="secondary" onClick={onBack}>
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
      </div>
    );
  }

  // ---------- Render: Pair index clamped ----------
  const safeIndex = Math.min(index, pairs.length - 1);
  const pair = pairs[safeIndex];

  return (
    <div className="flex flex-col gap-3 h-full min-h-0 relative">
      {/* Header */}
      <div className="flex items-center gap-4 pb-2 border-b border-[var(--color-border-subtle)]">
        <div className="text-sm font-bold text-[var(--color-on-surface)]">
          Pair {safeIndex + 1} / {pairs.length}
        </div>
        <div className="text-xs text-[var(--color-on-surface-secondary)] truncate max-w-[300px]">
          Key: {pair.prompt_key}
        </div>
        <div className="text-xs text-[var(--color-error-500)] ml-auto">Removed: {removed}</div>
      </div>

      {/* Caption */}
      <PromptExpander prompt={pair.caption || "(no caption)"} />

      {/* Images */}
      <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
        <div className="flex flex-col gap-1 min-h-0">
          <div className="text-sm font-semibold text-[var(--color-success-500)]">Chosen</div>
          <button
            type="button"
            className="flex-1 min-h-0 relative rounded overflow-hidden border border-[var(--color-border-subtle)] cursor-zoom-in bg-black/30 p-0"
            onClick={() => setPreviewPath(pair.chosen_path)}
          >
            {pair.chosen_path ? (
              <img
                src={imageUrl(pair.chosen_path)}
                className="w-full h-full object-contain"
                alt="chosen"
                draggable={false}
              />
            ) : (
              <div className="flex items-center justify-center w-full h-full text-[var(--color-on-surface-secondary)] text-sm">
                (missing)
              </div>
            )}
          </button>
          <div className="text-xs font-mono text-[var(--color-on-surface-secondary)] truncate" title={pair.chosen_path}>
            {pair.chosen_path ? basename(pair.chosen_path) : "(missing)"}
          </div>
        </div>

        <div className="flex flex-col gap-1 min-h-0">
          <div className="text-sm font-semibold text-[var(--color-error-500)]">Rejected</div>
          <button
            type="button"
            className="flex-1 min-h-0 relative rounded overflow-hidden border border-[var(--color-border-subtle)] cursor-zoom-in bg-black/30 p-0"
            onClick={() => setPreviewPath(pair.rejected_path)}
          >
            {pair.rejected_path ? (
              <img
                src={imageUrl(pair.rejected_path)}
                className="w-full h-full object-contain"
                alt="rejected"
                draggable={false}
              />
            ) : (
              <div className="flex items-center justify-center w-full h-full text-[var(--color-on-surface-secondary)] text-sm">
                (missing)
              </div>
            )}
          </button>
          <div
            className="text-xs font-mono text-[var(--color-on-surface-secondary)] truncate"
            title={pair.rejected_path}
          >
            {pair.rejected_path ? basename(pair.rejected_path) : "(missing)"}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between gap-2 pt-2 border-t border-[var(--color-border-subtle)]">
        <Button variant="ghost" onClick={goPrev} disabled={safeIndex === 0}>
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
        <div className="flex items-center gap-2">
          <Button variant="danger" onClick={() => void handleRemove()}>
            <Trash2 className="w-4 h-4" />
            Remove
          </Button>
          <Button variant="secondary" onClick={goNext} disabled={safeIndex >= pairs.length - 1}>
            Keep
            <ArrowRight className="w-4 h-4" />
          </Button>
        </div>
        <Button variant="ghost" onClick={onBack}>
          <X className="w-4 h-4" />
          Close
        </Button>
      </div>

      {/* Orphan confirmation overlay */}
      {orphanPromptVisible && (
        <div
          className="absolute inset-0 z-20 flex items-center justify-center"
          style={{ background: "rgba(0,0,0,0.55)" }}
        >
          <div
            className="w-[440px] max-w-[90%] rounded border border-[var(--color-border-subtle)] p-5 flex flex-col gap-4"
            style={{ background: "var(--color-surface-container)" }}
          >
            <div className="text-sm font-semibold text-[var(--color-warning-500)]">Orphaned Pairs Found</div>
            <div className="text-xs text-[var(--color-on-surface)]">
              Found {orphans.length} pair(s) with a missing chosen or rejected image. Remove them now?
            </div>
            <div className="flex items-center justify-end gap-2">
              <Button variant="ghost" onClick={handleKeepOrphans} disabled={orphanBusy}>
                No
              </Button>
              <Button variant="danger" onClick={() => void handleConfirmOrphans()} loading={orphanBusy}>
                Yes, remove
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Full-size preview */}
      {previewPath && (
        <PreviewOverlay path={previewPath} caption={basename(previewPath)} onClose={() => setPreviewPath(null)} />
      )}
    </div>
  );
}
