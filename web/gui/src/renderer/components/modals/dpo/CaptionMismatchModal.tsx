import { AlertTriangle, ArrowLeft, ArrowRight, Check, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { type CaptionMismatch, dpoApi } from "@/api/dpoApi";
import { Button } from "@/components/shared";

import { ModalBase } from "../ModalBase";
import { PreviewOverlay } from "./PreviewOverlay";

export interface CaptionMismatchModalProps {
  open: boolean;
  mismatches: CaptionMismatch[];
  onClose: () => void;
}

export function CaptionMismatchModal({ open, mismatches: initial, onClose }: CaptionMismatchModalProps) {
  const [mismatches, setMismatches] = useState<CaptionMismatch[]>(initial);
  const [index, setIndex] = useState(0);
  const [corrected, setCorrected] = useState(0);
  const [discarded, setDiscarded] = useState(0);
  const [customText, setCustomText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [previewPath, setPreviewPath] = useState<string | null>(null);

  // Reset local state every time the modal opens with a fresh mismatch set.
  useEffect(() => {
    if (open) {
      setMismatches(initial);
      setIndex(0);
      setCorrected(0);
      setDiscarded(0);
      setError(null);
    }
  }, [open, initial]);

  const current = mismatches[index] ?? null;

  // Pre-fill the custom textarea with the chosen caption whenever we move to a
  // new mismatch entry. Pre-fill with chosen caption is the user's chosen seed.
  useEffect(() => {
    setCustomText(current?.chosen_caption ?? "");
  }, [current]);

  const advanceAfterChange = useCallback((newList: CaptionMismatch[]) => {
    setMismatches(newList);
    if (newList.length === 0) {
      setIndex(0);
      return;
    }
    setIndex((prev) => Math.min(prev, newList.length - 1));
  }, []);

  const apply = useCallback(
    async (caption: string) => {
      if (!current) return;
      setBusy(true);
      setError(null);
      try {
        const res = await dpoApi.applyCaption(current.chosen_image, current.rejected_image, caption);
        if (!res.ok) {
          setError(res.error ?? "Failed to apply caption");
          return;
        }
        setCorrected((c) => c + 1);
        advanceAfterChange(mismatches.filter((_, i) => i !== index));
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(false);
      }
    },
    [current, mismatches, index, advanceAfterChange],
  );

  const discard = useCallback(async () => {
    if (!current) return;
    setBusy(true);
    setError(null);
    try {
      const res = await dpoApi.removePair(current.chosen_image, current.rejected_image);
      if (!res.ok) {
        setError("Failed to discard pair");
        return;
      }
      setDiscarded((d) => d + 1);
      advanceAfterChange(mismatches.filter((_, i) => i !== index));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [current, mismatches, index, advanceAfterChange]);

  const goBack = useCallback(() => {
    setIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const goSkip = useCallback(() => {
    setIndex((prev) => (prev + 1 < mismatches.length ? prev + 1 : prev));
  }, [mismatches.length]);

  const total = mismatches.length;
  const remaining = total - index;
  const allDone = total === 0;

  return (
    <ModalBase open={open} onClose={onClose} title="Resolve Caption Mismatches" size="full" closeOnBackdrop={false}>
      <div className="flex flex-col gap-3 h-[calc(100vh-180px)] min-h-[600px]">
        <div className="flex items-center gap-4 text-sm">
          {!allDone ? (
            <>
              <span className="font-semibold text-[var(--color-on-surface)]">
                Pair {index + 1} / {total}
              </span>
              <span className="text-[var(--color-on-surface-secondary)] truncate font-mono text-xs">
                {current?.key}
              </span>
            </>
          ) : (
            <span className="font-semibold text-[var(--color-success-500)]">All mismatches resolved.</span>
          )}
          <span className="ml-auto text-[var(--color-success-500)]">Corrected: {corrected}</span>
          <span className="text-[var(--color-error-500)]">Discarded: {discarded}</span>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)]">
            <AlertTriangle className="w-4 h-4" />
            {error}
          </div>
        )}

        {!allDone && current && (
          <>
            <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
              <div className="flex flex-col rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] overflow-hidden">
                <div className="px-3 py-1.5 text-sm font-semibold text-[var(--color-success-500)]">Chosen</div>
                <button
                  type="button"
                  className="flex-1 min-h-0 flex items-center justify-center bg-black/30 cursor-zoom-in"
                  onClick={() => setPreviewPath(current.chosen_image)}
                >
                  <img
                    src={dpoApi.imageUrl(current.chosen_image)}
                    alt="chosen"
                    className="max-w-full max-h-full object-contain"
                  />
                </button>
              </div>
              <div className="flex flex-col rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] overflow-hidden">
                <div className="px-3 py-1.5 text-sm font-semibold text-[var(--color-error-500)]">Rejected</div>
                <button
                  type="button"
                  className="flex-1 min-h-0 flex items-center justify-center bg-black/30 cursor-zoom-in"
                  onClick={() => setPreviewPath(current.rejected_image)}
                >
                  <img
                    src={dpoApi.imageUrl(current.rejected_image)}
                    alt="rejected"
                    className="max-w-full max-h-full object-contain"
                  />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <textarea
                  readOnly
                  value={current.chosen_caption}
                  className="w-full h-20 resize-none rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] p-2 text-sm font-mono text-[var(--color-on-surface)]"
                />
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={busy}
                  onClick={() => void apply(current.chosen_caption)}
                >
                  <ArrowLeft className="w-4 h-4" />
                  Use this caption
                </Button>
              </div>
              <div className="flex flex-col gap-1">
                <textarea
                  readOnly
                  value={current.rejected_caption}
                  className="w-full h-20 resize-none rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] p-2 text-sm font-mono text-[var(--color-on-surface)]"
                />
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={busy}
                  onClick={() => void apply(current.rejected_caption)}
                >
                  Use this caption
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs text-[var(--color-on-surface-secondary)]">
                Custom caption (applied to both sides; pre-filled with chosen):
              </label>
              <textarea
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                className="w-full h-24 resize-none rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] p-2 text-sm font-mono text-[var(--color-on-surface)]"
              />
              <Button size="sm" variant="primary" disabled={busy} onClick={() => void apply(customText)}>
                <Check className="w-4 h-4" />
                Apply Custom Caption
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" disabled={busy || index === 0} onClick={goBack}>
                <ArrowLeft className="w-4 h-4" />
                Back
              </Button>
              <Button size="sm" variant="danger" disabled={busy} onClick={() => void discard()}>
                <Trash2 className="w-4 h-4" />
                Discard Pair
              </Button>
              <div className="flex-1" />
              <span className="text-xs text-[var(--color-on-surface-secondary)]">{remaining} remaining</span>
              <Button size="sm" variant="ghost" disabled={busy || index >= mismatches.length - 1} onClick={goSkip}>
                Skip
                <ArrowRight className="w-4 h-4" />
              </Button>
            </div>
          </>
        )}

        {allDone && (
          <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center">
            <Check className="w-12 h-12 text-[var(--color-success-500)]" />
            <div className="text-lg font-semibold text-[var(--color-on-surface)]">Caption mismatch review complete</div>
            <div className="text-sm text-[var(--color-on-surface-secondary)]">
              Corrected: {corrected} Discarded: {discarded}
            </div>
          </div>
        )}

        <div className="flex justify-end pt-2 border-t border-[var(--color-border-subtle)]">
          <Button size="sm" variant="ghost" onClick={onClose}>
            <X className="w-4 h-4" />
            Close
          </Button>
        </div>
      </div>
      {previewPath && <PreviewOverlay path={previewPath} onClose={() => setPreviewPath(null)} />}
    </ModalBase>
  );
}
