import { AlertCircle, CheckCircle2, Loader2, Wand2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { dpoApi, type RepairRejectedResult } from "@/api/dpoApi";
import { Button, FormEntry } from "@/components/shared";

import { ModalBase } from "./ModalBase";

export interface RepairRejectedModalProps {
  open: boolean;
  onClose: () => void;
}

interface Progress {
  phase: string;
  imagesDone: number;
  imagesTotal: number;
  groupsDone: number;
  groupsTotal: number;
  pairsRepaired: number;
}

const PHASE_LABEL: Record<string, string> = {
  starting: "Scanning concept folders…",
  embedding: "Embedding images with DINOv2…",
  done: "Done",
  error: "Failed",
};

export function RepairRejectedModal({ open, onClose }: RepairRejectedModalProps) {
  const [floor, setFloor] = useState<number>(0.85);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [result, setResult] = useState<RepairRejectedResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const applyStatus = useCallback(
    (s: Awaited<ReturnType<typeof dpoApi.repairStatus>>) => {
      setProgress({
        phase: s.phase ?? "",
        imagesDone: s.images_done ?? 0,
        imagesTotal: s.images_total ?? 0,
        groupsDone: s.groups_done ?? 0,
        groupsTotal: s.groups_total ?? 0,
        pairsRepaired: s.pairs_repaired ?? 0,
      });
      setRunning(s.running);
      if (!s.running) {
        clearPoll();
        if (s.error) setError(s.error);
        else if (s.summary) setResult(s.summary);
      }
    },
    [clearPoll],
  );

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(() => {
      void dpoApi
        .repairStatus()
        .then((s) => {
          if (s.ok) applyStatus(s);
        })
        .catch(() => {
          /* transient — keep polling */
        });
    }, 600);
  }, [applyStatus, clearPoll]);

  // On open, reflect any job that's already running (e.g. a long run started
  // earlier) or a finished result, so reopening the modal never looks dead.
  useEffect(() => {
    if (!open) {
      clearPoll();
      return;
    }
    let cancelled = false;
    void dpoApi.repairStatus().then((s) => {
      if (cancelled || !s.ok) return;
      applyStatus(s);
      if (s.running) startPolling();
    });
    return () => {
      cancelled = true;
    };
  }, [open, applyStatus, startPolling, clearPoll]);

  useEffect(() => () => clearPoll(), [clearPoll]);

  const run = useCallback(async () => {
    const f = Number(floor);
    if (!Number.isFinite(f) || f < 0 || f > 1) {
      setError("Quality floor must be between 0 and 1.");
      return;
    }
    setError(null);
    setResult(null);
    setProgress(null);
    setRunning(true);
    try {
      const res = await dpoApi.repairRejected(f);
      if (!res.ok) {
        setRunning(false);
        setError(res.error ?? "Re-pair failed");
        return;
      }
      startPolling();
    } catch (e) {
      setRunning(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [floor, startPolling]);

  const pct =
    progress && progress.imagesTotal > 0
      ? Math.min(100, Math.round((progress.imagesDone / progress.imagesTotal) * 100))
      : 0;

  return (
    <ModalBase open={open} onClose={onClose} title="Re-pair Rejected by Similarity" size="lg">
      <div className="flex flex-col gap-4">
        <div className="text-sm text-[var(--color-on-surface-secondary)]">
          Within each caption group on your configured DPO concepts, rejected images are renamed so each is paired with
          the chosen image it most resembles (DINOv2 structural similarity, optimal assignment). This sharpens the DPO
          signal by holding composition constant so the gradient keys on the quality difference. Single-pair captions
          are left untouched and the operation keeps a strict bijection — no pairs are lost.
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-on-surface-secondary)] bg-[var(--color-surface-container)] rounded p-3">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>
            Rejected image files are renamed in place. The first run downloads ~330MB of model weights and embeds every
            image, which can take several minutes on a large dataset. Back up your dataset first.
          </span>
        </div>

        <div className="grid grid-cols-4 gap-3 items-end">
          <FormEntry
            label="Quality floor (0–1)"
            type="number"
            value={floor}
            onChange={(v) => setFloor(Number(v))}
            disabled={running}
          />
          <div className="col-span-3 flex items-end">
            <Button variant="primary" onClick={() => void run()} disabled={running}>
              {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
              {running ? "Re-pairing…" : "Re-pair"}
            </Button>
          </div>
        </div>
        <div className="text-xs text-[var(--color-on-surface-secondary)] -mt-2">
          <span className="font-semibold">1.0</span> = never make any pair less similar than it already is (safe). Lower
          values let a pair be reassigned to a slightly-less-similar image to unlock a bigger gain elsewhere, but never
          below this cosine. <span className="font-semibold">0.85</span> is a good middle ground;{" "}
          <span className="font-semibold">0</span> = pure global optimum (may worsen some pairs).
        </div>

        {running && progress && (
          <div className="flex flex-col gap-2 bg-[var(--color-surface-container)] rounded p-3">
            <div className="flex items-center justify-between text-sm text-[var(--color-on-surface)]">
              <span className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                {PHASE_LABEL[progress.phase] ?? "Working…"}
              </span>
              <span className="tabular-nums text-[var(--color-on-surface-secondary)]">{pct}%</span>
            </div>
            <div className="h-2 rounded bg-[var(--color-border-subtle)] overflow-hidden">
              <div className="h-full bg-[var(--color-cobalt-600)] transition-all" style={{ width: `${pct}%` }} />
            </div>
            <div className="text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
              {progress.imagesDone} / {progress.imagesTotal} images
              {progress.groupsTotal > 0 && (
                <>
                  {" "}
                  &middot; group {progress.groupsDone} / {progress.groupsTotal}
                </>
              )}
              {progress.pairsRepaired > 0 && <> &middot; {progress.pairsRepaired} re-paired so far</>}
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)] bg-[var(--color-surface-container)] rounded p-3">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}

        {result && !running && (
          <div className="flex items-start gap-2 text-sm text-[var(--color-on-surface)] bg-[var(--color-surface-container)] rounded p-3">
            <CheckCircle2 className="w-4 h-4 mt-0.5 text-[var(--color-success-500)]" />
            <div>
              Re-paired <span className="font-semibold">{result.pairs_repaired}</span> rejected image(s) across{" "}
              <span className="font-semibold">{result.groups_processed}</span> multi-pair caption group(s).
              <div className="text-xs text-[var(--color-on-surface-secondary)] pt-1">
                Single-pair groups skipped: {result.groups_skipped_single} &middot; Total matched pairs considered:{" "}
                {result.pairs_total}
              </div>
            </div>
          </div>
        )}
      </div>
    </ModalBase>
  );
}
