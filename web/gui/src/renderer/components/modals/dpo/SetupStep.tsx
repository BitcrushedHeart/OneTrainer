import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  FileWarning,
  MousePointer2,
  Scissors,
  Search,
  Wrench,
  Zap,
} from "lucide-react";
import { useCallback, useState } from "react";

import { type CaptionMismatch, dpoApi, type PairCheckResult } from "@/api/dpoApi";
import { Button, DirPicker, FormEntry, SectionCard } from "@/components/shared";

import { CaptionMismatchModal } from "./CaptionMismatchModal";
import type { SetupStepProps } from "./types";

function CheckResults({ result }: { result: PairCheckResult }) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="p-2 rounded bg-[var(--color-surface-container)]">
          <span className="text-[var(--color-on-surface-secondary)]">Matched: </span>
          <span className="font-medium text-[var(--color-success-500)]">{result.total_matched}</span>
        </div>
        <div className="p-2 rounded bg-[var(--color-surface-container)]">
          <span className="text-[var(--color-on-surface-secondary)]">Chosen strays: </span>
          <span className="font-medium text-[var(--color-warning-500)]">{result.total_chosen_stray}</span>
        </div>
        <div className="p-2 rounded bg-[var(--color-surface-container)]">
          <span className="text-[var(--color-on-surface-secondary)]">Rejected strays: </span>
          <span className="font-medium text-[var(--color-warning-500)]">{result.total_rejected_stray}</span>
        </div>
      </div>
      {result.multiline_captions > 0 && (
        <div className="flex items-center gap-2 text-sm text-[var(--color-warning-500)]">
          <AlertTriangle className="w-4 h-4" />
          {result.multiline_captions} caption(s) contain newlines
        </div>
      )}
      {Object.keys(result.format_stats).length > 0 && (
        <div className="text-xs text-[var(--color-on-surface-secondary)]">
          Formats:{" "}
          {Object.entries(result.format_stats)
            .map(([ext, count]) => `${ext}: ${count}`)
            .join(", ")}
        </div>
      )}
    </div>
  );
}

export function SetupStep({
  sourceFolder,
  outputDir,
  pairsPerGroup,
  onSourceChange,
  onOutputChange,
  onPairsPerGroupChange,
  onStart,
  onReview,
  onClose,
  resume,
  onSelectOutput,
  error,
}: SetupStepProps) {
  const [managementOpen, setManagementOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [checkResult, setCheckResult] = useState<PairCheckResult | null>(null);
  const [toolMessage, setToolMessage] = useState<string | null>(null);
  const [toolError, setToolError] = useState<string | null>(null);
  const [mismatches, setMismatches] = useState<CaptionMismatch[] | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);

  const canStart = sourceFolder.trim() !== "" && outputDir.trim() !== "";

  const clearToolState = () => {
    setToolMessage(null);
    setToolError(null);
  };

  const refreshMismatches = useCallback(async () => {
    const res = await dpoApi.checkCaptionMismatches();
    if (res.ok) {
      setMismatches(res.mismatches ?? []);
    } else {
      setMismatches(null);
    }
  }, []);

  const handleCheckPairs = useCallback(async () => {
    clearToolState();
    setLoading(true);
    try {
      const res = await dpoApi.checkPairs();
      if (res.ok && res.result) {
        setCheckResult(res.result);
        await refreshMismatches();
      } else {
        setToolError(res.error ?? "Check failed");
      }
    } catch (e) {
      setToolError(String(e));
    } finally {
      setLoading(false);
    }
  }, [refreshMismatches]);

  const handleCorrectAllToChosen = useCallback(async () => {
    clearToolState();
    setLoading(true);
    try {
      const res = await dpoApi.correctAllToChosen();
      if (res.ok) {
        setToolMessage(`Corrected ${res.corrected ?? 0} caption(s) to chosen.`);
        await refreshMismatches();
      } else {
        setToolError(res.error ?? "Correction failed");
      }
    } catch (e) {
      setToolError(String(e));
    } finally {
      setLoading(false);
    }
  }, [refreshMismatches]);

  const handleManualReviewClose = useCallback(async () => {
    setReviewOpen(false);
    setLoading(true);
    try {
      const check = await dpoApi.checkPairs();
      if (check.ok && check.result) setCheckResult(check.result);
      await refreshMismatches();
    } finally {
      setLoading(false);
    }
  }, [refreshMismatches]);

  const handleRemoveStrays = useCallback(async () => {
    clearToolState();
    setLoading(true);
    try {
      const res = await dpoApi.removeStrays();
      if (res.ok) {
        setToolMessage(`Removed ${res.removed ?? 0} stray file(s).`);
        const check = await dpoApi.checkPairs();
        if (check.ok && check.result) setCheckResult(check.result);
      }
    } catch (e) {
      setToolError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFixCaptions = useCallback(async () => {
    clearToolState();
    setLoading(true);
    try {
      const res = await dpoApi.fixCaptions();
      if (res.ok) {
        setToolMessage(`Fixed ${res.fixed ?? 0} multiline caption(s).`);
      }
    } catch (e) {
      setToolError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleOutputChange = (v: string) => {
    onOutputChange(v);
    onSelectOutput(v);
  };

  return (
    <div className="space-y-5">
      <div className="text-center space-y-1">
        <h2 className="text-2xl font-bold text-[var(--color-on-surface)]">DPO Pair Tool</h2>
        <p className="text-sm text-[var(--color-on-surface-secondary)]">
          Curate chosen/rejected pairs from generated images.
        </p>
      </div>

      <SectionCard title="Folders">
        <div className="space-y-3">
          <DirPicker label="Source Folder" value={sourceFolder} onChange={onSourceChange} />
          <DirPicker label="Output Folder" value={outputDir} onChange={handleOutputChange} />
        </div>
      </SectionCard>

      {resume && (resume.existingPairs > 0 || resume.pruned > 0) && (
        <div
          className="flex items-start gap-2 text-sm p-3 rounded border"
          style={{
            color: "#22c55e",
            borderColor: "rgba(34,197,94,0.3)",
            background: "rgba(34,197,94,0.08)",
          }}
        >
          <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
          <div>
            {resume.existingPairs > 0 && (
              <div>Found {resume.existingPairs} existing pairs — completed groups will be skipped.</div>
            )}
            {resume.pruned > 0 && <div>(Removed {resume.pruned} orphaned entries with missing files.)</div>}
          </div>
        </div>
      )}

      <SectionCard title="Settings">
        <div className="flex items-end gap-3">
          <div className="w-32">
            <FormEntry
              label="Pairs per group"
              type="number"
              value={pairsPerGroup}
              onChange={(v) => onPairsPerGroupChange(Math.max(1, Number(v)))}
            />
          </div>
          <p className="text-xs text-[var(--color-on-surface-secondary)] pb-2">
            How many pairs to collect before moving on. Triage mode ignores this — it pairs as many good/bad images as
            each group allows.
          </p>
        </div>
      </SectionCard>

      {error && (
        <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)]">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          disabled={!canStart}
          onClick={() => void onStart({ sourceFolder, outputDir, pairsPerGroup, mode: "selection" })}
        >
          <MousePointer2 className="w-4 h-4" />
          Start (Selection)
        </Button>
        <Button
          variant="primary"
          disabled={!canStart}
          onClick={() => void onStart({ sourceFolder, outputDir, pairsPerGroup, mode: "elo" })}
        >
          <MousePointer2 className="w-4 h-4" />
          Start (ELO)
        </Button>
        <Button
          variant="primary"
          disabled={!canStart}
          onClick={() => void onStart({ sourceFolder, outputDir, pairsPerGroup, mode: "triage" })}
        >
          <Zap className="w-4 h-4" />
          Start (Triage)
        </Button>
        <Button variant="secondary" onClick={onReview}>
          <Eye className="w-4 h-4" />
          Review Pairs
        </Button>
        <div className="flex-1" />
        <Button variant="ghost" onClick={onClose}>
          Close
        </Button>
      </div>

      <details
        className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)]"
        open={managementOpen}
        onToggle={(e) => setManagementOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary className="cursor-pointer select-none px-4 py-2 text-sm font-semibold text-[var(--color-on-surface)]">
          Management Tools
        </summary>
        <div className="px-4 pb-4 space-y-3">
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="secondary" onClick={handleCheckPairs} disabled={loading}>
              <Search className="w-4 h-4" />
              Check Pairs
            </Button>
            <Button size="sm" variant="secondary" onClick={handleRemoveStrays} disabled={loading}>
              <Scissors className="w-4 h-4" />
              Remove Strays
            </Button>
            <Button size="sm" variant="secondary" onClick={handleFixCaptions} disabled={loading}>
              <Wrench className="w-4 h-4" />
              Fix Captions
            </Button>
          </div>
          {toolMessage && (
            <div className="flex items-center gap-2 text-sm text-[var(--color-success-500)]">
              <CheckCircle2 className="w-4 h-4" />
              {toolMessage}
            </div>
          )}
          {toolError && (
            <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)]">
              <AlertTriangle className="w-4 h-4" />
              {toolError}
            </div>
          )}
          {checkResult && <CheckResults result={checkResult} />}
          {mismatches && mismatches.length > 0 && (
            <div className="space-y-2 rounded border border-[var(--color-warning-500)]/40 bg-[var(--color-warning-500)]/10 p-3">
              <div className="flex items-center gap-2 text-sm text-[var(--color-warning-500)]">
                <FileWarning className="w-4 h-4" />
                <span className="font-medium">
                  {mismatches.length} pair(s) have caption mismatches between chosen and rejected.
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="primary" onClick={handleCorrectAllToChosen} disabled={loading}>
                  Correct All to Chosen Pair Caption
                </Button>
                <Button size="sm" variant="secondary" onClick={() => setReviewOpen(true)} disabled={loading}>
                  Manually Review Pairs
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setMismatches(null)} disabled={loading}>
                  Dismiss
                </Button>
              </div>
            </div>
          )}
        </div>
      </details>
      {reviewOpen && mismatches && mismatches.length > 0 && (
        <CaptionMismatchModal
          open={reviewOpen}
          mismatches={mismatches}
          onClose={() => void handleManualReviewClose()}
        />
      )}
    </div>
  );
}
