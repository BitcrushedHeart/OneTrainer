import { AlertTriangle, ArrowRight, Search, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { validationApi, type ValidationMatch, type ValidationResults } from "@/api/validationApi";
import { Button, ProgressBar } from "@/components/shared";
import { ModalBase } from "./ModalBase";

interface Props {
  open: boolean;
  onClose: () => void;
}

function MatchRow({
  match,
  onRemove,
  onMove,
}: {
  match: ValidationMatch;
  onRemove: (path: string) => void;
  onMove: (path: string) => void;
}) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 border-b border-[var(--color-border-subtle)] text-sm">
      <div className="flex-1 truncate text-[var(--color-on-surface)]" title={match.train_image}>
        {match.train_image}
      </div>
      <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--color-border-subtle)] text-[var(--color-on-surface-secondary)]">
        {match.kind}
      </span>
      <div className="flex-1 truncate text-[var(--color-on-surface)]" title={match.val_image}>
        {match.val_image}
      </div>
      {match.score != null && (
        <span className="text-xs text-[var(--color-on-surface-secondary)] w-12 text-right">
          {(match.score * 100).toFixed(1)}%
        </span>
      )}
      <div className="flex gap-1 shrink-0">
        <button
          className="p-1 rounded hover:bg-[var(--color-border-subtle)] text-[var(--color-on-surface-secondary)]"
          onClick={() => onMove(match.val_path)}
          title="Move to training set"
        >
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
        <button
          className="p-1 rounded hover:bg-[rgba(239,68,68,0.15)] text-[var(--color-error-500)]"
          onClick={() => onRemove(match.val_path)}
          title="Remove validation image"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

export function ValidationCheckerModal({ open, onClose }: Props) {
  const [results, setResults] = useState<ValidationResults>({});
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState({ label: "", current: 0, total: 0 });
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const pollStatus = useCallback(async () => {
    try {
      const status = await validationApi.getStatus();
      setProgress(status.progress);
      if (status.status === "done" || status.status === "idle") {
        setScanning(false);
        if (pollRef.current) clearInterval(pollRef.current);
        const res = await validationApi.getResults();
        setResults(res);
      }
    } catch {
      // ignore polling errors
    }
  }, []);

  const startBasicScan = async () => {
    setScanning(true);
    setResults({});
    await validationApi.scanBasic();
    pollRef.current = setInterval(pollStatus, 1000);
  };

  const startDeepScan = async () => {
    setScanning(true);
    await validationApi.scanDeep();
    pollRef.current = setInterval(pollStatus, 1000);
  };

  const handleCancel = async () => {
    await validationApi.cancel();
  };

  const handleRemove = async (path: string) => {
    await validationApi.removeMatch(path);
    setResults((prev) => ({
      ...prev,
      matches: prev.matches?.filter((m) => m.val_path !== path),
    }));
  };

  const handleMove = async (path: string) => {
    await validationApi.moveMatch(path);
    setResults((prev) => ({
      ...prev,
      matches: prev.matches?.filter((m) => m.val_path !== path),
    }));
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const matches = results.matches ?? [];
  const summary = results.summary;

  return (
    <ModalBase open={open} onClose={onClose} title="Validation Checker" size="xl">
      <div className="space-y-4">
        {/* Actions */}
        <div className="flex items-center gap-2 flex-wrap">
          <Button variant="primary" onClick={startBasicScan} disabled={scanning}>
            <Search className="w-4 h-4 mr-1" />
            Basic Scan
          </Button>
          <Button
            variant="secondary"
            onClick={startDeepScan}
            disabled={scanning || !results.summary}
          >
            <Search className="w-4 h-4 mr-1" />
            Deep Scan (CLIP)
          </Button>
          {scanning && (
            <Button variant="danger" size="sm" onClick={handleCancel}>
              <X className="w-4 h-4 mr-1" />
              Cancel
            </Button>
          )}
        </div>

        {/* Progress */}
        {scanning && (
          <div className="space-y-1">
            <span className="text-xs text-[var(--color-on-surface-secondary)]">{progress.label}</span>
            <ProgressBar value={progress.total > 0 ? progress.current / progress.total : 0} />
          </div>
        )}

        {/* Summary */}
        {summary && (
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div className="p-2 rounded bg-[var(--color-surface-container)]">
              <span className="text-[var(--color-on-surface-secondary)]">Training: </span>
              <span className="font-medium text-[var(--color-on-surface)]">{summary.train_images}</span>
            </div>
            <div className="p-2 rounded bg-[var(--color-surface-container)]">
              <span className="text-[var(--color-on-surface-secondary)]">Validation: </span>
              <span className="font-medium text-[var(--color-on-surface)]">{summary.val_images}</span>
            </div>
            <div className="p-2 rounded bg-[var(--color-surface-container)]">
              <span className="text-[var(--color-on-surface-secondary)]">Matches: </span>
              <span className="font-medium text-[var(--color-error-500)]">{matches.length}</span>
            </div>
          </div>
        )}

        {/* Matches list */}
        {matches.length > 0 && (
          <div className="border border-[var(--color-border-subtle)] rounded-md overflow-hidden">
            <div className="flex items-center gap-3 px-3 py-1.5 bg-[var(--color-surface-container)] text-xs font-medium text-[var(--color-on-surface-secondary)]">
              <div className="flex-1">Training Image</div>
              <div className="w-16 text-center">Type</div>
              <div className="flex-1">Validation Image</div>
              <div className="w-12 text-right">Score</div>
              <div className="w-16">Actions</div>
            </div>
            <div className="max-h-80 overflow-y-auto">
              {matches.map((m) => (
                <MatchRow key={m.id} match={m} onRemove={handleRemove} onMove={handleMove} />
              ))}
            </div>
          </div>
        )}

        {/* Caption matches */}
        {results.captions && results.captions.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-[var(--color-on-surface)]">
              <AlertTriangle className="w-4 h-4 inline mr-1 text-[var(--color-warning-500)]" />
              Shared Captions ({results.captions.length})
            </h4>
            <div className="max-h-40 overflow-y-auto space-y-1">
              {results.captions.map((c, i) => (
                <div key={i} className="text-xs p-2 rounded bg-[var(--color-surface-container)]">
                  <span className="font-mono text-[var(--color-on-surface)]">&quot;{c.caption.slice(0, 80)}...&quot;</span>
                  <span className="text-[var(--color-on-surface-secondary)]">
                    {" "}
                    ({c.train_images.length} train, {c.val_images.length} val)
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!scanning && !summary && (
          <div className="text-center py-8 text-[var(--color-on-surface-secondary)]">
            <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Run a Basic Scan to check for validation data leaks.</p>
            <p className="text-xs mt-1">This compares your training and validation concepts for duplicates, near-duplicates, and shared captions.</p>
          </div>
        )}
      </div>
    </ModalBase>
  );
}
