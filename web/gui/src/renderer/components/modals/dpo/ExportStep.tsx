import { CheckCircle2 } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/shared";

import type { ExportStepProps } from "./types";

export function ExportStep({ totalPairs, onFinalize, onClose, finalResult }: ExportStepProps) {
  const [valPct, setValPct] = useState(10);
  const [submitting, setSubmitting] = useState(false);

  if (finalResult) {
    return (
      <div className="text-center py-6 space-y-4">
        <CheckCircle2 className="w-12 h-12 mx-auto text-[var(--color-success-500)]" />
        <h2 className="text-2xl font-bold text-[var(--color-on-surface)]">Finalized!</h2>
        <p className="text-sm text-[var(--color-on-surface-secondary)]">
          {finalResult.total_pairs} pairs ({finalResult.train_count} train, {finalResult.val_count} val)
        </p>
        <div className="flex justify-center">
          <Button variant="primary" onClick={onClose}>
            Done
          </Button>
        </div>
      </div>
    );
  }

  const handleFinalize = async () => {
    setSubmitting(true);
    try {
      await onFinalize(Math.max(0, Math.min(100, valPct)));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-5 py-4">
      <div className="text-center space-y-1">
        <CheckCircle2 className="w-10 h-10 mx-auto text-[var(--color-cobalt-600)]" />
        <h2 className="text-2xl font-bold text-[var(--color-on-surface)]">Scoring Complete</h2>
      </div>

      <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] p-4 text-center space-y-1">
        <p className="text-sm text-[var(--color-on-surface)]">{totalPairs} chosen/rejected pairs exported.</p>
      </div>

      <div className="flex items-center justify-center gap-3">
        <label htmlFor="dpo-val-pct" className="text-sm text-[var(--color-on-surface)]">
          Validation %:
        </label>
        <input
          id="dpo-val-pct"
          type="number"
          min={0}
          max={100}
          value={valPct}
          onChange={(e) => setValPct(Number(e.target.value))}
          className="w-20 px-2 py-1 rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] text-[var(--color-on-surface)] text-sm"
        />
        <span className="text-xs text-[var(--color-on-surface-secondary)]">(0 = no validation split)</span>
      </div>

      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-2">
        <Button variant="primary" onClick={handleFinalize} loading={submitting}>
          Finalize (Train/Val Split + Concepts)
        </Button>
        <Button variant="secondary" onClick={onClose}>
          Close (Pairs Already Saved)
        </Button>
      </div>
    </div>
  );
}
