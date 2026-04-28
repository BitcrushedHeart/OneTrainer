import { AlertTriangle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/shared";
import { useQueueStore } from "@/store/queueStore";

import { ModalBase } from "../modals/ModalBase";

interface QueueAutoBatchAllModalProps {
  open: boolean;
  onClose: () => void;
}

interface FormState {
  min_batch_size: number;
  max_batch_size: number;
  target_pct: number;
  max_drop_pct: number;
}

const DEFAULT_FORM: FormState = {
  min_batch_size: 1,
  max_batch_size: 4,
  target_pct: 100,
  max_drop_pct: 5,
};

function clampInt(value: number, lo: number, hi: number): number {
  if (Number.isNaN(value)) return lo;
  return Math.max(lo, Math.min(hi, Math.round(value)));
}

function clampFloat(value: number, lo: number, hi: number): number {
  if (Number.isNaN(value)) return lo;
  return Math.max(lo, Math.min(hi, value));
}

export function QueueAutoBatchAllModal({ open, onClose }: QueueAutoBatchAllModalProps) {
  const bulkAutoBatch = useQueueStore((s) => s.bulkAutoBatch);
  const entryCount = useQueueStore((s) => s.entries.length);

  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [error, setError] = useState<string | null>(null);
  const [isApplying, setIsApplying] = useState(false);

  useEffect(() => {
    if (open) {
      setForm(DEFAULT_FORM);
      setError(null);
      setIsApplying(false);
    }
  }, [open]);

  const minMaxInvalid = form.min_batch_size > form.max_batch_size;

  const handleApply = async () => {
    if (minMaxInvalid) {
      setError("Min must be <= Max");
      return;
    }
    setIsApplying(true);
    setError(null);
    try {
      await bulkAutoBatch(form);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsApplying(false);
    }
  };

  return (
    <ModalBase open={open} onClose={onClose} title="Auto-Batch All Entries" size="md">
      <div className="flex flex-col gap-4">
        <p className="text-sm text-[var(--color-on-surface-secondary)] leading-snug">
          Apply Auto-Batch to all {entryCount} {entryCount === 1 ? "entry" : "entries"} in the queue. The batch size and
          accumulation steps will be calculated when you start the queue (or when you click Calculate on each entry
          individually).
        </p>

        <Field
          label="Min batch size"
          value={form.min_batch_size}
          min={1}
          max={4096}
          onChange={(v) => setForm((s) => ({ ...s, min_batch_size: clampInt(v, 1, 4096) }))}
        />
        <Field
          label="Max batch size"
          value={form.max_batch_size}
          min={1}
          max={4096}
          onChange={(v) => setForm((s) => ({ ...s, max_batch_size: clampInt(v, 1, 4096) }))}
        />
        <Field
          label="Target Effective Batch Size %"
          value={form.target_pct}
          min={1}
          max={100}
          step={1}
          onChange={(v) => setForm((s) => ({ ...s, target_pct: clampFloat(v, 1, 100) }))}
        />
        <Field
          label="Max drop %"
          value={form.max_drop_pct}
          min={0}
          max={100}
          step={1}
          onChange={(v) => setForm((s) => ({ ...s, max_drop_pct: clampFloat(v, 0, 100) }))}
        />

        {minMaxInvalid && (
          <div className="flex items-center gap-1.5 text-[var(--text-caption)] text-[var(--color-error-500)]">
            <AlertTriangle className="w-3.5 h-3.5" />
            Min must be ≤ Max
          </div>
        )}

        {error && (
          <div className="flex items-center gap-1.5 text-[var(--text-caption)] text-[var(--color-error-500)]">
            <AlertTriangle className="w-3.5 h-3.5" />
            {error}
          </div>
        )}

        <div className="flex justify-end gap-2 pt-2 border-t border-[var(--color-border-subtle)]">
          <Button variant="ghost" onClick={onClose} disabled={isApplying}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={() => void handleApply()}
            disabled={minMaxInvalid || isApplying || entryCount === 0}
          >
            <Sparkles className="w-4 h-4 mr-1" />
            {isApplying ? "Applying..." : "Apply to all"}
          </Button>
        </div>
      </div>
    </ModalBase>
  );
}

interface FieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}

function Field({ label, value, min, max, step = 1, onChange }: FieldProps) {
  const [text, setText] = useState<string>(String(value));

  useEffect(() => {
    setText(String(value));
  }, [value]);

  return (
    <label className="flex items-center justify-between gap-3">
      <span className="text-sm text-[var(--color-on-surface)]">{label}</span>
      <input
        type="number"
        value={text}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const raw = e.target.value;
          setText(raw);
          if (raw === "" || raw === "-") return;
          const num = Number(raw);
          if (Number.isNaN(num)) return;
          onChange(num);
        }}
        className="w-24 px-2 py-1 rounded-[var(--radius-xs)] border border-[var(--color-border-subtle)]
          bg-[var(--color-surface)] text-[var(--color-on-surface)] text-sm
          focus:outline-none focus:border-[var(--color-cobalt-600-alpha-30)]"
      />
    </label>
  );
}
