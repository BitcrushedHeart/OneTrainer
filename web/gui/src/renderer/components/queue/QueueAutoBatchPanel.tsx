import { AlertTriangle, Loader2, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { AutoBatchSettingsData } from "@/api/queueApi";
import { Button, Toggle } from "@/components/shared";
import { useQueueOverrideStore } from "@/store/queueOverrideStore";
import { useQueueStore } from "@/store/queueStore";

const SAVE_DEBOUNCE_MS = 500;

interface QueueAutoBatchPanelProps {
  entryId: string;
}

type Draft = Pick<AutoBatchSettingsData, "min_batch_size" | "max_batch_size" | "target_pct" | "max_drop_pct">;

function clamp(value: number, lo: number, hi: number): number {
  if (Number.isNaN(value)) return lo;
  return Math.max(lo, Math.min(hi, value));
}

export function QueueAutoBatchPanel({ entryId }: QueueAutoBatchPanelProps) {
  const entry = useQueueStore((s) => s.entries.find((e) => e.id === entryId) ?? null);
  const updateAutoBatch = useQueueStore((s) => s.updateAutoBatch);
  const calculateAutoBatch = useQueueStore((s) => s.calculateAutoBatch);
  const validate = useQueueStore((s) => s.validate);
  const overrideSetField = useQueueOverrideStore((s) => s.setField);

  const settings = entry?.auto_batch;

  const [draft, setDraft] = useState<Draft>(() => ({
    min_batch_size: settings?.min_batch_size ?? 1,
    max_batch_size: settings?.max_batch_size ?? 8,
    target_pct: settings?.target_pct ?? 10,
    max_drop_pct: settings?.max_drop_pct ?? 5,
  }));
  const [isCalculating, setIsCalculating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Reset draft when the entry switches or its persisted settings change.
  useEffect(() => {
    if (!settings) return;
    setDraft({
      min_batch_size: settings.min_batch_size,
      max_batch_size: settings.max_batch_size,
      target_pct: settings.target_pct,
      max_drop_pct: settings.max_drop_pct,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entryId, settings?.min_batch_size, settings?.max_batch_size, settings?.target_pct, settings?.max_drop_pct]);

  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const scheduleSave = useCallback(
    (next: Draft) => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(() => {
        void (async () => {
          try {
            setErrorMessage(null);
            await updateAutoBatch(entryId, next);
          } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : String(err));
          }
        })();
      }, SAVE_DEBOUNCE_MS);
    },
    [entryId, updateAutoBatch],
  );

  useEffect(
    () => () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    },
    [],
  );

  const updateDraft = (patch: Partial<Draft>) => {
    setDraft((prev) => {
      const next = { ...prev, ...patch };
      scheduleSave(next);
      return next;
    });
  };

  const minMaxInvalid = draft.min_batch_size > draft.max_batch_size;
  const enabled = !!settings?.enabled;

  const handleToggle = async (value: boolean) => {
    try {
      setErrorMessage(null);
      await updateAutoBatch(entryId, { enabled: value });
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    }
  };

  const handleCalculate = async () => {
    if (minMaxInvalid) return;
    setIsCalculating(true);
    setErrorMessage(null);
    try {
      const result = await calculateAutoBatch(entryId);
      if (result) {
        overrideSetField("batch_size", result.batch_size);
        overrideSetField("gradient_accumulation_steps", result.accum);
        void validate();
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setIsCalculating(false);
    }
  };

  const chip = useMemo(() => {
    if (!settings) return null;
    if (settings.last_batch_size == null || settings.last_accum == null) {
      return enabled ? "Not yet calculated" : "Disabled";
    }
    const eff = settings.last_effective_samples;
    const dropped = settings.last_dropped;
    const dropFrag = eff != null && dropped != null ? ` · drops=${dropped}/${eff}` : "";
    return `bs=${settings.last_batch_size} · accum=${settings.last_accum}${dropFrag}`;
  }, [settings, enabled]);

  if (!entry || !settings) return null;

  return (
    <div
      className="flex items-center gap-3 flex-wrap rounded-[var(--radius-sm)] border border-[var(--color-border-subtle)]
        bg-[var(--color-surface-raised)] px-3 py-2"
    >
      <div className="flex items-center gap-2 shrink-0">
        <Toggle value={enabled} onChange={(v) => void handleToggle(v)} />
        <span className="text-sm font-medium text-[var(--color-on-surface)] inline-flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-[var(--color-cobalt-600)]" />
          Auto-Batch
        </span>
      </div>

      <NumberField
        label="Min"
        value={draft.min_batch_size}
        min={1}
        max={4096}
        disabled={!enabled}
        onChange={(v) => updateDraft({ min_batch_size: clamp(Math.round(v), 1, 4096) })}
      />
      <NumberField
        label="Max"
        value={draft.max_batch_size}
        min={1}
        max={4096}
        disabled={!enabled}
        onChange={(v) => updateDraft({ max_batch_size: clamp(Math.round(v), 1, 4096) })}
      />
      <NumberField
        label="Target %"
        value={draft.target_pct}
        min={1}
        max={100}
        step={1}
        disabled={!enabled}
        onChange={(v) => updateDraft({ target_pct: clamp(v, 1, 100) })}
      />
      <NumberField
        label="Max drop %"
        value={draft.max_drop_pct}
        min={0}
        max={100}
        step={1}
        disabled={!enabled}
        onChange={(v) => updateDraft({ max_drop_pct: clamp(v, 0, 100) })}
      />

      <Button
        variant="secondary"
        size="sm"
        onClick={() => void handleCalculate()}
        disabled={!enabled || minMaxInvalid || isCalculating}
        title={
          minMaxInvalid
            ? "Min must be <= Max"
            : enabled
              ? "Calculate batch size and accumulation steps"
              : "Enable Auto-Batch first"
        }
      >
        {isCalculating ? (
          <>
            <Loader2 className="w-4 h-4 mr-1 animate-spin" />
            Calculating
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4 mr-1" />
            Calculate
          </>
        )}
      </Button>

      {chip && (
        <span
          className="inline-flex items-center px-2.5 py-1 rounded-full
            border border-[var(--color-cobalt-600-alpha-25)] bg-[var(--color-cobalt-600-alpha-06)]
            text-[var(--text-caption)] font-medium text-[var(--color-cobalt-600)]"
        >
          {chip}
        </span>
      )}

      {minMaxInvalid && (
        <span className="inline-flex items-center gap-1 text-[var(--text-caption)] text-[var(--color-error-500)]">
          <AlertTriangle className="w-3.5 h-3.5" />
          Min must be ≤ Max
        </span>
      )}

      {errorMessage && (
        <span className="inline-flex items-center gap-1 text-[var(--text-caption)] text-[var(--color-error-500)]">
          <AlertTriangle className="w-3.5 h-3.5" />
          {errorMessage}
        </span>
      )}
    </div>
  );
}

interface NumberFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}

function NumberField({ label, value, min, max, step = 1, disabled, onChange }: NumberFieldProps) {
  const [text, setText] = useState<string>(String(value));

  useEffect(() => {
    setText(String(value));
  }, [value]);

  return (
    <label
      className={`inline-flex items-center gap-1.5 text-[var(--text-caption)]
        ${disabled ? "opacity-50" : ""}`}
    >
      <span className="text-[var(--color-on-surface-secondary)]">{label}</span>
      <input
        type="number"
        value={text}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const raw = e.target.value;
          setText(raw);
          if (raw === "" || raw === "-") return;
          const num = Number(raw);
          if (Number.isNaN(num)) return;
          onChange(num);
        }}
        className="w-16 px-2 py-1 rounded-[var(--radius-xs)] border border-[var(--color-border-subtle)]
          bg-[var(--color-surface)] text-[var(--color-on-surface)] text-sm
          focus:outline-none focus:border-[var(--color-cobalt-600-alpha-30)]"
      />
    </label>
  );
}
