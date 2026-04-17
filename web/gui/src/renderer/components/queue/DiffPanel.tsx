import { useEffect, useState } from "react";

import { queueApi, type QueueDiffResponse } from "@/api/queueApi";

export interface DiffPanelProps {
  entryId: string | null;
}

function formatValue(v: unknown): string {
  if (v == null) return "(unset)";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

export function DiffPanel({ entryId }: DiffPanelProps) {
  const [diff, setDiff] = useState<QueueDiffResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!entryId) {
      setDiff(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    queueApi
      .entryDiff(entryId)
      .then((res) => {
        if (!cancelled) setDiff(res);
      })
      .catch(() => {
        if (!cancelled) setDiff({ ok: false, error: "Failed to load diff" });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [entryId]);

  if (!entryId) return null;
  if (loading) {
    return <div className="text-xs text-[var(--color-on-surface-secondary)] p-3">Loading diff...</div>;
  }
  if (!diff?.ok || !diff.sections) {
    return (
      <div className="text-xs text-[var(--color-on-surface-secondary)] p-3">{diff?.error ?? "No diff available"}</div>
    );
  }

  const sectionEntries = Object.entries(diff.sections);
  if (sectionEntries.length === 0) {
    return (
      <div className="text-xs text-[var(--color-on-surface-secondary)] p-3">
        No overrides — this entry uses the current config defaults.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold text-[var(--color-on-surface)]">Overrides vs defaults</h4>
      {sectionEntries.map(([section, fields]) => (
        <details
          key={section}
          open
          className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)]"
        >
          <summary className="cursor-pointer px-3 py-1.5 text-sm font-medium text-[var(--color-on-surface)]">
            {section} <span className="text-xs text-[var(--color-on-surface-secondary)]">({fields.length})</span>
          </summary>
          <div className="px-3 py-2 grid grid-cols-[1fr_1fr_1fr] gap-x-3 gap-y-1 text-xs">
            <span className="font-medium text-[var(--color-on-surface-secondary)]">Field</span>
            <span className="font-medium text-[var(--color-on-surface-secondary)]">Default</span>
            <span className="font-medium text-[var(--color-on-surface-secondary)]">This entry</span>
            {fields.map((f) => (
              <FieldRow key={f.field} field={f} />
            ))}
          </div>
        </details>
      ))}
    </div>
  );
}

function FieldRow({ field }: { field: { field: string; label: string; current: unknown; default: unknown } }) {
  return (
    <>
      <span className="text-[var(--color-on-surface)] truncate" title={field.field}>
        {field.label}
      </span>
      <span className="text-[var(--color-on-surface-secondary)] truncate font-mono" title={formatValue(field.default)}>
        {formatValue(field.default)}
      </span>
      <span className="text-[var(--color-cobalt-600)] truncate font-mono" title={formatValue(field.current)}>
        {formatValue(field.current)}
      </span>
    </>
  );
}
