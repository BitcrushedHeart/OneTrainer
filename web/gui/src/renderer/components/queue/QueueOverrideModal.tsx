import { AlertTriangle, RotateCcw, Sparkles, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/shared";
import { FieldBindingProvider } from "@/hooks/fieldBinding";
import { useQueueOverrideStore } from "@/store/queueOverrideStore";
import { useQueueStore } from "@/store/queueStore";
import { useUiSchemaStore } from "@/store/uiSchemaStore";
import type { TabDef } from "@/types/uiSchema";

import { ModalBase } from "../modals/ModalBase";
import { QueueAutoBatchPanel } from "./QueueAutoBatchPanel";
import { QueueOverrideConceptsTab } from "./QueueOverrideConceptsTab";
import { QueueOverrideTabRenderer } from "./QueueOverrideTabRenderer";

const CONCEPTS_TAB_ID = "concepts";

export interface QueueOverrideModalProps {
  open: boolean;
  onClose: () => void;
  entryId: string | null;
}

interface TabEntry {
  id: string;
  label: string;
  /** The schema tab ref; null for the synthetic Concepts tab. */
  schema: TabDef | null;
}

function countOverrides(node: unknown): number {
  if (node === undefined || node === null) return 0;
  if (Array.isArray(node)) return node.length > 0 ? 1 : 0;
  if (typeof node !== "object") return 1;
  let total = 0;
  for (const value of Object.values(node as Record<string, unknown>)) {
    total += countOverrides(value);
  }
  return total;
}

export function QueueOverrideModal({ open, onClose, entryId }: QueueOverrideModalProps) {
  const schema = useUiSchemaStore((s) => s.schema);
  const entry = useQueueStore((s) => s.entries.find((e) => e.id === entryId) ?? null);
  const updateEntry = useQueueStore((s) => s.updateEntry);
  const validate = useQueueStore((s) => s.validate);
  const validation = useQueueStore((s) => (entryId ? s.validationResults[entryId] : undefined));

  const openSession = useQueueOverrideStore((s) => s.open);
  const closeSession = useQueueOverrideStore((s) => s.close);
  const flushSession = useQueueOverrideStore((s) => s.flush);
  const resetAll = useQueueOverrideStore((s) => s.resetAll);
  const overrides = useQueueOverrideStore((s) => s.overrides);
  const sessionEntryId = useQueueOverrideStore((s) => s.entryId);

  const tabs: TabEntry[] = useMemo(() => {
    if (!schema) return [{ id: CONCEPTS_TAB_ID, label: "Concepts", schema: null }];
    const list: TabEntry[] = [];
    for (const tab of schema.tabs) {
      if (tab.renderer !== "schema") continue;
      list.push({ id: tab.id, label: tab.label, schema: tab });
    }
    // Slot Concepts right after Data, falling back to the front of the strip.
    const dataIdx = list.findIndex((t) => t.id === "data");
    const conceptsTab: TabEntry = { id: CONCEPTS_TAB_ID, label: "Concepts", schema: null };
    if (dataIdx >= 0) {
      list.splice(dataIdx + 1, 0, conceptsTab);
    } else {
      list.unshift(conceptsTab);
    }
    return list;
  }, [schema]);

  const [activeTabId, setActiveTabId] = useState<string>("");

  useEffect(() => {
    if (!activeTabId && tabs.length > 0) setActiveTabId(tabs[0].id);
  }, [tabs, activeTabId]);

  // Open / close the override session in the store as the modal toggles.
  useEffect(() => {
    if (open && entry && sessionEntryId !== entry.id) {
      openSession(entry.id, entry.overrides);
    }
  }, [open, entry, sessionEntryId, openSession]);

  useEffect(() => {
    if (!open) return;
    void validate();
  }, [open, validate]);

  const handleClose = async () => {
    await flushSession();
    closeSession();
    onClose();
  };

  const handleNameChange = (next: string) => {
    if (entry) void updateEntry(entry.id, { name: next });
  };

  const overrideCount = useMemo(() => countOverrides(overrides), [overrides]);
  const conceptsOverridden = overrides.concepts !== undefined;
  const activeTab = tabs.find((t) => t.id === activeTabId);

  const errorCount = validation?.errors.length ?? 0;
  const warningCount = validation?.warnings.length ?? 0;

  return (
    <ModalBase
      open={open}
      onClose={() => void handleClose()}
      title="Edit Queue Entry"
      size="full"
      closeOnBackdrop={false}
    >
      <FieldBindingProvider mode="override">
        <div className="flex flex-col gap-5 h-full">
          {/* Header — identity on the left, status chips, then actions */}
          <header className="flex flex-wrap items-center gap-x-6 gap-y-3 pb-4 border-b border-[var(--color-border-subtle)]">
            <div className="flex-1 min-w-[280px] flex flex-col gap-1">
              <span className="text-[var(--text-label)] uppercase tracking-[0.12em] text-[var(--color-on-surface-secondary)]">
                Queue entry
              </span>
              <input
                type="text"
                value={entry?.name ?? ""}
                onChange={(e) => handleNameChange(e.target.value)}
                placeholder="Untitled run"
                className="w-full bg-transparent border-0 border-b border-transparent
                  px-0 py-1 -ml-px
                  text-2xl font-semibold tracking-tight font-[var(--font-display)]
                  text-[var(--color-on-surface)]
                  placeholder:text-[var(--color-on-surface-secondary)] placeholder:font-normal
                  outline-none focus:border-[var(--color-cobalt-600-alpha-30)]
                  transition-[border-color] duration-200"
              />
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              <StatusChip
                tone={overrideCount > 0 ? "cobalt" : "muted"}
                icon={<Sparkles className="w-3.5 h-3.5" />}
                label={
                  overrideCount === 0
                    ? "Inheriting all global settings"
                    : `${overrideCount} override${overrideCount === 1 ? "" : "s"}`
                }
              />
              {errorCount > 0 && (
                <StatusChip
                  tone="error"
                  icon={<XCircle className="w-3.5 h-3.5" />}
                  label={`${errorCount} error${errorCount === 1 ? "" : "s"}`}
                />
              )}
              {warningCount > 0 && (
                <StatusChip
                  tone="warning"
                  icon={<AlertTriangle className="w-3.5 h-3.5" />}
                  label={`${warningCount} warning${warningCount === 1 ? "" : "s"}`}
                />
              )}
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={resetAll}
                disabled={overrideCount === 0}
                title={overrideCount === 0 ? "No overrides to reset" : "Discard every override on this entry"}
              >
                <RotateCcw className="w-4 h-4" />
                Reset all
              </Button>
              <Button variant="primary" size="sm" onClick={() => void handleClose()}>
                Done
              </Button>
            </div>
          </header>

          {entry && <QueueAutoBatchPanel entryId={entry.id} />}

          {/* Tab strip — horizontally scrollable, animated underline */}
          <nav
            aria-label="Override sections"
            className="relative flex items-stretch gap-1 overflow-x-auto -mx-1 px-1 pb-px
              border-b border-[var(--color-border-subtle)]
              scrollbar-thin"
          >
            {tabs.map((tab) => {
              const isActive = activeTabId === tab.id;
              const isConceptsTab = tab.id === CONCEPTS_TAB_ID;
              const showOverrideHint = isConceptsTab && conceptsOverridden;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTabId(tab.id)}
                  aria-current={isActive ? "page" : undefined}
                  className={`relative shrink-0 px-4 py-2.5 text-sm font-medium tracking-tight cursor-pointer
                    transition-colors duration-150
                    focus-visible:outline-none focus-visible:bg-[var(--color-cobalt-600-alpha-06)]
                    ${
                      isActive
                        ? "text-[var(--color-on-surface)]"
                        : "text-[var(--color-on-surface-secondary)] hover:text-[var(--color-on-surface)]"
                    }`}
                >
                  <span className="inline-flex items-center gap-1.5">
                    {tab.label}
                    {showOverrideHint && (
                      <span
                        aria-hidden="true"
                        className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--color-cobalt-600)]
                          shadow-[0_0_6px_var(--color-cobalt-600-alpha-30)]"
                      />
                    )}
                  </span>
                  {isActive && (
                    <span
                      aria-hidden="true"
                      className="absolute left-2 right-2 -bottom-px h-[2px] rounded-full
                        bg-gradient-to-r from-[var(--color-cobalt-600)] to-[var(--color-azure-500)]
                        origin-left animate-[tabSlide_220ms_ease-out]"
                    />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Tab body */}
          <div
            key={activeTabId}
            className="flex-1 overflow-y-auto pr-1 -mr-1 animate-[sectionReveal_240ms_ease-out_both]"
          >
            {activeTab?.id === CONCEPTS_TAB_ID && <QueueOverrideConceptsTab />}
            {activeTab?.schema && <QueueOverrideTabRenderer tab={activeTab.schema} />}
          </div>

          {validation && validation.errors.length > 0 && (
            <div
              className="rounded-[var(--radius-sm)] border border-[var(--color-error-500-alpha-40)]
                bg-[var(--color-error-500-alpha-08)] px-3 py-2
                text-[var(--text-caption)] text-[var(--color-error-500)] space-y-1"
            >
              {validation.errors.slice(0, 5).map((err, i) => (
                <div key={i} className="flex items-start gap-2">
                  <XCircle className="w-3.5 h-3.5 shrink-0 mt-[3px]" />
                  <span className="leading-snug">{err}</span>
                </div>
              ))}
              {validation.errors.length > 5 && (
                <div className="text-[var(--color-on-surface-secondary)] pl-[22px]">
                  + {validation.errors.length - 5} more
                </div>
              )}
            </div>
          )}
        </div>
      </FieldBindingProvider>
    </ModalBase>
  );
}

interface StatusChipProps {
  tone: "cobalt" | "muted" | "error" | "warning";
  icon: React.ReactNode;
  label: string;
}

const TONE_CLASSES: Record<StatusChipProps["tone"], string> = {
  cobalt:
    "border-[var(--color-cobalt-600-alpha-25)] bg-[var(--color-cobalt-600-alpha-06)] text-[var(--color-cobalt-600)]",
  muted: "border-[var(--color-border-subtle)] bg-transparent text-[var(--color-on-surface-secondary)]",
  error: "border-[var(--color-error-500-alpha-40)] bg-[var(--color-error-500-alpha-08)] text-[var(--color-error-500)]",
  warning:
    "border-[var(--color-warning-500-alpha-15)] bg-[var(--color-warning-500-alpha-06)] text-[var(--color-warning-500)]",
};

function StatusChip({ tone, icon, label }: StatusChipProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full
        border text-[var(--text-caption)] font-medium leading-none
        ${TONE_CLASSES[tone]}`}
    >
      {icon}
      {label}
    </span>
  );
}
