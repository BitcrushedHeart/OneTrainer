import { AlertCircle, CheckCircle2, Loader2, Plug, Plus, Trash2, Wand2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

import type { RegenPreviewItem } from "@/api/dpoApi";
import { dpoApi } from "@/api/dpoApi";
import { swarmApi } from "@/api/swarmApi";
import { Button, DirPicker, FormEntry, Select, Toggle } from "@/components/shared";
import { INPUT_FULL, PLACEHOLDER, TEXTAREA_FULL } from "@/utils/inputStyles";

import {
  customRow,
  type FixRow,
  loadRegenSettings,
  type RegenSettings,
  type RegenStartRequest,
  rowsToFixes,
  saveRegenSettings,
  segmentParamsFromForm,
} from "./regenFixes";

export interface RegenFixDialogProps {
  /** The images to regenerate — the Shift+click selection plus the one right-clicked. */
  paths: string[];
  onClose: () => void;
  /** Resolves to an error message, or null when the run started. */
  onStart: (req: RegenStartRequest) => Promise<string | null>;
}

function basename(path: string): string {
  return path.split(/[/\\]/).pop() ?? path;
}

/**
 * SwarmUI's "Segment Refining" parameter group, by API id. Blank means "not
 * sent", so SwarmUI's own default applies; the hints quote those defaults.
 */
const SEGMENT_FIELDS: Array<{
  key: string;
  label: string;
  kind: "number" | "text" | "select" | "bool";
  hint: string;
  options?: string[];
}> = [
  { key: "segmentmaskblur", label: "Mask blur", kind: "number", hint: "0–64. SwarmUI default 10." },
  { key: "segmentmaskgrow", label: "Mask grow (px)", kind: "number", hint: "0–512. SwarmUI default 16." },
  {
    key: "segmentmaskoversize",
    label: "Mask oversize (px)",
    kind: "number",
    hint: "0–512, default 16. Larger includes more context for a more accurate inpaint; smaller gets closer for finer detail.",
  },
  {
    key: "segmentthresholdmax",
    label: "Threshold max",
    kind: "number",
    hint: "0–1, default 1. Lower forces more of the mask to full strength.",
  },
  { key: "segmentsteps", label: "Segment steps", kind: "number", hint: "Blank = the image's own step count." },
  { key: "segmentcfgscale", label: "Segment CFG", kind: "number", hint: "Blank = the image's own CFG." },
  {
    key: "segmenttargetresolution",
    label: "Target resolution",
    kind: "text",
    hint: "e.g. 1024x1024. Controls the size the region is regenerated at.",
  },
  {
    key: "segmentmodel",
    label: "Segment model",
    kind: "text",
    hint: "A different checkpoint for the refine pass. Blank = the same model.",
  },
  {
    key: "segmentsortorder",
    label: "Sort order",
    kind: "text",
    hint: "How yolo-…-N indices are ordered. Default left-right.",
  },
  {
    key: "segmentapplyafter",
    label: "Apply after",
    kind: "select",
    hint: "Refiner (default) runs the fix after the refiner step; Base runs it before.",
    options: ["", "Refiner", "Base"],
  },
  { key: "savesegmentmask", label: "Save segment mask", kind: "bool", hint: "Also write the mask SwarmUI used." },
];

export function RegenFixDialog({ paths, onClose, onStart }: RegenFixDialogProps) {
  const [settings, setSettings] = useState<RegenSettings>(loadRegenSettings);
  const [advanced, setAdvanced] = useState(false);
  const [preview, setPreview] = useState<RegenPreviewItem[]>([]);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [connection, setConnection] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const update = useCallback((patch: Partial<RegenSettings>) => {
    setSettings((previous) => {
      const next = { ...previous, ...patch };
      saveRegenSettings(next);
      return next;
    });
  }, []);

  const updateRow = useCallback(
    (id: string, patch: Partial<FixRow>) => {
      setSettings((previous) => {
        const next = { ...previous, rows: previous.rows.map((row) => (row.id === id ? { ...row, ...patch } : row)) };
        saveRegenSettings(next);
        return next;
      });
    },
    [setSettings],
  );

  const removeRow = useCallback((id: string) => {
    setSettings((previous) => {
      const next = { ...previous, rows: previous.rows.filter((row) => row.id !== id) };
      saveRegenSettings(next);
      return next;
    });
  }, []);

  // Esc closes only this dialog. Capture phase + stopPropagation keeps the
  // DPO modal's own Escape handler from closing the whole session.
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      onClose();
    };
    window.addEventListener("keydown", handleKey, true);
    return () => window.removeEventListener("keydown", handleKey, true);
  }, [onClose]);

  const fixes = useMemo(() => rowsToFixes(settings.rows), [settings.rows]);
  const segmentParams = useMemo(() => segmentParamsFromForm(settings.segment), [settings.segment]);
  const subPromptUsed = fixes.some((fix) => fix.sub_prompt.length > 0);

  // Live preview of what each image would be sent as, debounced so typing a
  // target does not fire a request per keystroke.
  useEffect(() => {
    if (fixes.length === 0) {
      setPreview([]);
      setPreviewError(null);
      return;
    }
    let cancelled = false;
    setPreviewBusy(true);
    const timer = setTimeout(() => {
      void dpoApi
        .regenPreview(paths, fixes, segmentParams, settings.modelOverride)
        .then((res) => {
          if (cancelled) return;
          if (res.ok) {
            setPreview(res.items ?? []);
            setPreviewError(null);
          } else {
            setPreview([]);
            setPreviewError(res.error ?? "Preview failed");
          }
        })
        .catch((e: unknown) => {
          if (!cancelled) setPreviewError(String(e));
        })
        .finally(() => {
          if (!cancelled) setPreviewBusy(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [paths, fixes, segmentParams, settings.modelOverride]);

  const check = useCallback(async () => {
    setChecking(true);
    setConnection(null);
    try {
      const res = await swarmApi.check(settings.baseUrl, settings.token);
      if (res.ok) {
        setConnection(`SwarmUI ${res.version ?? ""} — ${(res.loaded ?? []).join(", ") || "no model loaded"}`);
        const all = [...(res.loaded ?? []), ...(res.models ?? [])];
        setModels(Array.from(new Set(all)));
      } else {
        setConnection(res.error ?? "SwarmUI did not answer");
      }
    } catch (e) {
      setConnection(String(e));
    } finally {
      setChecking(false);
    }
  }, [settings.baseUrl, settings.token]);

  const readyCount = preview.filter((item) => item.ok).length;
  const canStart = fixes.length > 0 && readyCount > 0 && !starting && !previewBusy;

  const start = useCallback(async () => {
    setStarting(true);
    setError(null);
    const message = await onStart({
      paths,
      fixes,
      segmentParams,
      modelOverride: settings.modelOverride,
      baseUrl: settings.baseUrl,
      token: settings.token,
      generatedFolder: settings.generated,
      concurrency: settings.concurrency,
    });
    setStarting(false);
    if (message) {
      setError(message);
      return;
    }
    onClose();
  }, [onStart, onClose, paths, fixes, segmentParams, settings]);

  const modelOptions = useMemo(() => {
    const names = new Set(models);
    if (settings.modelOverride) names.add(settings.modelOverride);
    return [
      { label: "Image's own model", value: "" },
      ...Array.from(names).map((name) => ({ label: name, value: name })),
    ];
  }, [models, settings.modelOverride]);

  return createPortal(
    <div
      className="fixed inset-0 bg-black/70 z-[56] flex items-center justify-center p-6"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="bg-[var(--color-surface-container)] border border-[var(--color-border-subtle)] rounded-lg p-5 max-w-5xl w-full max-h-[92vh] overflow-y-auto shadow-xl flex flex-col gap-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-bold text-[var(--color-on-surface)] flex items-center gap-2">
              <Wand2 className="w-5 h-5" />
              Fix &amp; Regenerate
            </h3>
            <div className="text-xs text-[var(--color-on-surface-secondary)]">
              Resends each image to SwarmUI with its own seed, model and settings, plus a{" "}
              <code className="font-mono">&lt;segment:…&gt;</code> refine pass per fix. Results are added to this group;
              nothing is replaced.
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} title="Close">
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Fix rows */}
        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-[auto_1fr_1.4fr_6rem_6rem_1.4fr_auto] gap-2 items-center text-[11px] text-[var(--color-on-surface-secondary)] px-1">
            <span />
            <span>Fix</span>
            <span title="CLIPSeg text, a|b for one merged mask, or yolo-<model>[-index][:classes:]">Target</span>
            <span title="Inpaint strength for the region. SwarmUI's own default is 0.6.">Creativity</span>
            <span title="Mask cutoff. SwarmUI's own default is 0.5 (0.25 for yolo targets).">Threshold</span>
            <span title="Prompt for the region instead of the global prompt. Leave blank to keep the image's grouping intact.">
              Sub-prompt (optional)
            </span>
            <span />
          </div>
          {settings.rows.map((row) => (
            <div
              key={row.id}
              className="grid grid-cols-[auto_1fr_1.4fr_6rem_6rem_1.4fr_auto] gap-2 items-center px-1"
              style={{ opacity: row.enabled ? 1 : 0.6 }}
            >
              <Toggle value={row.enabled} onChange={(v) => updateRow(row.id, { enabled: v })} />
              {row.custom ? (
                <input
                  className={`${INPUT_FULL} ${PLACEHOLDER}`}
                  value={row.label}
                  placeholder="Fix …"
                  onChange={(e) => updateRow(row.id, { label: e.target.value })}
                />
              ) : (
                <span className="text-sm text-[var(--color-on-surface)]">{row.label}</span>
              )}
              <input
                className={`${INPUT_FULL} ${PLACEHOLDER} font-mono`}
                value={row.target}
                placeholder="face, text, hands|feet, yolo-face_yolov8m-seg_60.pt"
                onChange={(e) => updateRow(row.id, { target: e.target.value })}
              />
              <input
                className={INPUT_FULL}
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={row.creativity}
                onChange={(e) => updateRow(row.id, { creativity: Number(e.target.value) })}
              />
              <input
                className={INPUT_FULL}
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={row.threshold}
                onChange={(e) => updateRow(row.id, { threshold: Number(e.target.value) })}
              />
              <input
                className={`${INPUT_FULL} ${PLACEHOLDER}`}
                value={row.subPrompt}
                placeholder="blank = global prompt"
                onChange={(e) => updateRow(row.id, { subPrompt: e.target.value })}
              />
              {row.custom ? (
                <Button variant="ghost" size="sm" onClick={() => removeRow(row.id)} title="Remove this fix">
                  <Trash2 className="w-4 h-4" />
                </Button>
              ) : (
                <span className="w-8" />
              )}
            </div>
          ))}
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => update({ rows: [...settings.rows, customRow()] })}
              title="Add a fix with your own target"
            >
              <Plus className="w-4 h-4" />
              Fix …
            </Button>
            {subPromptUsed ? (
              <span className="text-[11px]" style={{ color: "#f97316" }}>
                A sub-prompt is free text: SwarmUI uses it for the region, and the regenerated image will land in a
                different group on a future scan.
              </span>
            ) : null}
          </div>
        </div>

        {/* Advanced */}
        <div className="border border-[var(--color-border-subtle)] rounded">
          <button
            type="button"
            className="w-full text-left px-3 py-2 text-sm text-[var(--color-on-surface)] flex items-center justify-between"
            onClick={() => setAdvanced((v) => !v)}
          >
            <span>Advanced — segment refining, model override, connection</span>
            <span className="text-xs text-[var(--color-on-surface-secondary)]">{advanced ? "hide" : "show"}</span>
          </button>
          {advanced ? (
            <div className="px-3 pb-3 flex flex-col gap-3">
              <div className="grid grid-cols-3 gap-3">
                {SEGMENT_FIELDS.map((field) => {
                  const raw = settings.segment[field.key];
                  if (field.kind === "bool") {
                    return (
                      <div key={field.key} className="flex items-end pb-1">
                        <Toggle
                          label={field.label}
                          tooltip={field.hint}
                          value={raw === true}
                          onChange={(v) => update({ segment: { ...settings.segment, [field.key]: v } })}
                        />
                      </div>
                    );
                  }
                  if (field.kind === "select") {
                    return (
                      <Select
                        key={field.key}
                        label={field.label}
                        tooltip={field.hint}
                        options={(field.options ?? []).map((o) => ({ label: o || "(default)", value: o }))}
                        value={typeof raw === "string" ? raw : ""}
                        onChange={(v) => update({ segment: { ...settings.segment, [field.key]: v } })}
                      />
                    );
                  }
                  return (
                    <FormEntry
                      key={field.key}
                      label={field.label}
                      tooltip={field.hint}
                      type={field.kind}
                      nullable
                      value={typeof raw === "string" ? raw : ""}
                      onChange={(v) => update({ segment: { ...settings.segment, [field.key]: String(v ?? "") } })}
                    />
                  );
                })}
              </div>
              <div className="grid grid-cols-3 gap-3 items-end">
                <Select
                  label="Model"
                  tooltip="Blank resends the image's own checkpoint. Pick another when your adapter has moved on since the image was made. Check the connection to list what SwarmUI has."
                  options={modelOptions}
                  value={settings.modelOverride}
                  onChange={(v) => update({ modelOverride: v })}
                />
                <FormEntry
                  label="Requests in flight"
                  tooltip="How many images SwarmUI is asked for at once. 1–4."
                  type="number"
                  value={settings.concurrency}
                  onChange={(v) => update({ concurrency: Math.max(1, Math.min(4, Number(v) || 1)) })}
                />
                <DirPicker
                  label="SwarmUI output folder"
                  tooltip="SwarmUI's raw output root (…/Output/local/raw). Searched after the session's source folder; if the file is in neither, it is downloaded next to the source image."
                  value={settings.generated}
                  onChange={(v) => update({ generated: v })}
                />
              </div>
              <div className="grid grid-cols-[2fr_1fr_auto] gap-3 items-end">
                <FormEntry
                  label="SwarmUI URL"
                  value={settings.baseUrl}
                  onChange={(v) => update({ baseUrl: String(v) })}
                />
                <FormEntry
                  label="Token"
                  tooltip="Only for a SwarmUI with accounts enabled."
                  value={settings.token}
                  onChange={(v) => update({ token: String(v) })}
                />
                <Button variant="secondary" size="sm" onClick={() => void check()} disabled={checking}>
                  {checking ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plug className="w-4 h-4" />}
                  Check
                </Button>
              </div>
              {connection ? (
                <div className="text-xs text-[var(--color-on-surface-secondary)] flex items-center gap-2">
                  {connection.startsWith("SwarmUI") ? (
                    <CheckCircle2 className="w-4 h-4" style={{ color: "#22c55e" }} />
                  ) : (
                    <AlertCircle className="w-4 h-4" style={{ color: "#ef4444" }} />
                  )}
                  {connection}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        {/* Preview */}
        <div className="flex flex-col gap-2">
          <div className="text-xs text-[var(--color-on-surface-secondary)] flex items-center gap-2">
            <span>
              {paths.length} image{paths.length === 1 ? "" : "s"} selected
            </span>
            {previewBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
            {previewError ? <span style={{ color: "#ef4444" }}>{previewError}</span> : null}
            {fixes.length === 0 ? <span style={{ color: "#f97316" }}>Pick at least one fix.</span> : null}
          </div>
          <div className="flex flex-col gap-2 max-h-[34vh] overflow-y-auto">
            {(preview.length > 0 ? preview : paths.map(() => null as RegenPreviewItem | null)).map((item, index) => {
              const path = item?.path ?? paths[index];
              return (
                <div
                  key={path}
                  className="border border-[var(--color-border-subtle)] rounded p-2 flex flex-col gap-1"
                  style={{ borderColor: item && !item.ok ? "#ef4444" : undefined }}
                >
                  <div className="flex items-center gap-3 text-xs">
                    <span className="font-mono text-[var(--color-on-surface)] truncate" title={path}>
                      {basename(path)}
                    </span>
                    {item?.ok ? (
                      <span className="text-[var(--color-on-surface-secondary)]">
                        seed {item.seed ?? "random"} · {item.model || "?"} · {item.width}×{item.height}
                        {item.steps != null ? ` · ${item.steps} steps` : ""}
                        {item.cfgscale != null ? ` · cfg ${item.cfgscale}` : ""}
                        {item.loras.length > 0 ? ` · ${item.loras.join(", ")}` : ""}
                      </span>
                    ) : null}
                    {item && !item.ok ? <span style={{ color: "#ef4444" }}>{item.error}</span> : null}
                  </div>
                  {item?.ok ? (
                    <textarea
                      className={`${TEXTAREA_FULL} font-mono text-[11px]`}
                      rows={2}
                      readOnly
                      value={item.prompt}
                    />
                  ) : null}
                  {item?.warnings.map((warning) => (
                    <div key={warning} className="text-[11px]" style={{ color: "#f97316" }}>
                      {warning}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>

        {error ? (
          <div className="text-sm flex items-center gap-2" style={{ color: "#ef4444" }}>
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        ) : null}

        <div className="flex items-center justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={() => void start()} disabled={!canStart}>
            {starting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
            Generate {readyCount} of {paths.length}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
