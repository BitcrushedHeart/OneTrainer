import { useEffect, useRef, useState } from "react";

import { toolsApi, type ToolStatusResponse } from "@/api/toolsApi";
import { Button, DirPicker, FilePicker, FormEntry, ProgressBar, Select, Toggle } from "@/components/shared";

import { ModalBase } from "./ModalBase";

export interface MaskToolModalProps {
  open: boolean;
  onClose: () => void;
}

const MASK_MODELS = ["ClipSeg", "Rembg", "Rembg-Human", "Hex Color", "YOLO"];

const MASK_MODES = ["replace", "fill", "add", "subtract", "blend"];
const MASK_MODE_LABELS: Record<string, string> = {
  replace: "Replace all masks",
  fill: "Create if absent",
  add: "Add to existing",
  subtract: "Subtract from existing",
  blend: "Blend with existing",
};

interface MaskState {
  model: string;
  folder: string;
  prompt: string;
  mode: string;
  threshold: number;
  smooth: number;
  expand: number;
  alpha: number;
  include_subdirectories: boolean;
  model_path: string;
}

const DEFAULT_STATE: MaskState = {
  model: "ClipSeg",
  folder: "",
  prompt: "",
  mode: "fill",
  threshold: 0.3,
  smooth: 5,
  expand: 10,
  alpha: 1.0,
  include_subdirectories: false,
  model_path: "",
};

export function MaskToolModal({ open, onClose }: MaskToolModalProps) {
  const [state, setState] = useState<MaskState>({ ...DEFAULT_STATE });
  const [status, setStatus] = useState<ToolStatusResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ultralyticsAvailable, setUltralyticsAvailable] = useState<boolean | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const update = <K extends keyof MaskState>(field: K, value: MaskState[K]) => {
    setState((prev) => ({ ...prev, [field]: value }));
  };

  useEffect(() => {
    if (!open) return;
    toolsApi
      .getCapabilities()
      .then((caps) => {
        setUltralyticsAvailable(caps.ultralytics_available);
      })
      .catch(() => {
        /* ignore */
      });
  }, [open]);

  useEffect(() => {
    if (!isRunning) return;

    const poll = async () => {
      try {
        const s = await toolsApi.getStatus();
        setStatus(s);
        if (s.status === "completed" || s.status === "error" || s.status === "idle") {
          setIsRunning(false);
          if (s.status === "error" && s.error) {
            setError(s.error);
          }
        }
      } catch {
        /* ignore */
      }
    };

    pollRef.current = setInterval(poll, 500);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [isRunning]);

  const handleGenerate = async () => {
    if (!state.folder) {
      setError("Please select a folder.");
      return;
    }
    if (state.model === "YOLO" && !state.model_path) {
      setError("Please select a YOLO model file (.pt).");
      return;
    }
    setError(null);
    setIsRunning(true);
    setStatus(null);

    try {
      const result = await toolsApi.generateMasks({
        model: state.model,
        folder: state.folder,
        prompt: state.prompt,
        mode: state.mode,
        threshold: state.threshold,
        smooth: state.smooth,
        expand: state.expand,
        alpha: state.alpha,
        include_subdirectories: state.include_subdirectories,
        model_path: state.model === "YOLO" ? state.model_path : undefined,
      });

      if (!result.ok) {
        setError(result.error ?? "Failed to start mask generation");
        setIsRunning(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setIsRunning(false);
    }
  };

  const handleCancel = async () => {
    try {
      await toolsApi.cancel();
      setIsRunning(false);
    } catch {
      /* ignore */
    }
  };

  const handleOpenEditor = async () => {
    const api = window.electronAPI;
    if (api?.openMaskEditor) {
      await api.openMaskEditor(state.folder);
    } else {
      window.open(`/?route=mask-editor&folder=${encodeURIComponent(state.folder)}`, "_blank");
    }
  };

  const progress = status && status.max_progress > 0 ? (status.progress / status.max_progress) * 100 : 0;

  const progressLabel =
    status && status.max_progress > 0
      ? `${status.progress} / ${status.max_progress}`
      : isRunning
        ? "Starting..."
        : "0 / 0";

  return (
    <ModalBase open={open} onClose={onClose} title="Batch Generate Masks" size="md" closeOnBackdrop={!isRunning}>
      <div className="flex flex-col gap-4">
        <Select
          label="Model"
          options={MASK_MODELS}
          value={state.model}
          onChange={(v) => update("model", v)}
          disabled={isRunning}
          formatLabel={(v) => v}
        />

        {state.model === "YOLO" && (
          <>
            <FilePicker
              label="YOLO Model File"
              value={state.model_path}
              onChange={(v) => update("model_path", v)}
              disabled={isRunning}
              filters={[{ name: "YOLO Models", extensions: ["pt"] }]}
              tooltip="Select a YOLO .pt model file for segmentation"
            />
            {ultralyticsAvailable === false && (
              <div className="p-2 rounded-[var(--radius-sm)] bg-[var(--color-error-500)]/10 border border-[var(--color-error-500)]/30">
                <p className="text-xs text-[var(--color-error-500)]">
                  Ultralytics is not installed. Install with: pip install ultralytics
                </p>
              </div>
            )}
          </>
        )}

        <DirPicker label="Folder" value={state.folder} onChange={(v) => update("folder", v)} disabled={isRunning} />

        {state.model !== "YOLO" && (
          <FormEntry
            label="Prompt"
            value={state.prompt}
            onChange={(v) => update("prompt", String(v))}
            placeholder="Masking prompt (for ClipSeg)..."
            disabled={isRunning}
          />
        )}

        <Select
          label="Mode"
          options={MASK_MODES}
          value={state.mode}
          onChange={(v) => update("mode", v)}
          disabled={isRunning}
          formatLabel={(v) => MASK_MODE_LABELS[v] ?? v}
        />

        <div className="grid grid-cols-2 gap-3">
          <FormEntry
            label="Threshold"
            value={state.threshold}
            onChange={(v) => update("threshold", Number(v))}
            type="number"
            placeholder="0.0 - 1.0"
            disabled={isRunning}
          />
          <FormEntry
            label="Smooth"
            value={state.smooth}
            onChange={(v) => update("smooth", Number(v))}
            type="number"
            placeholder="5"
            disabled={isRunning}
          />
          <FormEntry
            label="Expand"
            value={state.expand}
            onChange={(v) => update("expand", Number(v))}
            type="number"
            placeholder="10"
            disabled={isRunning}
          />
          <FormEntry
            label="Alpha"
            value={state.alpha}
            onChange={(v) => update("alpha", Number(v))}
            type="number"
            placeholder="1.0"
            disabled={isRunning}
          />
        </div>

        <Toggle
          label="Include subfolders"
          value={state.include_subdirectories}
          onChange={(v) => update("include_subdirectories", v)}
          disabled={isRunning}
        />

        <div className="pt-2">
          <ProgressBar value={progress} label={progressLabel} indeterminate={isRunning && progress === 0} />
        </div>

        {error && (
          <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-error-500)]/10 border border-[var(--color-error-500)]/30">
            <p className="text-sm text-[var(--color-error-500)]">{error}</p>
          </div>
        )}

        {status?.status === "completed" && (
          <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-cobalt-600)]/10 border border-[var(--color-cobalt-600)]/30">
            <p className="text-sm text-[var(--color-cobalt-600)]">Mask generation completed successfully.</p>
          </div>
        )}
      </div>

      <div className="flex justify-between mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <div className="flex gap-2">
          <Button variant="primary" onClick={handleGenerate} disabled={isRunning}>
            {isRunning ? "Generating..." : "Create Masks"}
          </Button>
          <Button variant="secondary" onClick={handleOpenEditor} disabled={isRunning}>
            Open Interactive Editor
          </Button>
          {isRunning && (
            <Button variant="danger" onClick={handleCancel}>
              Cancel
            </Button>
          )}
        </div>
        <Button variant="secondary" onClick={onClose} disabled={isRunning}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
