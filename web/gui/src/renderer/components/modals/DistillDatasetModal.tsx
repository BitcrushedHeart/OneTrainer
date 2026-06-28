import { useState } from "react";

import { distillApi, type DistillManifest } from "@/api/distillApi";
import { Button, DirPicker, FormEntry } from "@/components/shared";

import { ModalBase } from "./ModalBase";

interface Props {
  open: boolean;
  onClose: () => void;
}

type Status = { kind: "ready" | "building" | "success" | "error"; message: string };

const READY: Status = { kind: "ready", message: "Ready" };

function numberOrNull(value: string | number): number | null {
  if (value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function DistillDatasetModal({ open, onClose }: Props) {
  const [sourceFolder, setSourceFolder] = useState("");
  const [outputFolder, setOutputFolder] = useState("");
  const [includeSubdirectories, setIncludeSubdirectories] = useState(true);
  const [valPercentage, setValPercentage] = useState<number>(0);
  const [modelContains, setModelContains] = useState("");
  const [steps, setSteps] = useState<string | number>("");
  const [cfgScale, setCfgScale] = useState<string | number>("");
  const [scheduler, setScheduler] = useState("");
  const [width, setWidth] = useState<string | number>("");
  const [height, setHeight] = useState<string | number>("");
  const [status, setStatus] = useState<Status>(READY);
  const [manifest, setManifest] = useState<DistillManifest | null>(null);

  const isBuilding = status.kind === "building";

  const build = async () => {
    if (!sourceFolder.trim()) {
      setStatus({ kind: "error", message: "Source folder is required." });
      return;
    }
    if (!outputFolder.trim()) {
      setStatus({ kind: "error", message: "Output folder is required." });
      return;
    }

    setStatus({ kind: "building", message: "Building dataset..." });
    setManifest(null);
    try {
      const result = await distillApi.buildDataset({
        source_folder: sourceFolder,
        output_folder: outputFolder,
        include_subdirectories: includeSubdirectories,
        val_percentage: valPercentage,
        filters: {
          model_contains: modelContains.trim() || null,
          steps: numberOrNull(steps),
          cfg_scale: numberOrNull(cfgScale),
          scheduler: scheduler.trim() || null,
          width: numberOrNull(width),
          height: numberOrNull(height),
        },
      });
      setManifest(result);
      setStatus({
        kind: result.ok ? "success" : "error",
        message: result.ok ? `Accepted ${result.accepted} of ${result.scanned} image(s).` : "Dataset build failed.",
      });
    } catch (err) {
      setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  const statusColor =
    status.kind === "error"
      ? "var(--color-error-500)"
      : status.kind === "success"
        ? "var(--color-success-500)"
        : "var(--color-on-surface-secondary)";

  return (
    <ModalBase open={open} onClose={onClose} title="Build Distillation Dataset" size="lg" closeOnBackdrop={!isBuilding}>
      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DirPicker label="Source Folder" value={sourceFolder} onChange={setSourceFolder} disabled={isBuilding} />
          <DirPicker label="Output Folder" value={outputFolder} onChange={setOutputFolder} disabled={isBuilding} />
        </div>

        <label className="flex items-center gap-2 text-sm text-[var(--color-on-surface)]">
          <input
            type="checkbox"
            checked={includeSubdirectories}
            onChange={(e) => setIncludeSubdirectories(e.target.checked)}
            disabled={isBuilding}
          />
          Include subdirectories
        </label>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <FormEntry
            label="Validation %"
            type="number"
            value={valPercentage}
            onChange={(v) => setValPercentage(Number(v))}
            disabled={isBuilding}
          />
          <FormEntry
            label="Model Contains"
            value={modelContains}
            onChange={(v) => setModelContains(String(v))}
            disabled={isBuilding}
          />
          <FormEntry
            label="Scheduler"
            value={scheduler}
            onChange={(v) => setScheduler(String(v))}
            disabled={isBuilding}
          />
          <FormEntry label="Steps" type="number" value={steps} onChange={setSteps} disabled={isBuilding} nullable />
          <FormEntry label="CFG" type="number" value={cfgScale} onChange={setCfgScale} disabled={isBuilding} nullable />
          <FormEntry label="Width" type="number" value={width} onChange={setWidth} disabled={isBuilding} nullable />
          <FormEntry label="Height" type="number" value={height} onChange={setHeight} disabled={isBuilding} nullable />
        </div>

        <div className="flex items-center justify-between gap-3">
          <span className="text-sm" style={{ color: statusColor }}>
            {status.message}
          </span>
          <Button onClick={build} disabled={isBuilding}>
            Build Dataset
          </Button>
        </div>

        {manifest && (
          <div className="rounded-md border border-[var(--color-border-subtle)] p-3 text-sm text-[var(--color-on-surface)]">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <div>Scanned: {manifest.scanned}</div>
              <div>Accepted: {manifest.accepted}</div>
              <div>Train: {manifest.train}</div>
              <div>Val: {manifest.val}</div>
            </div>
            {Object.keys(manifest.rejected).length > 0 && (
              <pre className="mt-3 text-xs overflow-auto">{JSON.stringify(manifest.rejected, null, 2)}</pre>
            )}
          </div>
        )}
      </div>
    </ModalBase>
  );
}
