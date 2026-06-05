import { useState } from "react";

import { type MergeOFTRequest, toolsApi } from "@/api/toolsApi";
import { Button, FilePicker, FormEntry, Select } from "@/components/shared";
import { useConfigStore } from "@/store/configStore";
import { CONVERT_OUTPUT_DTYPES, CONVERT_OUTPUT_FORMATS } from "@/types/generated/dropdownSources";

import { ModalBase } from "./ModalBase";

export interface MergeOFTModalProps {
  open: boolean;
  onClose: () => void;
}

type StatusKind = "ready" | "merging" | "success" | "error";

interface Status {
  kind: StatusKind;
  message: string;
}

const STATUS_READY: Status = { kind: "ready", message: "Ready" };
const STATUS_MERGING: Status = { kind: "merging", message: "Merging..." };

export function MergeOFTModal({ open, onClose }: MergeOFTModalProps) {
  // Inherit base/transformer/vae from the active TrainConfig (Model tab).
  const config = useConfigStore((s) => s.config);

  const [adapterPath, setAdapterPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [outputDtype, setOutputDtype] = useState("BFLOAT_16");
  const [outputFormat, setOutputFormat] = useState("SAFETENSORS");
  const [strength, setStrength] = useState<number>(1.0);
  const [status, setStatus] = useState<Status>(STATUS_READY);

  const isMerging = status.kind === "merging";

  // transformer/vae model names live on per-component sub-configs in modern TrainConfig.
  const baseModelName = (config?.base_model_name as string | undefined) ?? "";
  const transformerOverride = (config?.transformer?.model_name as string | undefined) ?? "";
  const vaeOverride = (config?.vae?.model_name as string | undefined) ?? "";
  const modelType = (config?.model_type as string | undefined) ?? "";

  const handleMerge = async () => {
    if (!adapterPath.trim()) {
      setStatus({ kind: "error", message: "OFT adapter path is required." });
      return;
    }
    if (!outputPath.trim()) {
      setStatus({ kind: "error", message: "Output path is required." });
      return;
    }
    if (!baseModelName.trim()) {
      setStatus({
        kind: "error",
        message: "Base Model not set in the Model tab. Set it there before merging (e.g. Tongyi-MAI/Z-Image).",
      });
      return;
    }

    setStatus(STATUS_MERGING);
    try {
      const params: MergeOFTRequest = {
        oft_adapter_path: adapterPath,
        output_path: outputPath,
        output_dtype: outputDtype,
        output_model_format: outputFormat,
        strength,
        base_model_name: baseModelName,
        transformer_model_name: transformerOverride,
        vae_model_name: vaeOverride,
      };
      const result = await toolsApi.mergeOft(params);
      if (result.ok) {
        setStatus({
          kind: "success",
          message: `Merge OK — ${result.modules_merged} module(s) baked. Output: ${outputPath}`,
        });
      } else {
        setStatus({ kind: "error", message: result.error ?? "Unknown error during merge." });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setStatus({ kind: "error", message });
    }
  };

  const statusColor = (() => {
    switch (status.kind) {
      case "ready":
        return "var(--color-on-surface-secondary)";
      case "merging":
        return "var(--color-cobalt-600)";
      case "success":
        return "var(--color-success-500)";
      case "error":
        return "var(--color-error-500)";
    }
  })();

  return (
    <ModalBase open={open} onClose={onClose} title="Merge OFT into base" size="md" closeOnBackdrop={!isMerging}>
      <div className="flex flex-col gap-4">
        <div className="rounded-md border border-[var(--color-border-subtle)] p-3 text-sm font-mono whitespace-pre">
          {`Inherited from Model tab:
  model_type:           ${modelType || "(unset)"}
  base_model_name:      ${baseModelName || "(unset)"}
  transformer_override: ${transformerOverride || "(stock)"}
  vae_override:         ${vaeOverride || "(stock)"}`}
        </div>

        <FilePicker
          label="OFT Adapter"
          value={adapterPath}
          onChange={setAdapterPath}
          disabled={isMerging}
          tooltip="Trained OFT / DoRA-OFT adapter (safetensors). Training-time hyperparameters are read from its ot_config metadata."
        />

        <FilePicker
          label="Output Path"
          value={outputPath}
          onChange={setOutputPath}
          disabled={isMerging}
          tooltip="Path for the merged checkpoint. Must differ from the base model / transformer override."
        />

        <Select
          label="Output Data Type"
          options={CONVERT_OUTPUT_DTYPES}
          value={outputDtype}
          onChange={setOutputDtype}
          disabled={isMerging}
        />

        <Select
          label="Output Format"
          options={CONVERT_OUTPUT_FORMATS}
          value={outputFormat}
          onChange={setOutputFormat}
          disabled={isMerging}
        />

        <FormEntry
          label="Strength"
          type="number"
          value={strength}
          onChange={(v) => setStrength(typeof v === "number" ? v : Number(v))}
          disabled={isMerging}
          tooltip={
            "Merge strength: W_final = (1-s)*W_base + s*W_dora. " +
            "1.0 = full merge (matches training output). " +
            "0.7 reproduces the ComfyUI 'LoRA weight = 0.7' interpolation. " +
            "DoRA row-norm invariant check only runs at strength=1.0."
          }
        />
      </div>

      <div className="flex items-center justify-between mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <span
          className="text-sm font-medium truncate max-w-[60%]"
          style={{ color: statusColor }}
          title={status.message}
        >
          {status.message}
        </span>

        <div className="flex gap-3">
          <Button variant="primary" size="md" onClick={handleMerge} disabled={isMerging} loading={isMerging}>
            {isMerging ? "Merging..." : "Merge"}
          </Button>
          <Button variant="secondary" size="md" onClick={onClose} disabled={isMerging}>
            Close
          </Button>
        </div>
      </div>
    </ModalBase>
  );
}
