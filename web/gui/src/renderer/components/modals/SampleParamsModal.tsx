import { FilePicker, FormEntry, Select, Toggle } from "@/components/shared";
import { NoiseSchedulerValues } from "@/types/generated/enums";

import { ModalBase } from "./ModalBase";

export interface SampleParamsModalProps {
  open: boolean;
  onClose: () => void;
  sampleIndex: number;
}

export function SampleParamsModal({ open, onClose, sampleIndex }: SampleParamsModalProps) {
  const p = `samples.${sampleIndex}`;

  return (
    <ModalBase open={open} onClose={onClose} title={`Sample ${sampleIndex + 1} Parameters`} size="lg">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormEntry label="Width" configPath={`${p}.width`} type="number" />
        <FormEntry label="Height" configPath={`${p}.height`} type="number" />
        <FormEntry label="Seed" configPath={`${p}.seed`} type="number" />
        <Toggle configPath={`${p}.random_seed`} label="Random Seed" />
        <FormEntry label="Diffusion Steps" configPath={`${p}.diffusion_steps`} type="number" />
        <FormEntry label="CFG Scale" configPath={`${p}.cfg_scale`} type="number" />
        <Select label="Noise Scheduler" configPath={`${p}.noise_scheduler`} options={[...NoiseSchedulerValues]} />

        <FormEntry label="Frames" configPath={`${p}.frames`} type="number" tooltip="Frame count for video models" />
        <FormEntry
          label="Audio Length (s)"
          configPath={`${p}.length`}
          type="number"
          tooltip="Length in seconds for audio models"
        />

        <FormEntry label="TE1 Layer Skip" configPath={`${p}.text_encoder_1_layer_skip`} type="number" />
        <FormEntry label="TE1 Seq Length" configPath={`${p}.text_encoder_1_sequence_length`} type="number" nullable />
        <FormEntry label="TE2 Layer Skip" configPath={`${p}.text_encoder_2_layer_skip`} type="number" />
        <FormEntry label="TE2 Seq Length" configPath={`${p}.text_encoder_2_sequence_length`} type="number" nullable />
        <FormEntry label="TE3 Layer Skip" configPath={`${p}.text_encoder_3_layer_skip`} type="number" />
        <FormEntry label="TE4 Layer Skip" configPath={`${p}.text_encoder_4_layer_skip`} type="number" />

        <Toggle configPath={`${p}.transformer_attention_mask`} label="Attention Mask" />
        <Toggle configPath={`${p}.force_last_timestep`} label="Force Last Timestep" />
        <Toggle configPath={`${p}.sample_inpainting`} label="Sample Inpainting" />
      </div>

      <div className="grid grid-cols-1 gap-4 mt-4">
        <FilePicker label="Base Image" configPath={`${p}.base_image_path`} tooltip="Base image for inpainting" />
        <FilePicker label="Mask Image" configPath={`${p}.mask_image_path`} tooltip="Mask image for inpainting" />
      </div>

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <button
          onClick={onClose}
          className="px-4 py-2 rounded-[var(--radius-sm)] text-sm font-medium bg-transparent border border-[var(--color-border-subtle)] text-[var(--color-on-surface)] hover:border-[var(--color-cobalt-600)] transition-colors duration-200 cursor-pointer"
        >
          Close
        </button>
      </div>
    </ModalBase>
  );
}
