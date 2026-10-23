import { DirPicker, FilePicker, FormEntry, Select, Toggle } from "@/components/shared";
import type { ConceptConfig } from "@/types/generated/config";
import { PROMPT_SOURCES } from "@/types/generated/dropdownSources";
import {
  BalancingStrategyValues,
  ConceptTypeValues,
  KTOLabelValues,
  TimestepDistributionValues,
} from "@/types/generated/enums";

// Per-concept timestep sampling: an empty field / "inherit" leaves the run's value in place.
const INHERIT = "";
const TIMESTEP_DISTRIBUTION_OPTIONS = [
  { label: "inherit", value: INHERIT },
  ...TimestepDistributionValues.map((v) => ({ label: v, value: v })),
];

export interface ConceptGeneralTabProps {
  draft: ConceptConfig;
  update: (path: string, value: unknown) => void;
  updateText: (field: keyof ConceptConfig["text"], value: unknown) => void;
}

export function ConceptGeneralTab({ draft, update, updateText }: ConceptGeneralTabProps) {
  return (
    <div className="grid grid-cols-[140px_1fr] gap-x-4 gap-y-3 items-center">
      <span className="text-sm font-medium text-[var(--color-on-surface)]">Name</span>
      <FormEntry label="" value={draft.name} onChange={(v) => update("name", v)} />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Enabled</span>
      <Toggle value={draft.enabled} onChange={(v) => update("enabled", v)} />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Concept Type</span>
      <Select label="" options={ConceptTypeValues} value={draft.type} onChange={(v) => update("type", v)} />

      {draft.type === "KTO" && (
        <>
          <span className="text-sm font-medium text-[var(--color-on-surface)]">KTO Label</span>
          <Select
            label=""
            options={KTOLabelValues}
            value={draft.kto_label}
            onChange={(v) => update("kto_label", v)}
            tooltip="Liked: the KTO loss pushes these images above the reference model. Disliked: below it. One label per concept."
          />
        </>
      )}

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Path</span>
      <DirPicker
        label=""
        value={draft.path}
        onChange={(v) => update("path", v)}
        tooltip="Directory containing training images"
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Prompt Source</span>
      <Select
        label=""
        options={PROMPT_SOURCES}
        value={draft.text.prompt_source}
        onChange={(v) => updateText("prompt_source", v)}
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Prompt Path</span>
      <FilePicker
        label=""
        value={draft.text.prompt_path}
        onChange={(v) => updateText("prompt_path", v)}
        disabled={draft.text.prompt_source !== "concept"}
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Include Subdirectories</span>
      <Toggle value={draft.include_subdirectories} onChange={(v) => update("include_subdirectories", v)} />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Cache Directory</span>
      <DirPicker
        label=""
        value={draft.cache_dir ?? ""}
        onChange={(v) => update("cache_dir", v)}
        tooltip="Optional. Caches this concept's latents and text encodings here (in image/ and text/ subfolders) instead of the run's cache directory, so every run that trains on this concept can share one cache. Leave empty to use the run's cache directory. Clear Cache leaves this folder alone; Cache GC sweeps it."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Image Variations</span>
      <FormEntry
        label=""
        type="number"
        value={draft.image_variations}
        onChange={(v) => update("image_variations", v)}
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Text Variations</span>
      <FormEntry label="" type="number" value={draft.text_variations} onChange={(v) => update("text_variations", v)} />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Balancing</span>
      <div className="flex gap-2">
        <div className="flex-1">
          <FormEntry label="" type="number" value={draft.balancing} onChange={(v) => update("balancing", v)} />
        </div>
        <div className="w-36">
          <Select
            label=""
            options={BalancingStrategyValues}
            value={draft.balancing_strategy}
            onChange={(v) => update("balancing_strategy", v)}
          />
        </div>
      </div>

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Loss Weight</span>
      <FormEntry label="" type="number" value={draft.loss_weight} onChange={(v) => update("loss_weight", v)} />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Seed</span>
      <FormEntry label="" type="number" value={draft.seed} onChange={(v) => update("seed", v)} />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">DPO Chosen Pattern</span>
      <FormEntry
        label=""
        value={draft.dpo_chosen_pattern}
        onChange={(v) => update("dpo_chosen_pattern", v)}
        tooltip="Turns this concept into a DPO pair concept. Filename pattern (relative to Path) selecting the chosen images, e.g. 'chosen/{}'. Leave empty for a normal concept; both patterns must be set together."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">DPO Rejected Pattern</span>
      <FormEntry
        label=""
        value={draft.dpo_rejected_pattern}
        onChange={(v) => update("dpo_rejected_pattern", v)}
        tooltip="Pattern locating the rejected image for each chosen image, e.g. 'rejected/{}'. '{}' is the stem matched by the chosen pattern; the extension may differ."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">DPO Beta Override</span>
      <FormEntry
        label=""
        type="number"
        value={draft.dpo_beta_override}
        onChange={(v) => update("dpo_beta_override", v)}
        tooltip="Beta for this concept's pairs only, overriding the global RLHF beta. 0 inherits the global beta. Adaptive beta still applies (unless DPO Adaptive Beta is off below): this value is the starting point this concept's own controller scales within x1/4..x4, from its margins alone. Ignored by the IPO objective, which uses tau."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">DPO Adaptive Beta</span>
      <Toggle
        value={draft.dpo_adaptive_beta}
        onChange={(v) => update("dpo_adaptive_beta", v)}
        tooltip="Give this concept its own adaptive beta controller: it scales this concept's beta from this concept's margins alone, so concepts whose margins sit on different scales adapt independently instead of following the dominant one. Off applies the configured beta unscaled and keeps the margins out of adaptive beta altogether."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Supervised Mix Override</span>
      <FormEntry
        label=""
        type="number"
        value={draft.supervised_mix_override}
        onChange={(v) => update("supervised_mix_override", v)}
        tooltip="Weight of the supervised loss on this concept's chosen images, overriding the global RLHF Supervised Mix. -1 (any negative value) inherits the global mix, 0 turns the term off for this concept, a positive value replaces it. Uses the run's masked loss, so a low unmasked weight keeps it inside the mask."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Chosen Anchor Override</span>
      <FormEntry
        label=""
        type="number"
        value={draft.dpo_chosen_anchor_override}
        onChange={(v) => update("dpo_chosen_anchor_override", v)}
        tooltip="DPOP chosen anchor coefficient for this concept's pairs, overriding the global RLHF Chosen Anchor. -1 (any negative value) inherits the global value, 0 turns the anchor off for this concept, a positive value replaces it (and applies even when the global anchor is 0). Still multiplied by the pair's beta, so a low-beta concept needs a proportionally larger value: at beta 2, 25-50 pulls about as hard as 0.25-0.5 at beta 200. Ignored by KTO."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Unmasked Weight Override</span>
      <FormEntry
        label=""
        type="number"
        value={draft.unmasked_weight_override}
        onChange={(v) => update("unmasked_weight_override", v)}
        tooltip="Loss weight outside the mask for this concept only (masked training). 0 trains the masked region alone; a DPO concept's margin is then computed inside the mask only. Negative inherits the run's Unmasked Weight."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Unmasked Probability Override</span>
      <FormEntry
        label=""
        type="number"
        value={draft.unmasked_probability_override}
        onChange={(v) => update("unmasked_probability_override", v)}
        tooltip="Chance (0-1) that a sample of this concept trains on the whole image instead of its mask, drawn per sample each epoch; a DPO pair is unmasked together. Negative inherits the run's Unmasked Probability. Inpainting models apply only the run-wide value."
      />

      <span className="col-span-2 mt-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-on-surface-variant)]">
        Timestep sampling (empty = inherit the run's value)
      </span>

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Noising Strength Min / Max</span>
      <div className="flex gap-2">
        <div className="flex-1">
          <FormEntry
            label=""
            type="number"
            nullable
            value={draft.min_noising_strength_override}
            onChange={(v) => update("min_noising_strength_override", v)}
            tooltip="Minimum noising strength (0-1) for this concept's samples, overriding the run's Min Noising Strength. E.g. 0.1 skips the very lowest noise levels (JPEG artifacts, sensor grain). Empty inherits."
          />
        </div>
        <div className="flex-1">
          <FormEntry
            label=""
            type="number"
            nullable
            value={draft.max_noising_strength_override}
            onChange={(v) => update("max_noising_strength_override", v)}
            tooltip="Maximum noising strength (0-1) for this concept's samples, overriding the run's Max Noising Strength. E.g. 0.6 keeps a real-photo concept on fine detail. Empty inherits."
          />
        </div>
      </div>

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Noising Weight / Bias</span>
      <div className="flex gap-2">
        <div className="flex-1">
          <FormEntry
            label=""
            type="number"
            nullable
            value={draft.noising_weight_override}
            onChange={(v) => update("noising_weight_override", v)}
            tooltip="Weight parameter of the timestep distribution for this concept (same meaning as the run's Noising Weight). Empty inherits."
          />
        </div>
        <div className="flex-1">
          <FormEntry
            label=""
            type="number"
            nullable
            value={draft.noising_bias_override}
            onChange={(v) => update("noising_bias_override", v)}
            tooltip="Bias parameter of the timestep distribution for this concept (same meaning as the run's Noising Bias; a negative bias nudges logit-normal draws toward lower noise). Empty inherits."
          />
        </div>
      </div>

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Timestep Distribution</span>
      <Select
        label=""
        options={TIMESTEP_DISTRIBUTION_OPTIONS}
        value={draft.timestep_distribution_override ?? INHERIT}
        onChange={(v) => update("timestep_distribution_override", v === INHERIT ? null : v)}
        tooltip="Timestep distribution for this concept's samples; inherit uses the run's. A concept on a different distribution converts the run's random draw instead of consuming extra random numbers."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Timestep Shift</span>
      <FormEntry
        label=""
        type="number"
        nullable
        value={draft.timestep_shift_override}
        onChange={(v) => update("timestep_shift_override", v)}
        tooltip="Timestep shift for this concept's samples, overriding the run's Timestep Shift (and the dynamic resolution-based shift when enabled). Empty inherits."
      />

      <span className="text-sm font-medium text-[var(--color-on-surface)]">Range After Shift</span>
      <Toggle
        value={draft.noising_range_after_shift}
        onChange={(v) => update("noising_range_after_shift", v)}
        tooltip="Off: the min / max range applies before the timestep shift, like the run's values, so under dynamic shifting the same bounds land at a different noise level per resolution. On: the draw (shaped by weight / bias) covers the whole schedule, is shifted, and the shifted value is mapped linearly into [min, max], so the bounds mean the same noise level at every resolution without piling probability at the bounds."
      />
    </div>
  );
}
