import { DirPicker, FilePicker, FormEntry, Select, Toggle } from "@/components/shared";
import type { ConceptConfig } from "@/types/generated/config";
import { PROMPT_SOURCES } from "@/types/generated/dropdownSources";
import { BalancingStrategyValues, ConceptTypeValues } from "@/types/generated/enums";

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
    </div>
  );
}
