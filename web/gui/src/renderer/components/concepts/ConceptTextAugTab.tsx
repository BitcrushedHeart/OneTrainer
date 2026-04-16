import { AlertTriangle } from "lucide-react";

import { FormEntry, Select, Toggle } from "@/components/shared";
import type { ConceptConfig } from "@/types/generated/config";

const TAG_DROPOUT_MODE_OPTIONS = ["FULL", "RANDOM", "RANDOM WEIGHTED"];
const TAG_DROPOUT_SPECIAL_TAGS_MODE_OPTIONS = ["NONE", "BLACKLIST", "WHITELIST"];

export interface ConceptTextAugTabProps {
    draft: ConceptConfig;
    updateText: (field: keyof ConceptConfig["text"], value: unknown) => void;
}

export function ConceptTextAugTab({ draft, updateText }: ConceptTextAugTabProps) {
    const isDpo = draft.type.startsWith("DPO_");

    return (
        <div className="grid grid-cols-[140px_1fr_140px_1fr] gap-x-4 gap-y-3 items-center">
            {isDpo && (
                <div className="col-span-4 flex items-center gap-2 p-3 rounded-lg bg-[rgba(234,179,8,0.1)] text-[var(--color-warning-500)] text-sm mb-2">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    Text augmentations are disabled for DPO concepts. Chosen/rejected pairs must use identical captions.
                </div>
            )}
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Tag Shuffling</span>
            <Toggle
                value={draft.text.enable_tag_shuffling}
                onChange={(v) => updateText("enable_tag_shuffling", v)}
                disabled={isDpo}
            />
            <span />
            <span />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Tag Delimiter</span>
            <FormEntry
                label=""
                value={draft.text.tag_delimiter}
                onChange={(v) => updateText("tag_delimiter", v)}
                disabled={isDpo}
            />
            <span />
            <span />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Keep Tag Count</span>
            <FormEntry
                label=""
                type="number"
                value={draft.text.keep_tags_count}
                onChange={(v) => updateText("keep_tags_count", v)}
                disabled={isDpo}
            />
            <span />
            <span />

            <div className="col-span-4 border-t border-[var(--color-border-subtle)] mt-1" />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Tag Dropout</span>
            <Toggle
                value={draft.text.tag_dropout_enable}
                onChange={(v) => updateText("tag_dropout_enable", v)}
                disabled={isDpo}
            />
            <span />
            <span />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Dropout Mode</span>
            <Select
                label=""
                options={TAG_DROPOUT_MODE_OPTIONS}
                value={draft.text.tag_dropout_mode}
                onChange={(v) => updateText("tag_dropout_mode", v)}
                disabled={isDpo}
            />
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Probability</span>
            <FormEntry
                label=""
                type="number"
                value={draft.text.tag_dropout_probability}
                onChange={(v) => updateText("tag_dropout_probability", v)}
                disabled={isDpo}
            />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Special Tags</span>
            <Select
                label=""
                options={TAG_DROPOUT_SPECIAL_TAGS_MODE_OPTIONS}
                value={draft.text.tag_dropout_special_tags_mode}
                onChange={(v) => updateText("tag_dropout_special_tags_mode", v)}
                disabled={isDpo}
            />
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Tags</span>
            <FormEntry
                label=""
                value={draft.text.tag_dropout_special_tags}
                onChange={(v) => updateText("tag_dropout_special_tags", v)}
                disabled={isDpo}
            />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Special Tags Regex</span>
            <Toggle
                value={draft.text.tag_dropout_special_tags_regex}
                onChange={(v) => updateText("tag_dropout_special_tags_regex", v)}
                disabled={isDpo}
            />
            <span />
            <span />

            <div className="col-span-4 border-t border-[var(--color-border-subtle)] mt-1" />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Randomize Caps</span>
            <Toggle
                value={draft.text.caps_randomize_enable}
                onChange={(v) => updateText("caps_randomize_enable", v)}
                disabled={isDpo}
            />
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Force Lowercase</span>
            <Toggle
                value={draft.text.caps_randomize_lowercase}
                onChange={(v) => updateText("caps_randomize_lowercase", v)}
                disabled={isDpo}
            />

            <span className="text-sm font-medium text-[var(--color-on-surface)]">Caps Mode</span>
            <FormEntry
                label=""
                value={draft.text.caps_randomize_mode}
                onChange={(v) => updateText("caps_randomize_mode", v)}
                placeholder="capslock,title,first,random"
                disabled={isDpo}
            />
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Probability</span>
            <FormEntry
                label=""
                type="number"
                value={draft.text.caps_randomize_probability}
                onChange={(v) => updateText("caps_randomize_probability", v)}
                disabled={isDpo}
            />
        </div>
    );
}
