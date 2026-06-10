import { Layers, RotateCcw, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { request } from "@/api/request";
import { ConceptGrid } from "@/components/concepts/ConceptGrid";
import { ConceptEditorModal } from "@/components/modals/ConceptEditorModal";
import { Button, FilePicker } from "@/components/shared";
import { useBoundField, useFieldStatus } from "@/hooks/fieldBinding";
import { useConfigStore } from "@/store/configStore";
import { useQueueOverrideStore } from "@/store/queueOverrideStore";
import type { ConceptConfig } from "@/types/generated/config";

const DEFAULT_CONCEPT: ConceptConfig = {
  name: "",
  path: "",
  seed: 42,
  enabled: true,
  type: "STANDARD",
  include_subdirectories: false,
  image_variations: 1,
  text_variations: 1,
  balancing: 1,
  balancing_strategy: "REPEATS",
  loss_weight: 1.0,
  dpo_chosen_pattern: "",
  dpo_rejected_pattern: "",
  concept_stats: {},
  image: {
    enable_crop_jitter: true,
    enable_random_flip: false,
    enable_fixed_flip: false,
    enable_random_rotate: false,
    enable_fixed_rotate: false,
    random_rotate_max_angle: 0,
    enable_random_brightness: false,
    enable_fixed_brightness: false,
    random_brightness_max_strength: 0,
    enable_random_contrast: false,
    enable_fixed_contrast: false,
    random_contrast_max_strength: 0,
    enable_random_saturation: false,
    enable_fixed_saturation: false,
    random_saturation_max_strength: 0,
    enable_random_hue: false,
    enable_fixed_hue: false,
    random_hue_max_strength: 0,
    enable_resolution_override: false,
    resolution_override: "512",
    enable_random_circular_mask_shrink: false,
    enable_random_mask_rotate_crop: false,
  },
  text: {
    prompt_source: "sample",
    prompt_path: "",
    enable_tag_shuffling: false,
    tag_delimiter: ",",
    keep_tags_count: 1,
    tag_dropout_enable: false,
    tag_dropout_mode: "RANDOM",
    tag_dropout_probability: 0,
    tag_dropout_special_tags_mode: "NONE",
    tag_dropout_special_tags: "",
    tag_dropout_special_tags_regex: false,
    caps_randomize_enable: false,
    caps_randomize_mode: "",
    caps_randomize_probability: 0,
    caps_randomize_lowercase: false,
  },
};

function cloneConcept(c: ConceptConfig): ConceptConfig {
  return JSON.parse(JSON.stringify(c)) as ConceptConfig;
}

export function QueueOverrideConceptsTab() {
  const [conceptFileName] = useBoundField<string>("concept_file_name");
  const fileNameStatus = useFieldStatus("concept_file_name");
  const overrideConcepts = useQueueOverrideStore((s) => s.overrides.concepts as ConceptConfig[] | undefined);
  const conceptsOverridden = overrideConcepts !== undefined;
  const setOverrideField = useQueueOverrideStore((s) => s.setField);
  const removeOverrideField = useQueueOverrideStore((s) => s.removeField);
  const globalConcepts = useConfigStore((s) => s.config?.concepts);

  const [fileConcepts, setFileConcepts] = useState<ConceptConfig[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);

  // Load concepts from the effective path. When the override path matches the
  // global path the cached store value is fine; otherwise hit the API.
  const globalConceptFileName = useConfigStore((s) => s.config?.concept_file_name);
  const usingGlobalFile = !fileNameStatus?.isOverridden;

  useEffect(() => {
    if (!conceptFileName) {
      setFileConcepts([]);
      setLoadError(null);
      return;
    }
    if (usingGlobalFile && globalConceptFileName === conceptFileName) {
      setFileConcepts((globalConcepts ?? []) as ConceptConfig[]);
      setLoadError(null);
      return;
    }
    let cancelled = false;
    request<ConceptConfig[]>(`/concepts?path=${encodeURIComponent(conceptFileName)}`)
      .then((data) => {
        if (!cancelled) {
          setFileConcepts(data);
          setLoadError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setFileConcepts([]);
          setLoadError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [conceptFileName, usingGlobalFile, globalConceptFileName, globalConcepts]);

  const effectiveConcepts: ConceptConfig[] = useMemo(() => {
    if (overrideConcepts) return overrideConcepts;
    return fileConcepts ?? [];
  }, [overrideConcepts, fileConcepts]);

  const writeConcepts = useCallback(
    (next: ConceptConfig[]) => {
      setOverrideField("concepts", next);
    },
    [setOverrideField],
  );

  const handleResetToFile = useCallback(() => {
    removeOverrideField("concepts");
  }, [removeOverrideField]);

  const handleAdd = useCallback(() => {
    const next = [
      ...effectiveConcepts,
      { ...cloneConcept(DEFAULT_CONCEPT), seed: Math.floor(Math.random() * 2 ** 30) },
    ];
    writeConcepts(next);
  }, [effectiveConcepts, writeConcepts]);

  const handleRemove = useCallback(
    (index: number) => {
      writeConcepts(effectiveConcepts.filter((_, i) => i !== index));
    },
    [effectiveConcepts, writeConcepts],
  );

  const handleClone = useCallback(
    (index: number) => {
      const cloned = cloneConcept(effectiveConcepts[index]);
      cloned.seed = Math.floor(Math.random() * 2 ** 30);
      const next = [...effectiveConcepts];
      next.splice(index + 1, 0, cloned);
      writeConcepts(next);
    },
    [effectiveConcepts, writeConcepts],
  );

  const handleToggle = useCallback(
    (index: number, enabled: boolean) => {
      writeConcepts(effectiveConcepts.map((c, i) => (i === index ? { ...c, enabled } : c)));
    },
    [effectiveConcepts, writeConcepts],
  );

  const handleOpen = useCallback((index: number) => {
    setEditingIndex(index);
  }, []);

  const handleSave = useCallback(
    (updated: ConceptConfig) => {
      if (editingIndex === null) return;
      writeConcepts(effectiveConcepts.map((c, i) => (i === editingIndex ? updated : c)));
    },
    [editingIndex, effectiveConcepts, writeConcepts],
  );

  return (
    <div className="flex flex-col gap-5">
      {/* Source row — file picker on the left, source-mode chip + reset on the right */}
      <div
        className="rounded-[var(--radius-sm)] border border-[var(--color-border-subtle)]
          bg-[var(--color-surface-raised)] px-4 py-3
          flex flex-wrap items-end gap-x-4 gap-y-3"
      >
        <div className="flex-1 min-w-[300px]">
          <FilePicker
            label="Concept File"
            configPath="concept_file_name"
            tooltip="JSON file describing this run's concepts. Override the path to point at a different concept set."
          />
        </div>

        <div className="flex items-center gap-2">
          <SourceModeChip overridden={conceptsOverridden} count={effectiveConcepts.length} />
          <Button
            variant={conceptsOverridden ? "secondary" : "ghost"}
            size="sm"
            onClick={handleResetToFile}
            disabled={!conceptsOverridden}
            title={
              conceptsOverridden
                ? "Discard concept-list overrides for this entry and use the file as-is"
                : "No concept-list overrides to reset"
            }
          >
            <RotateCcw className="w-4 h-4" />
            Reset to file
          </Button>
        </div>
      </div>

      {loadError && (
        <div
          className="rounded-[var(--radius-sm)] border border-[var(--color-error-500-alpha-40)]
            bg-[var(--color-error-500-alpha-08)] px-3 py-2
            text-[var(--text-caption)] text-[var(--color-error-500)]"
        >
          Failed to load concepts: {loadError}
        </div>
      )}

      <ConceptGrid
        concepts={effectiveConcepts}
        onAdd={handleAdd}
        onOpen={handleOpen}
        onRemove={handleRemove}
        onClone={handleClone}
        onToggle={handleToggle}
      />

      <ConceptEditorModal
        open={editingIndex !== null}
        onClose={() => setEditingIndex(null)}
        concept={editingIndex !== null ? (effectiveConcepts[editingIndex] ?? null) : null}
        onSave={handleSave}
      />
    </div>
  );
}

function SourceModeChip({ overridden, count }: { overridden: boolean; count: number }) {
  if (overridden) {
    return (
      <span
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full
          border border-[var(--color-cobalt-600-alpha-25)] bg-[var(--color-cobalt-600-alpha-06)]
          text-[var(--text-caption)] font-medium leading-none text-[var(--color-cobalt-600)]"
        title="This entry's concept list is snapshotted into the queue and won't change if the file changes."
      >
        <Sparkles className="w-3.5 h-3.5" />
        {count} snapshot{count === 1 ? "" : "ted"} concept{count === 1 ? "" : "s"}
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full
        border border-[var(--color-border-subtle)] bg-transparent
        text-[var(--text-caption)] font-medium leading-none text-[var(--color-on-surface-secondary)]"
      title="This entry follows the concept file live; edits to the file affect this run."
    >
      <Layers className="w-3.5 h-3.5" />
      Following file ({count})
    </span>
  );
}
