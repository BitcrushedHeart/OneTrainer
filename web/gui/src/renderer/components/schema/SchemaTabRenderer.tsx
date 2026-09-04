import { Search, SlidersHorizontal, Trash2 } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { lazy, Suspense } from "react";

import { request } from "@/api/request";
import { Button, SchemaField, SectionCard } from "@/components/shared";
import { useConfigStore } from "@/store/configStore";
import { useTrainingStore } from "@/store/trainingStore";
import { useUiSchemaStore } from "@/store/uiSchemaStore";
import type { LossWeight, ModelType } from "@/types/generated/enums";
import { LOSS_WEIGHT_SUPPORTS_FLOW_MATCHING } from "@/types/generated/lossWeightInfo";
import { MODEL_TYPE_GROUPS } from "@/types/generated/modelTypeInfo";
import type { FieldDef, SectionDef, TabDef } from "@/types/uiSchema";
import { evaluateCondition } from "@/utils/conditionEval";

import { SchemaModal } from "./SchemaModal";

const OptimizerParamsModal = lazy(() =>
  import("@/components/modals/OptimizerParamsModal").then((m) => ({ default: m.OptimizerParamsModal })),
);
const SchedulerParamsModal = lazy(() =>
  import("@/components/modals/SchedulerParamsModal").then((m) => ({ default: m.SchedulerParamsModal })),
);
const TimestepDistModal = lazy(() =>
  import("@/components/modals/TimestepDistModal").then((m) => ({ default: m.TimestepDistModal })),
);
const OffloadingModal = lazy(() =>
  import("@/components/modals/OffloadingModal").then((m) => ({ default: m.OffloadingModal })),
);
const ValidationCheckerModal = lazy(() =>
  import("@/components/modals/ValidationCheckerModal").then((m) => ({ default: m.ValidationCheckerModal })),
);
const AdaptiveResolutionLadderModal = lazy(() =>
  import("@/components/modals/AdaptiveResolutionLadderModal").then((m) => ({
    default: m.AdaptiveResolutionLadderModal,
  })),
);

const CUSTOM_MODAL_RENDERERS: Record<
  string,
  React.LazyExoticComponent<React.ComponentType<{ open: boolean; onClose: () => void }>>
> = {
  "optimizer-params": OptimizerParamsModal,
  "scheduler-params": SchedulerParamsModal,
  "timestep-dist": TimestepDistModal,
  offloading: OffloadingModal,
  "validation-checker": ValidationCheckerModal,
  "adaptive-resolution-ladder": AdaptiveResolutionLadderModal,
};

interface SchemaTabRendererProps {
  tab: TabDef;
}

export function SchemaTabRenderer({ tab }: SchemaTabRendererProps) {
  const schema = useUiSchemaStore((s) => s.schema);
  const config = useConfigStore((s) => s.config);
  const [activeModal, setActiveModal] = useState<string | null>(null);
  const [cleanCacheBusy, setCleanCacheBusy] = useState(false);
  const [pruneCacheBusy, setPruneCacheBusy] = useState(false);
  const trainingStatus = useTrainingStore((s) => s.status);
  const trainingActive = trainingStatus === "preparing" || trainingStatus === "training";

  const predicates = useMemo(() => schema?.predicates ?? {}, [schema]);
  const configRecord = useMemo(() => (config ?? {}) as Record<string, unknown>, [config]);

  const resolvedSections = useMemo(() => {
    const defs = tab.sectionDefs ?? {};

    if (tab.variants && tab.variants.length > 0 && tab.sectionDefs) {
      let matchedVariant = tab.variants[0];
      for (const variant of tab.variants) {
        if (evaluateCondition(variant.when, configRecord, predicates)) {
          matchedVariant = variant;
          break;
        }
      }

      if (tab.layout === "three-column") {
        return matchedVariant.columns.map((col) => col.sections.map((sectionId) => defs[sectionId]).filter(Boolean));
      }
      const allSections = matchedVariant.columns.flatMap((col) =>
        col.sections.map((sectionId) => defs[sectionId]).filter(Boolean),
      );
      return [allSections];
    }

    return tab.sections ? [tab.sections] : [];
  }, [tab, configRecord, predicates]);

  const resolveEnumOptions = useCallback(
    (field: FieldDef): string[] => {
      if (field.enumRef && schema?.enums[field.enumRef]) {
        const values = schema.enums[field.enumRef].values;
        if (field.key === "loss_weight_fn") {
          const modelType = configRecord["model_type"] as ModelType | undefined;
          if (modelType) {
            const flowMatching = MODEL_TYPE_GROUPS.is_flow_matching ?? [];
            const isFlowMatching = flowMatching.includes(modelType);
            return values.filter(
              (v) => v === "CONSTANT" || LOSS_WEIGHT_SUPPORTS_FLOW_MATCHING[v as LossWeight] === isFlowMatching,
            );
          }
        }
        return values;
      }
      return [];
    },
    [schema, configRecord],
  );

  const resolveKvOptions = useCallback(
    (field: FieldDef): Array<{ label: string; value: string }> | undefined => {
      if (field.dtypeSubset && schema?.dtypeSubsets[field.dtypeSubset]) {
        return schema.dtypeSubsets[field.dtypeSubset];
      }
      return field.options;
    },
    [schema],
  );

  const handleAdvancedClick = useCallback(
    (fieldKey: string) => {
      // Walk resolved sections to find the field and its modal ref
      for (const group of resolvedSections) {
        for (const section of group) {
          for (const field of section.fields) {
            if (field.key === fieldKey && field.modal) {
              setActiveModal(field.modal);
              return;
            }
          }
        }
      }
    },
    [resolvedSections],
  );

  const closeModal = useCallback(() => setActiveModal(null), []);

  const visibleFields = useCallback(
    (fields: FieldDef[]) => fields.filter((f) => evaluateCondition(f.visibility, configRecord, predicates)),
    [configRecord, predicates],
  );

  const isSectionVisible = useCallback(
    (section: SectionDef) => evaluateCondition(section.visibility, configRecord, predicates),
    [configRecord, predicates],
  );

  const renderSection = (section: SectionDef) => {
    if (!isSectionVisible(section)) return null;
    const fields = visibleFields(section.fields);
    if (fields.length === 0) return null;

    return (
      <SectionCard key={section.id} title={section.label}>
        <div
          className={tab.layout === "three-column" ? "flex flex-col gap-4" : "grid grid-cols-1 md:grid-cols-2 gap-4"}
        >
          {fields.map((field) => (
            <SchemaField
              key={field.key}
              field={field}
              onAdvancedClick={handleAdvancedClick}
              resolveOptions={resolveEnumOptions}
              resolveKvOptions={resolveKvOptions}
            />
          ))}
          {tab.id === "data" && fields.some((field) => field.key === "adaptive_resolution") && (
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setActiveModal("adaptive-resolution-ladder")}
                aria-label="Configure Resolution Ladder"
              >
                <SlidersHorizontal className="h-4 w-4" />
                Configure Ladder
              </Button>
            </div>
          )}
          {tab.id === "general" && fields.some((f) => f.key === "validation") && (
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setActiveModal("validation-checker")}
                disabled={!configRecord.validation}
              >
                <Search className="w-4 h-4 mr-1" />
                Check Validation
              </Button>
            </div>
          )}
          {tab.id === "data" && fields.some((f) => f.key === "clear_cache_before_training") && (
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                loading={cleanCacheBusy}
                disabled={cleanCacheBusy}
                onClick={async () => {
                  if (cleanCacheBusy) return;
                  setCleanCacheBusy(true);
                  try {
                    const preview = await request<{
                      ok: boolean;
                      total_files: number;
                      total_mb: number;
                      error?: string;
                    }>("/system/cache/gc-preview", { method: "POST" });
                    if (!preview.ok) {
                      alert(preview.error ?? "Failed to preview cache");
                      return;
                    }
                    if (preview.total_files === 0) {
                      alert("No orphaned cache files found.");
                      return;
                    }
                    if (confirm(`Remove ${preview.total_files} orphaned cache files (${preview.total_mb} MB)?`)) {
                      const result = await request<{ ok: boolean; error?: string }>("/system/cache/gc-clean", {
                        method: "POST",
                      });
                      if (!result.ok) {
                        alert(result.error ?? "Failed to clean cache");
                      } else {
                        alert(`Removed ${preview.total_files} orphaned cache files (${preview.total_mb} MB).`);
                      }
                    }
                  } catch (err) {
                    alert(`Clean cache failed: ${err instanceof Error ? err.message : String(err)}`);
                  } finally {
                    setCleanCacheBusy(false);
                  }
                }}
              >
                <Trash2 className="w-4 h-4 mr-1" />
                {cleanCacheBusy ? "Scanning…" : "Clean Cache"}
              </Button>
              <Button
                variant="secondary"
                size="sm"
                loading={pruneCacheBusy}
                disabled={pruneCacheBusy || trainingActive || configRecord.adaptive_resolution !== true}
                title={
                  configRecord.adaptive_resolution !== true
                    ? "Enable Adaptive Resolution first"
                    : trainingActive
                      ? "Unavailable while training is active"
                      : "Preview and remove only cache variants that enlarged source images"
                }
                onClick={async () => {
                  if (pruneCacheBusy || trainingActive || configRecord.adaptive_resolution !== true) return;
                  setPruneCacheBusy(true);
                  try {
                    const preview = await request<{
                      ok: boolean;
                      affected_entries: number;
                      removed_variants: number;
                      files: number;
                      bytes: number;
                      skipped_missing_metadata: number;
                      entries_without_usable_variant: number;
                      message?: string;
                      error?: string;
                    }>("/system/cache/adaptive-prune-preview", { method: "POST" });
                    if (!preview.ok) {
                      alert(preview.error ?? "Failed to preview adaptive cache cleanup");
                      return;
                    }
                    if (preview.removed_variants === 0) {
                      alert(preview.message ?? "No upscaled image-cache variants were found.");
                      return;
                    }
                    const warnings = [
                      preview.entries_without_usable_variant > 0
                        ? `${preview.entries_without_usable_variant} entries will require sourced recaching or be dropped in sourceless mode.`
                        : "",
                      preview.skipped_missing_metadata > 0
                        ? `${preview.skipped_missing_metadata} legacy entries lack dimensions and will be skipped.`
                        : "",
                    ]
                      .filter(Boolean)
                      .join("\n");
                    const mb = (preview.bytes / (1024 * 1024)).toFixed(1);
                    const prompt = `Remove ${preview.removed_variants} upscaled variants from ${preview.affected_entries} entries and delete ${preview.files} unreferenced files (${mb} MB)?${warnings ? `\n\n${warnings}` : ""}`;
                    if (!confirm(prompt)) return;
                    const result = await request<{
                      ok: boolean;
                      removed_variants: number;
                      deleted_files: number;
                      error?: string;
                    }>("/system/cache/adaptive-prune-clean", { method: "POST" });
                    if (!result.ok) {
                      alert(result.error ?? "Failed to prune upscaled cache variants");
                    } else {
                      alert(
                        `Removed ${result.removed_variants} variants and ${result.deleted_files} unreferenced files.`,
                      );
                    }
                  } catch (err) {
                    alert(`Prune cache failed: ${err instanceof Error ? err.message : String(err)}`);
                  } finally {
                    setPruneCacheBusy(false);
                  }
                }}
              >
                <Trash2 className="w-4 h-4 mr-1" />
                {pruneCacheBusy ? "Scanning…" : "Prune Upscaled Cache"}
              </Button>
            </div>
          )}
        </div>
      </SectionCard>
    );
  };

  const activeModalDef = activeModal ? schema?.modals[activeModal] : undefined;
  const CustomModalComponent = activeModal
    ? CUSTOM_MODAL_RENDERERS[activeModalDef?.renderer ?? activeModal]
    : undefined;

  return (
    <>
      {tab.layout === "three-column" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {resolvedSections.map((column, colIdx) => (
            <div key={colIdx} className="flex flex-col gap-6">
              {column.map((section) => renderSection(section))}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-6">{resolvedSections.flat().map((section) => renderSection(section))}</div>
      )}

      {CustomModalComponent && (
        <Suspense fallback={null}>
          <CustomModalComponent open={!!activeModal} onClose={closeModal} />
        </Suspense>
      )}

      {activeModalDef && !activeModalDef.renderer && (
        <SchemaModal
          open={!!activeModal}
          onClose={closeModal}
          modal={activeModalDef}
          configRecord={configRecord}
          predicates={predicates}
        />
      )}
    </>
  );
}
