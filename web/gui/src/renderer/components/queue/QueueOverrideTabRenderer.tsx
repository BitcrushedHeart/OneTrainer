import { SchemaTabRenderer } from "@/components/schema/SchemaTabRenderer";
import { FieldBindingProvider } from "@/hooks/fieldBinding";
import type { TabDef } from "@/types/uiSchema";

export interface QueueOverrideTabRendererProps {
  tab: TabDef;
}

/**
 * Renders a regular schema-driven config tab inside the queue override editor.
 * The wrapping `FieldBindingProvider mode="override"` flips every nested
 * `useBoundField` call to read merged (global+override) values and write into
 * the queue override store.
 */
export function QueueOverrideTabRenderer({ tab }: QueueOverrideTabRendererProps) {
  return (
    <FieldBindingProvider mode="override">
      <SchemaTabRenderer tab={tab} />
    </FieldBindingProvider>
  );
}
