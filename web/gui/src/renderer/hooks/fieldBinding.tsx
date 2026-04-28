import { createContext, type ReactNode, useCallback, useContext, useMemo } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { getOverrideAt, hasOverrideAt, useQueueOverrideStore } from "@/store/queueOverrideStore";

type Mode = "global" | "override";

interface FieldBindingCtx {
  mode: Mode;
}

const FieldBindingContext = createContext<FieldBindingCtx>({ mode: "global" });

export function FieldBindingProvider({ mode, children }: { mode: Mode; children: ReactNode }) {
  const value = useMemo(() => ({ mode }), [mode]);
  return <FieldBindingContext.Provider value={value}>{children}</FieldBindingContext.Provider>;
}

export function useFieldBindingMode(): Mode {
  return useContext(FieldBindingContext).mode;
}

/**
 * Bound field hook used by every schema-driven widget.
 *
 * In default ("global") mode the value reads/writes the global config store
 * exactly like `useConfigField`. In "override" mode the value reads from the
 * queue override store, falling back to the global config; writes deep-merge
 * into the override map (matching `QueueExecutor.merge_config` shape).
 */
export function useBoundField<T>(path: string | undefined): [T | undefined, (value: T) => void] {
  const mode = useFieldBindingMode();
  const [globalValue, setGlobalValue] = useConfigField<T>(path);
  const overrideValue = useQueueOverrideStore((s) =>
    path && mode === "override" ? (getOverrideAt(s.overrides, path) as T | undefined) : undefined,
  );
  const overrideHas = useQueueOverrideStore((s) =>
    path && mode === "override" ? hasOverrideAt(s.overrides, path) : false,
  );
  const setOverrideField = useQueueOverrideStore((s) => s.setField);

  const setValue = useCallback(
    (next: T) => {
      if (!path) return;
      if (mode === "override") {
        setOverrideField(path, next);
      } else {
        setGlobalValue(next);
      }
    },
    [path, mode, setOverrideField, setGlobalValue],
  );

  if (mode === "override") {
    return [overrideHas ? overrideValue : globalValue, setValue];
  }
  return [globalValue, setValue];
}

export interface FieldStatus {
  isOverridden: boolean;
  reset: () => void;
}

/**
 * Returns override metadata for a path, or null when not in override mode or
 * no path is supplied. Used by `FieldStatusAdorner` and any widget that wants
 * to show the override indicator inline.
 */
export function useFieldStatus(path: string | undefined): FieldStatus | null {
  const mode = useFieldBindingMode();
  const isOverridden = useQueueOverrideStore((s) =>
    path && mode === "override" ? hasOverrideAt(s.overrides, path) : false,
  );
  const removeField = useQueueOverrideStore((s) => s.removeField);
  const reset = useCallback(() => {
    if (path) removeField(path);
  }, [path, removeField]);
  if (mode !== "override" || !path) return null;
  return { isOverridden, reset };
}
