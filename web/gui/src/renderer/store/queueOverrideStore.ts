import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

import { queueApi } from "@/api/queueApi";

const SYNC_DEBOUNCE_MS = 500;

export function getOverrideAt(obj: Record<string, unknown>, path: string): unknown {
  const keys = path.split(".");
  let cur: unknown = obj;
  for (const key of keys) {
    if (cur === undefined || cur === null) return undefined;
    if (typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[key];
  }
  return cur;
}

export function hasOverrideAt(obj: Record<string, unknown>, path: string): boolean {
  const keys = path.split(".");
  let cur: unknown = obj;
  for (const key of keys) {
    if (cur === undefined || cur === null) return false;
    if (typeof cur !== "object") return false;
    if (!Object.prototype.hasOwnProperty.call(cur, key)) return false;
    cur = (cur as Record<string, unknown>)[key];
  }
  return true;
}

function setOverrideAt(obj: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split(".");
  let cur: Record<string, unknown> = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    const next = cur[k];
    if (next === undefined || next === null || typeof next !== "object" || Array.isArray(next)) {
      cur[k] = {};
    }
    cur = cur[k] as Record<string, unknown>;
  }
  cur[keys[keys.length - 1]] = value;
}

function deleteKey(obj: Record<string, unknown>, key: string): void {
  // ESLint forbids dynamic `delete obj[var]`. Reflect.deleteProperty is the
  // sanctioned equivalent and is faithful to the same JS semantics.
  Reflect.deleteProperty(obj, key);
}

function removeOverrideAt(obj: Record<string, unknown>, path: string): void {
  const keys = path.split(".");
  const stack: Array<{ parent: Record<string, unknown>; key: string }> = [];
  let cur: Record<string, unknown> = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    const next = cur[k];
    if (next === undefined || next === null || typeof next !== "object" || Array.isArray(next)) return;
    stack.push({ parent: cur, key: k });
    cur = next as Record<string, unknown>;
  }
  deleteKey(cur, keys[keys.length - 1]);
  while (stack.length > 0) {
    const frame = stack.pop();
    if (!frame) break;
    const { parent, key } = frame;
    const node = parent[key];
    if (node && typeof node === "object" && !Array.isArray(node) && Object.keys(node).length === 0) {
      deleteKey(parent, key);
    } else {
      break;
    }
  }
}

interface QueueOverrideState {
  entryId: string | null;
  overrides: Record<string, unknown>;
  isSaving: boolean;
  saveError: string | null;
  _syncTimer: ReturnType<typeof setTimeout> | null;

  open: (entryId: string, overrides: Record<string, unknown>) => void;
  close: () => void;
  setField: (path: string, value: unknown) => void;
  removeField: (path: string) => void;
  setOverrides: (overrides: Record<string, unknown>) => void;
  resetAll: () => void;
  flush: () => Promise<void>;
}

export const useQueueOverrideStore = create<QueueOverrideState>()(
  immer((set, get) => {
    function scheduleSync(): void {
      const existing = get()._syncTimer;
      if (existing !== null) clearTimeout(existing);
      const timer = setTimeout(() => {
        set((draft) => {
          draft._syncTimer = null;
        });
        void persist();
      }, SYNC_DEBOUNCE_MS);
      set((draft) => {
        draft._syncTimer = timer;
      });
    }

    async function persist(): Promise<void> {
      const { entryId, overrides } = get();
      if (!entryId) return;
      set((draft) => {
        draft.isSaving = true;
        draft.saveError = null;
      });
      try {
        await queueApi.updateEntry(entryId, { overrides });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        set((draft) => {
          draft.saveError = message;
        });
      } finally {
        set((draft) => {
          draft.isSaving = false;
        });
      }
    }

    return {
      entryId: null,
      overrides: {},
      isSaving: false,
      saveError: null,
      _syncTimer: null,

      open: (entryId, overrides) => {
        set((draft) => {
          draft.entryId = entryId;
          draft.overrides = JSON.parse(JSON.stringify(overrides ?? {})) as Record<string, unknown>;
          draft.saveError = null;
        });
      },

      close: () => {
        const timer = get()._syncTimer;
        if (timer !== null) clearTimeout(timer);
        set((draft) => {
          draft._syncTimer = null;
          draft.entryId = null;
          draft.overrides = {};
          draft.saveError = null;
        });
      },

      setField: (path, value) => {
        set((draft) => {
          setOverrideAt(draft.overrides, path, value);
        });
        scheduleSync();
      },

      removeField: (path) => {
        set((draft) => {
          removeOverrideAt(draft.overrides, path);
        });
        scheduleSync();
      },

      setOverrides: (overrides) => {
        set((draft) => {
          draft.overrides = JSON.parse(JSON.stringify(overrides ?? {})) as Record<string, unknown>;
        });
        scheduleSync();
      },

      resetAll: () => {
        set((draft) => {
          draft.overrides = {};
        });
        scheduleSync();
      },

      flush: async () => {
        const timer = get()._syncTimer;
        if (timer !== null) {
          clearTimeout(timer);
          set((draft) => {
            draft._syncTimer = null;
          });
          await persist();
        }
      },
    };
  }),
);
