import { create } from "zustand";

import { queueApi, type QueueEntryData, type QueueSettingsData, type QueueState } from "@/api/queueApi";

interface QueueStore {
  entries: QueueEntryData[];
  settings: QueueSettingsData;
  status: string;
  currentEntryId: string | null;
  runIndex: number;
  totalEntries: number;
  selectedEntryId: string | null;
  validationResults: Record<string, { errors: string[]; warnings: string[] }>;
  loading: boolean;

  loadQueue: () => Promise<void>;
  addEntry: (name?: string) => Promise<void>;
  updateEntry: (id: string, data: { name?: string; overrides?: Record<string, unknown> }) => Promise<void>;
  removeEntry: (id: string) => Promise<void>;
  duplicateEntry: (id: string) => Promise<void>;
  reorder: (id: string, direction: "up" | "down") => Promise<void>;
  selectEntry: (id: string | null) => void;
  updateSettings: (settings: Partial<QueueSettingsData>) => Promise<void>;
  validate: () => Promise<void>;
  execute: () => Promise<void>;
  stopCurrent: () => Promise<void>;
  stopAll: () => Promise<void>;
  handleWsMessage: (msg: Record<string, unknown>) => void;
}

export const useQueueStore = create<QueueStore>((set, get) => ({
  entries: [],
  settings: {
    retry_on_error: true,
    retry_from_backup: true,
    max_retries: 2,
    oom_skip_threshold: 3,
    oom_step_window: 10,
  },
  status: "idle",
  currentEntryId: null,
  runIndex: 0,
  totalEntries: 0,
  selectedEntryId: null,
  validationResults: {},
  loading: false,

  loadQueue: async () => {
    set({ loading: true });
    try {
      const state: QueueState = await queueApi.getState();
      set({
        entries: state.entries,
        settings: state.settings,
        status: state.status,
        currentEntryId: state.current_entry_id,
        runIndex: state.run_index,
        totalEntries: state.total_entries,
      });
    } finally {
      set({ loading: false });
    }
  },

  addEntry: async (name = "") => {
    const entry = await queueApi.createEntry(name);
    set((s) => ({ entries: [...s.entries, entry], selectedEntryId: entry.id }));
  },

  updateEntry: async (id, data) => {
    const updated = await queueApi.updateEntry(id, data);
    set((s) => ({
      entries: s.entries.map((e) => (e.id === id ? updated : e)),
    }));
  },

  removeEntry: async (id) => {
    await queueApi.deleteEntry(id);
    set((s) => ({
      entries: s.entries.filter((e) => e.id !== id),
      selectedEntryId: s.selectedEntryId === id ? null : s.selectedEntryId,
    }));
  },

  duplicateEntry: async (id) => {
    const result = await queueApi.duplicateEntry(id);
    if (result.ok && result.entry) {
      await get().loadQueue();
      set({ selectedEntryId: result.entry.id });
    }
  },

  reorder: async (id, direction) => {
    await queueApi.reorder(id, direction);
    await get().loadQueue();
  },

  selectEntry: (id) => set({ selectedEntryId: id }),

  updateSettings: async (settings) => {
    const updated = await queueApi.updateSettings(settings);
    set({ settings: updated });
  },

  validate: async () => {
    const results = await queueApi.validate();
    set({ validationResults: results });
  },

  execute: async () => {
    const result = await queueApi.execute();
    if (result.ok) {
      set({ status: "running" });
      await get().loadQueue();
    }
  },

  stopCurrent: async () => {
    await queueApi.stopCurrent();
  },

  stopAll: async () => {
    await queueApi.stopAll();
    set({ status: "stopping" });
  },

  handleWsMessage: (msg) => {
    const type = msg.type as string;
    if (!type?.startsWith("queue:")) return;

    switch (type) {
      case "queue:entry_started":
        set({
          currentEntryId: msg.entry_id as string,
          runIndex: msg.run_index as number,
          totalEntries: msg.total as number,
        });
        set((s) => ({
          entries: s.entries.map((e) => (e.id === msg.entry_id ? { ...e, status: "RUNNING" } : e)),
        }));
        break;

      case "queue:entry_completed":
        set((s) => ({
          entries: s.entries.map((e) => (e.id === msg.entry_id ? { ...e, status: "COMPLETED" } : e)),
        }));
        break;

      case "queue:entry_failed":
        set((s) => ({
          entries: s.entries.map((e) => (e.id === msg.entry_id ? { ...e, status: "FAILED" } : e)),
        }));
        break;

      case "queue:entry_skipped":
        set((s) => ({
          entries: s.entries.map((e) => (e.id === msg.entry_id ? { ...e, status: "SKIPPED" } : e)),
        }));
        break;

      case "queue:complete":
        set({ status: "idle", currentEntryId: null });
        break;
    }
  },
}));
