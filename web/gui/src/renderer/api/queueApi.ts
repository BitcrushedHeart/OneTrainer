import { request } from "./request";

export interface QueueEntryData {
  id: string;
  name: string;
  overrides: Record<string, unknown>;
  status: string;
  failure_history: Array<{ step: number; error: string; timestamp: string }>;
}

export interface QueueSettingsData {
  retry_on_error: boolean;
  retry_from_backup: boolean;
  max_retries: number;
  oom_skip_threshold: number;
  oom_step_window: number;
}

export interface QueueState {
  status: string;
  current_entry_id: string | null;
  run_index: number;
  total_entries: number;
  settings: QueueSettingsData;
  entries: QueueEntryData[];
}

export interface ActionResponse {
  ok: boolean;
  error?: string;
}

export const queueApi = {
  getState: () => request<QueueState>("/queue"),

  createEntry: (name = "", overrides: Record<string, unknown> = {}) =>
    request<QueueEntryData>("/queue/entry", {
      method: "POST",
      body: JSON.stringify({ name, overrides }),
    }),

  updateEntry: (entryId: string, data: { name?: string; overrides?: Record<string, unknown> }) =>
    request<QueueEntryData>(`/queue/entry/${entryId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  deleteEntry: (entryId: string) =>
    request<ActionResponse>(`/queue/entry/${entryId}`, { method: "DELETE" }),

  duplicateEntry: (entryId: string) =>
    request<{ ok: boolean; entry?: QueueEntryData }>(`/queue/entry/${entryId}/duplicate`, {
      method: "POST",
    }),

  reorder: (entryId: string, direction: "up" | "down") =>
    request<ActionResponse>("/queue/reorder", {
      method: "POST",
      body: JSON.stringify({ entry_id: entryId, direction }),
    }),

  validate: () =>
    request<Record<string, { errors: string[]; warnings: string[] }>>("/queue/validate", {
      method: "POST",
    }),

  execute: () =>
    request<ActionResponse>("/queue/execute", { method: "POST" }),

  stopCurrent: () =>
    request<ActionResponse>("/queue/stop", { method: "POST" }),

  stopAll: () =>
    request<ActionResponse>("/queue/stop-all", { method: "POST" }),

  updateSettings: (settings: Partial<QueueSettingsData>) =>
    request<QueueSettingsData>("/queue/settings", {
      method: "PATCH",
      body: JSON.stringify(settings),
    }),

  exportQueue: () => request<{ settings: QueueSettingsData; entries: QueueEntryData[] }>("/queue/export"),

  importQueue: (data: Record<string, unknown>) =>
    request<{ ok: boolean; warnings: string[] }>("/queue/import", {
      method: "POST",
      body: JSON.stringify({ data }),
    }),
};
