import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { configApi } from "@/api/configApi";
import { useConfigStore } from "@/store/configStore";

vi.mock("@/api/configApi", () => ({
  configApi: {
    getConfig: vi.fn(),
    updateConfig: vi.fn(),
    saveConcepts: vi.fn(),
    saveSamples: vi.fn(),
    getConcepts: vi.fn(),
    getSamples: vi.fn(),
  },
}));

const minimalConfig = {
  learning_rate: 0.0001,
  concepts: [{ name: "foo" }],
  samples: [{ prompt: "bar" }],
} as unknown as Parameters<typeof useConfigStore.setState>[0];

describe("configStore.flushPendingChanges", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    (configApi.updateConfig as ReturnType<typeof vi.fn>).mockResolvedValue(minimalConfig);
    (configApi.saveConcepts as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    (configApi.saveSamples as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    useConfigStore.setState({
      config: { ...minimalConfig } as never,
      isDirty: false,
      _syncTimer: null,
      _conceptSaveTimer: null,
      _sampleSaveTimer: null,
    } as never);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    useConfigStore.getState().destroy();
  });

  it("flushes pending config sync immediately when dirty", async () => {
    useConfigStore.getState().updateField("learning_rate", 0.0002);
    expect(useConfigStore.getState().isDirty).toBe(true);
    expect(configApi.updateConfig).not.toHaveBeenCalled();

    await useConfigStore.getState().flushPendingChanges();

    expect(configApi.updateConfig).toHaveBeenCalledTimes(1);
    expect(useConfigStore.getState().isDirty).toBe(false);
    expect(useConfigStore.getState()._syncTimer).toBeNull();
  });

  it("flushes pending concepts save immediately", async () => {
    useConfigStore.getState().updateField("concepts", [{ name: "baz" }]);
    expect(useConfigStore.getState()._conceptSaveTimer).not.toBeNull();

    await useConfigStore.getState().flushPendingChanges();

    expect(configApi.saveConcepts).toHaveBeenCalledTimes(1);
    expect(useConfigStore.getState()._conceptSaveTimer).toBeNull();
  });

  it("flushes pending samples save immediately", async () => {
    useConfigStore.getState().updateField("samples", [{ prompt: "new" }]);
    expect(useConfigStore.getState()._sampleSaveTimer).not.toBeNull();

    await useConfigStore.getState().flushPendingChanges();

    expect(configApi.saveSamples).toHaveBeenCalledTimes(1);
    expect(useConfigStore.getState()._sampleSaveTimer).toBeNull();
  });

  it("is a no-op when nothing is dirty", async () => {
    await useConfigStore.getState().flushPendingChanges();
    expect(configApi.updateConfig).not.toHaveBeenCalled();
    expect(configApi.saveConcepts).not.toHaveBeenCalled();
    expect(configApi.saveSamples).not.toHaveBeenCalled();
  });
});
