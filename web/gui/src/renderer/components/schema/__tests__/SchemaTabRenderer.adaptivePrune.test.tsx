// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "@/api/request";
import { SchemaTabRenderer } from "@/components/schema/SchemaTabRenderer";
import { useConfigStore } from "@/store/configStore";
import { useTrainingStore } from "@/store/trainingStore";
import { useUiSchemaStore } from "@/store/uiSchemaStore";
import type { TabDef, UiSchema } from "@/types/uiSchema";

vi.hoisted(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  });
});

vi.mock("@/api/request", () => ({ request: vi.fn() }));

const tab: TabDef = {
  id: "data",
  label: "Data",
  renderer: "schema",
  sections: [
    {
      id: "data-settings",
      label: "Data",
      fields: [
        { key: "adaptive_resolution", label: "Adaptive Resolution", type: "toggle" },
        { key: "clear_cache_before_training", label: "Clear Cache", type: "toggle" },
      ],
    },
  ],
};

const schema = {
  $version: 1,
  predicates: {},
  enums: {},
  dtypeSubsets: {},
  tabs: [tab],
  modals: {},
  optimizerDefaults: {},
  optimizerKeyDetails: {},
  trainingMethodsByModel: {},
} as UiSchema;

describe("adaptive cache prune button", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConfigStore.setState({
      config: { adaptive_resolution: true, clear_cache_before_training: false } as never,
    });
    useUiSchemaStore.setState({ schema });
    useTrainingStore.setState({ status: "idle" });
    vi.stubGlobal(
      "confirm",
      vi.fn(() => true),
    );
    vi.stubGlobal("alert", vi.fn());
  });

  afterEach(() => cleanup());

  it("previews, confirms, then requests targeted cleanup", async () => {
    vi.mocked(request)
      .mockResolvedValueOnce({
        ok: true,
        affected_entries: 2,
        removed_variants: 3,
        files: 1,
        bytes: 1024,
        skipped_missing_metadata: 0,
        entries_without_usable_variant: 1,
      })
      .mockResolvedValueOnce({ ok: true, removed_variants: 3, deleted_files: 1 });

    render(<SchemaTabRenderer tab={tab} />);
    fireEvent.click(screen.getByRole("button", { name: "Prune Upscaled Cache" }));

    await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
    expect(request).toHaveBeenNthCalledWith(1, "/system/cache/adaptive-prune-preview", { method: "POST" });
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("require sourced recaching"));
    expect(request).toHaveBeenNthCalledWith(2, "/system/cache/adaptive-prune-clean", { method: "POST" });
  });

  it("does not clean when confirmation is declined", async () => {
    vi.mocked(request).mockResolvedValueOnce({
      ok: true,
      affected_entries: 1,
      removed_variants: 1,
      files: 1,
      bytes: 10,
      skipped_missing_metadata: 0,
      entries_without_usable_variant: 0,
    });
    vi.mocked(confirm).mockReturnValue(false);

    render(<SchemaTabRenderer tab={tab} />);
    fireEvent.click(screen.getByRole("button", { name: "Prune Upscaled Cache" }));

    await waitFor(() => expect(request).toHaveBeenCalledTimes(1));
  });

  it("opens the ladder submenu and saves a custom snap ladder", async () => {
    useConfigStore.setState({
      config: {
        adaptive_resolution: true,
        adaptive_resolution_ladder: "1024,768,512,256,128",
        resolution: "1024",
        clear_cache_before_training: false,
      } as never,
    });

    render(<SchemaTabRenderer tab={tab} />);
    fireEvent.click(screen.getByRole("button", { name: "Configure Resolution Ladder" }));

    expect(await screen.findByRole("dialog", { name: "Adaptive Resolution Ladder" })).toBeTruthy();
    const input = screen.getByLabelText("Snap resolutions");
    fireEvent.change(input, { target: { value: "896,640,320,192" } });
    expect(screen.getByText("1024")).toBeTruthy();
    expect(screen.getByText("640")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Save Ladder" }));

    await waitFor(() => expect(useConfigStore.getState().config?.adaptive_resolution_ladder).toBe("896,640,320,192"));
  });
});
