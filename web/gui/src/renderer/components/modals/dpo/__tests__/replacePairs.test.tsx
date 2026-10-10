// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { dpoReplaceApi, type ReplaceGroup, type ReplaceStatus } from "@/api/dpoReplaceApi";

import { ReplacePairsModal } from "../../ReplacePairsModal";

vi.mock("@/api/dpoReplaceApi", () => ({
  MAX_IMAGES_PER_PAIR: 10,
  dpoReplaceApi: Object.fromEntries(
    [
      "concepts",
      "start",
      "restore",
      "status",
      "current",
      "pause",
      "resume",
      "close",
      "best",
      "apply",
      "applyPairs",
      "skip",
      "retry",
      "undo",
    ].map((key) => [key, vi.fn()]),
  ),
}));
vi.mock("@/api/swarmApi", () => ({
  swarmApi: { check: vi.fn(), lastParams: vi.fn().mockResolvedValue({ ok: true, params: { model: "test" } }) },
}));

let group: ReplaceGroup | null;
let status: ReplaceStatus;

function makeGroup(count = 4): ReplaceGroup {
  return {
    session_id: "session",
    group_id: "group",
    caption: "Exact on-disk caption",
    best_id: null,
    size: [768, 1024],
    images: Array.from({ length: count }, (_, i) => ({
      id: `id-${i}`,
      url: `/dpo/replace/image/session/group/id-${i}`,
    })),
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  group = makeGroup();
  status = {
    ok: true,
    active: true,
    session_id: "session",
    running: true,
    paused: false,
    total: 9,
    counts: { queued: 8, generating: 0, ready: 1, applied: 0, skipped: 0, failed: 0 },
    can_undo: false,
  };
  vi.mocked(dpoReplaceApi.concepts).mockResolvedValue({ ok: true, concepts: ["/dataset"] });
  vi.mocked(dpoReplaceApi.status).mockImplementation(async () => ({ ...status }));
  vi.mocked(dpoReplaceApi.current).mockImplementation(async () => ({ ok: true, group: group ? { ...group } : null }));
  vi.mocked(dpoReplaceApi.best).mockImplementation(async (_, id) => {
    if (!group) throw new Error("No group");
    group = { ...group, best_id: id };
    return { ok: true, group };
  });
  vi.mocked(dpoReplaceApi.apply).mockImplementation(async () => {
    group = null;
    return { ok: true };
  });
  vi.mocked(dpoReplaceApi.applyPairs).mockImplementation(async (_, pairs) => {
    group = null;
    return { ok: true, pairs: pairs.length };
  });
  vi.mocked(dpoReplaceApi.pause).mockImplementation(async () => {
    status = { ...status, paused: true };
    return { ok: true };
  });
});
afterEach(cleanup);

describe("blind replacement", () => {
  it.each([3, 4, 5])("presents %i anonymous candidates with no source labels", async (count) => {
    group = makeGroup(count);
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    await screen.findByText("Select the BEST image");
    expect(screen.getAllByRole("img")).toHaveLength(count);
    screen.getAllByRole("img").forEach((image, index) => {
      expect(image).toHaveAttribute("alt", `Image ${index + 1}`);
      expect(image.getAttribute("src")).not.toMatch(/chosen|rejected|\.png|path=/);
    });
    expect(screen.queryByText(/P\(AI\)|suggestion|sorted by|NEW/)).toBeNull();
  });

  it("marks only the current rejected image, and warns against keeping it as WORST", async () => {
    group = makeGroup(4);
    group.images[2] = { ...group.images[2], current_rejected: true };
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    await screen.findByText("Select the BEST image");
    expect(screen.getAllByText("CURRENT REJECTED")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Preview Image 3" })).toHaveTextContent("current rejected");
    expect(screen.queryByText(/current chosen/i)).toBeNull();
    expect(screen.queryByText(/keeps the old negative/)).toBeNull();
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the WORST image");
    expect(screen.getByText(/keeps the old negative/)).toBeInTheDocument();
  });

  it("picks BEST then WORST and saves immediately without a general confirmation", async () => {
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.contextMenu(await screen.findByRole("button", { name: "Preview Image 2" }));
    await screen.findByText("Select the WORST image");
    expect(dpoReplaceApi.best).toHaveBeenCalledWith(expect.anything(), "id-1");
    fireEvent.keyDown(window, { key: "3" });
    await waitFor(() =>
      expect(dpoReplaceApi.applyPairs).toHaveBeenCalledWith(expect.objectContaining({ best_id: "id-1" }), [
        { chosen_id: "id-1", rejected_id: "id-2" },
      ]),
    );
    await waitFor(() => expect(screen.queryByText("Select the WORST image")).toBeNull());
  });

  it("picking BEST again frees it, and change best works without reordering", async () => {
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.contextMenu(await screen.findByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the WORST image");
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the BEST image");
    expect(dpoReplaceApi.best).toHaveBeenLastCalledWith(expect.anything(), null);
    expect(dpoReplaceApi.applyPairs).not.toHaveBeenCalled();
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 2" }));
    await screen.findByText("Select the WORST image");
    fireEvent.click(screen.getByRole("button", { name: "Change best" }));
    await screen.findByText("Select the BEST image");
    expect(screen.getAllByRole("img").map((image) => image.getAttribute("alt"))).toEqual([
      "Image 1",
      "Image 2",
      "Image 3",
      "Image 4",
    ]);
  });

  it("uses neutral labels in the full-size preview", async () => {
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Preview Image 3" }));
    const images = screen.getAllByAltText("Image 3");
    expect(images).toHaveLength(2);
    expect(images[1]).toHaveAttribute("src", "/api/dpo/replace/image/session/group/id-2");
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.getAllByAltText("Image 3")).toHaveLength(1));
  });

  it("warns only when the backend identifies an exact reversal and supports override", async () => {
    vi.mocked(dpoReplaceApi.applyPairs).mockResolvedValueOnce({ ok: false, requires_reversal_confirmation: true });
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.contextMenu(await screen.findByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the WORST image");
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 2" }));
    const confirm = await screen.findByRole("button", { name: "Replace anyway" });
    fireEvent.click(confirm);
    await waitFor(() =>
      expect(dpoReplaceApi.applyPairs).toHaveBeenLastCalledWith(
        expect.anything(),
        [{ chosen_id: "id-0", rejected_id: "id-1" }],
        true,
      ),
    );
  });

  it("keeps review usable while Queue 1 is pausing", async () => {
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Pause" }));
    await screen.findByText("Pausing after current image…");
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the WORST image");
  });

  it("ignores obsolete saved controls and retains connection settings", async () => {
    status = { ok: true, active: false };
    group = null;
    localStorage.setItem(
      "ot.replacePairs",
      JSON.stringify({
        concept: "/dataset",
        generated: "/output",
        baseUrl: "http://swarm:7801",
        autoApply: true,
        allowPromote: false,
        order: "stale_first",
        imagesPerPair: 8,
      }),
    );
    vi.mocked(dpoReplaceApi.restore).mockResolvedValue({ ok: true, active: false });
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    await screen.findByText("Select DPO Concept");
    expect(screen.getByDisplayValue("http://swarm:7801")).toBeInTheDocument();
    expect(screen.queryByText(/Allow another|Auto.apply|Order/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Start / resume pass" }));
    await waitFor(() =>
      expect(dpoReplaceApi.start).toHaveBeenCalledWith(expect.objectContaining({ images_per_pair: 2 })),
    );
  });

  it("Shift adds worse images so one BEST can penalise the old chosen too, then a plain pick saves all", async () => {
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.contextMenu(await screen.findByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the WORST image");
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 3" }), { shiftKey: true });
    fireEvent.keyDown(window, { key: "!", code: "Digit4", shiftKey: true });
    expect(await screen.findByText("Pairs to save:")).toBeInTheDocument();
    expect(screen.getAllByText(/WORSE than 1/)).toHaveLength(2);
    expect(dpoReplaceApi.applyPairs).not.toHaveBeenCalled();
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 2" }));
    await waitFor(() =>
      expect(dpoReplaceApi.applyPairs).toHaveBeenCalledWith(expect.anything(), [
        { chosen_id: "id-0", rejected_id: "id-2" },
        { chosen_id: "id-0", rejected_id: "id-3" },
        { chosen_id: "id-0", rejected_id: "id-1" },
      ]),
    );
  });

  it("builds pairs with a second BEST and saves them with Enter", async () => {
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.contextMenu(await screen.findByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the WORST image");
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 4" }), { shiftKey: true });
    // Free the first BEST, then pick a second one against the same negative.
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 1" }));
    await screen.findByText(/Select another BEST image, or press Enter to save 1 pair/);
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 2" }));
    await screen.findByText("Select the WORST image");
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 4" }), { shiftKey: true });
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() =>
      expect(dpoReplaceApi.applyPairs).toHaveBeenCalledWith(expect.anything(), [
        { chosen_id: "id-0", rejected_id: "id-3" },
        { chosen_id: "id-1", rejected_id: "id-3" },
      ]),
    );
  });

  it("refuses a worse image as a BEST", async () => {
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    fireEvent.contextMenu(await screen.findByRole("button", { name: "Preview Image 1" }));
    await screen.findByText("Select the WORST image");
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 2" }), { shiftKey: true });
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 1" }));
    await screen.findByText(/Select another BEST image/);
    fireEvent.contextMenu(screen.getByRole("button", { name: "Preview Image 2" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("cannot be on both sides");
  });

  it("selects the eleventh and twelfth images with - and =", async () => {
    group = makeGroup(12);
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    await screen.findByText("Select the BEST image");
    fireEvent.keyDown(window, { key: "=", code: "Equal" });
    await waitFor(() => expect(dpoReplaceApi.best).toHaveBeenCalledWith(expect.anything(), "id-11"));
    await screen.findByText("Select the WORST image");
    fireEvent.keyDown(window, { key: "-", code: "Minus" });
    await waitFor(() =>
      expect(dpoReplaceApi.applyPairs).toHaveBeenCalledWith(expect.anything(), [
        { chosen_id: "id-11", rejected_id: "id-10" },
      ]),
    );
  });

  it("offers up to ten images per pair", async () => {
    status = { ok: true, active: false };
    group = null;
    vi.mocked(dpoReplaceApi.restore).mockResolvedValue({ ok: true, active: false });
    localStorage.setItem("ot.replacePairs.v2", JSON.stringify({ concept_dir: "/dataset", images_per_pair: 10 }));
    render(<ReplacePairsModal open onClose={vi.fn()} />);
    await screen.findByText("Select DPO Concept");
    expect(screen.getByDisplayValue("10")).toBeInTheDocument();
    const counts = screen.getAllByRole("option").filter((option) => /^[0-9]+$/.test(option.textContent ?? ""));
    expect(counts).toHaveLength(10);
  });
});
