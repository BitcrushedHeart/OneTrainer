import { describe, expect, it } from "vitest";

import { customRow, mergeAdded, PRESET_ROWS, reconcileRows, rowsToFixes, segmentParamsFromForm } from "../regenFixes";

describe("rowsToFixes", () => {
  it("sends only enabled rows with a target, trimmed and clamped", () => {
    const rows = PRESET_ROWS.map((row) => ({ ...row }));
    rows[0].enabled = true; // faces
    rows[2].enabled = true; // text
    rows[2].creativity = 7;
    rows[2].threshold = -1;
    rows[2].subPrompt = "  crisp letters ";
    const blank = { ...customRow(), target: "   " };
    expect(rowsToFixes([...rows, blank])).toEqual([
      { target: "face", creativity: 0.95, threshold: 0.3, sub_prompt: "" },
      { target: "text", creativity: 1, threshold: 0, sub_prompt: "crisp letters" },
    ]);
  });

  it("ships a Hands/Feet preset as one merged mask", () => {
    const hands = PRESET_ROWS.find((row) => row.id === "hands_feet");
    expect(hands?.target).toBe("hands|feet");
    expect(PRESET_ROWS.map((row) => row.label)).toEqual([
      "Fix Faces",
      "Fix Background People",
      "Fix Text",
      "Fix Hands/Feet",
      "Fix Limbs/Body",
    ]);
  });
});

describe("segmentParamsFromForm", () => {
  it("drops blanks and false toggles", () => {
    expect(
      segmentParamsFromForm({ segmentmaskblur: "12", segmentmodel: "", savesegmentmask: false, segmentsteps: "" }),
    ).toEqual({ segmentmaskblur: "12" });
    expect(segmentParamsFromForm({ savesegmentmask: true })).toEqual({ savesegmentmask: true });
  });
});

describe("mergeAdded", () => {
  it("appends new paths once, in order, without touching existing ones", () => {
    expect(mergeAdded(["a", "b"], ["b", "c", "c", "d"])).toEqual(["a", "b", "c", "d"]);
    const same = ["a"];
    expect(mergeAdded(same, [])).toBe(same);
  });
});

describe("reconcileRows", () => {
  it("keeps stored edits, restores missing presets and keeps custom rows", () => {
    const stored = [
      { ...PRESET_ROWS[0], enabled: true, creativity: 0.7, label: "renamed" },
      { ...customRow(), id: "custom-1", target: "yolo-hand.pt" },
    ];
    const rows = reconcileRows(stored);
    expect(rows.map((row) => row.id)).toEqual([
      "faces",
      "background_people",
      "text",
      "hands_feet",
      "limbs",
      "custom-1",
    ]);
    expect(rows[0]).toMatchObject({ enabled: true, creativity: 0.7, label: "Fix Faces", custom: false });
    expect(rows[5]).toMatchObject({ target: "yolo-hand.pt", custom: true });
    expect(reconcileRows(undefined)).toHaveLength(5);
  });
});
