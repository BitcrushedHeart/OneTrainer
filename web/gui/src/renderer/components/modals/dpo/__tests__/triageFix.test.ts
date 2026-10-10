// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";

import type { TriageItem } from "@/api/dpoTriageFixApi";

import {
  BACKGROUND_PEOPLE_TARGET,
  captionFilterActive,
  categoriesFromDetected,
  cleanColorCorrection,
  cleanCropReference,
  cleanLoras,
  cleanTriageSessions,
  cursorForStem,
  DEFAULT_CATEGORIES,
  emptyTags,
  filterByCaption,
  filterByCaptionOrStem,
  INSTRUCTIONS,
  ledgerCounts,
  LEGACY_INSTRUCTIONS,
  loadTriageFixSettings,
  loadTriageSessions,
  migrateSegment,
  newOtherFix,
  parseCaptionFilter,
  parseSegmentString,
  pushRecent,
  reconcileCategories,
  saveTriageFixSettings,
  secondKind,
  secondRoundHandled,
  secondRoundOrder,
  seededShuffle,
  sessionIdFor,
  setOthers,
  tagsFromFixes,
  tagsToFixes,
  toggleTag,
  TRIAGE_FIX_STORAGE_KEY,
  triageOrder,
} from "../triageFix";

function item(stem: string, status: TriageItem["status"] = null): TriageItem {
  return {
    stem,
    pair_id: 0,
    chosen_path: `/ds/chosen/${stem}.png`,
    caption: "",
    caption_source: "pair",
    width: 64,
    height: 48,
    status,
    detected: null,
    fixes: [],
    error: null,
    attempt: 0,
    round: 1,
    decided_at: null,
    original_path: null,
    has_fix_pair: false,
    second_pass: null,
    second_from: null,
    waiting_on: null,
    claimed_by: [],
  };
}

describe("seededShuffle", () => {
  it("is deterministic, keeps every element and leaves the input alone", () => {
    const input = Array.from({ length: 20 }, (_, i) => `pair_${i}`);
    const copy = [...input];
    const a = seededShuffle(input, 4242);
    const b = seededShuffle(input, 4242);
    const c = seededShuffle(input, 4243);
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    expect(a).not.toEqual(input);
    expect([...a].sort()).toEqual([...input].sort());
    expect(input).toEqual(copy);
  });
});

describe("triageOrder", () => {
  it("drops decided items without reordering the rest", () => {
    const items = Array.from({ length: 12 }, (_, i) => item(`pair_${String(i).padStart(4, "0")}`));
    const before = triageOrder(items, 7).map((i) => i.stem);
    const decided = new Set([before[0], before[3], before[8]]);
    const after = triageOrder(
      items.map((i) => (decided.has(i.stem) ? { ...i, status: "fixed" as const } : i)),
      7,
    ).map((i) => i.stem);
    expect(after).toEqual(before.filter((stem) => !decided.has(stem)));
  });

  it("keeps only undecided or tagged items and ignores input order", () => {
    const items = [
      item("b", "queued"),
      item("a"),
      item("c", "tagged"),
      item("d", "ready"),
      item("e", "skipped"),
      item("f", "kept"),
      item("g", "deleted"),
    ];
    const order = triageOrder(items, 1).map((i) => i.stem);
    expect([...order].sort()).toEqual(["a", "c"]);
    expect(triageOrder([...items].reverse(), 1).map((i) => i.stem)).toEqual(order);
  });
});

describe("tags", () => {
  it("toggles and converts to fixes in category order", () => {
    let tags = toggleTag(emptyTags(), "hands");
    tags = toggleTag(tags, "faces");
    expect([...tags.ids]).toEqual(["hands", "faces"]);
    expect(tagsToFixes(tags, DEFAULT_CATEGORIES).map((f) => f.id)).toEqual(["faces", "hands"]);
    tags = toggleTag(tags, "faces");
    expect(tagsToFixes(tags, DEFAULT_CATEGORIES).map((f) => f.id)).toEqual(["hands"]);
  });

  it("only sends Other with a target and carries its instruction", () => {
    let tags = toggleTag(emptyTags(), "other");
    expect(tagsToFixes(tags, DEFAULT_CATEGORIES)).toEqual([]);
    tags = setOthers(tags, [
      { ...newOtherFix(DEFAULT_CATEGORIES, " newspaper "), instruction: " Make the headline legible " },
    ]);
    const fixes = tagsToFixes(tags, DEFAULT_CATEGORIES);
    expect(fixes).toHaveLength(1);
    expect(fixes[0]).toMatchObject({
      id: "other",
      target: "newspaper",
      instruction: "Make the headline legible",
      include_caption: false,
      creativity: 1,
      threshold: 0.3,
    });
    const cleared = setOthers(tags, [newOtherFix(DEFAULT_CATEGORIES)]);
    expect(cleared.ids.has("other")).toBe(false);
    expect(tagsToFixes(cleared, DEFAULT_CATEGORIES)).toEqual([]);
    const back = tagsFromFixes(fixes);
    expect(back.others.map((o) => o.target)).toEqual(["newspaper"]);
    expect(back.ids.has("other")).toBe(true);
    const withMask = tagsFromFixes([{ id: "mask", target: "", instruction: "x" }, ...fixes]);
    expect([...withMask.ids]).toEqual(["other"]);
    // Older ledger records with the pre-split id toggle both new keys.
    const legacy = tagsFromFixes([{ id: "hands_feet", target: "hands|feet", instruction: "" }]);
    expect([...legacy.ids].sort()).toEqual(["feet", "hands"]);
  });

  it("sends several Other selections, each with its own strengths", () => {
    const rows = DEFAULT_CATEGORIES.map((c) => (c.id === "other" ? { ...c, creativity: 0.8, threshold: 0.4 } : c));
    // New rows start from the Other category's session values.
    expect(newOtherFix(rows)).toMatchObject({ creativity: 0.8, threshold: 0.4 });
    const tags = setOthers(toggleTag(emptyTags(), "faces"), [
      { target: "newspaper", instruction: "", creativity: 0.6, threshold: 0.2 },
      { target: " ", instruction: "dropped", creativity: 1, threshold: 0.3 },
      { target: "watch", instruction: "Fix the dial.", creativity: 5, threshold: -0.5 },
      { target: "Newspaper", instruction: "repeat", creativity: 1, threshold: 0.3 },
    ]);
    const fixes = tagsToFixes(tags, rows);
    expect(fixes.map((f) => [f.id, f.target, f.creativity, f.threshold])).toEqual([
      ["faces", "yolo-face_yolov8m.pt", 1, 0.25],
      ["other", "newspaper", 0.6, 0.2],
      ["other", "watch", 1, -0.5],
    ]);
    // A stored fix list comes back with every row and its values.
    expect(tagsFromFixes(fixes).others).toEqual([
      { target: "newspaper", instruction: "", creativity: 0.6, threshold: 0.2 },
      { target: "watch", instruction: "Fix the dial.", creativity: 1, threshold: -0.5 },
    ]);
  });

  it("never emits a quick-mask fix (that goes through confirmMask)", () => {
    const tags = toggleTag(toggleTag(emptyTags(), "mask"), "faces");
    expect(tagsToFixes(tags, DEFAULT_CATEGORIES).map((f) => f.id)).toEqual(["faces"]);
    expect(DEFAULT_CATEGORIES.map((c) => c.key)).toEqual(["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]);
    expect(DEFAULT_CATEGORIES.map((c) => c.id)).toEqual([
      "faces",
      "hands",
      "feet",
      "limbs",
      "text",
      "background_people",
      "background",
      "other",
      "mask",
      "photo",
    ]);
    // The whole-image edit is an ordinary toggle with no segment target.
    const photo = tagsToFixes(toggleTag(emptyTags(), "photo"), DEFAULT_CATEGORIES);
    expect(photo).toMatchObject([{ id: "photo", target: "", instruction: INSTRUCTIONS.photo }]);
  });

  it("migrates a stored Hands/Feet row and the legacy face target", () => {
    const stored = DEFAULT_CATEGORIES.filter((c) => c.id !== "hands" && c.id !== "feet").map((c) =>
      c.id === "faces" ? { ...c, target: "face" } : c,
    );
    stored.push({ ...DEFAULT_CATEGORIES[1], id: "hands_feet", key: "2", target: "hands|feet", creativity: 0.7 });
    const rows = reconcileCategories(stored);
    expect(rows).toHaveLength(10);
    expect(rows.map((c) => c.id)).not.toContain("hands_feet");
    expect(rows.find((c) => c.id === "hands")).toMatchObject({ creativity: 0.7, target: "yolo-hand_yolov8s.pt" });
    expect(rows.find((c) => c.id === "feet")).toMatchObject({
      creativity: 0.7,
      target: "yolo-adetailerFootYolov8x_v20.pt",
    });
    expect(rows.find((c) => c.id === "faces")?.target).toBe("yolo-face_yolov8m.pt");
    // A custom face target is not touched.
    const custom = reconcileCategories([{ ...DEFAULT_CATEGORIES[0], target: "yolo-mine.pt" }]);
    expect(custom[0].target).toBe("yolo-mine.pt");
  });

  it("carries the detection ladder fields and restores them on old rows", () => {
    const byId = Object.fromEntries(DEFAULT_CATEGORIES.map((c) => [c.id, c]));
    expect(byId.faces).toMatchObject({ fallback_target: "face", low_threshold: 0.1 });
    expect(byId.hands.fallback_target).toBe("hands");
    expect(byId.feet.fallback_target).toBe("feet");
    expect(byId.background_people.fallback_target).toBe("people in the background");
    expect(byId.limbs.fallback_target).toBe("");
    // A row stored before the ladder existed gets the defaults; edits survive.
    const bare = { ...byId.faces } as Partial<(typeof DEFAULT_CATEGORIES)[number]>;
    delete bare.fallback_target;
    delete bare.low_threshold;
    const rows = reconcileCategories([
      bare as (typeof DEFAULT_CATEGORIES)[number],
      { ...byId.hands, fallback_target: "hand", low_threshold: 0.05 },
    ]);
    expect(rows.find((c) => c.id === "faces")).toMatchObject({ fallback_target: "face", low_threshold: 0.1 });
    expect(rows.find((c) => c.id === "hands")).toMatchObject({ fallback_target: "hand", low_threshold: 0.05 });
    const sent = tagsToFixes(toggleTag(emptyTags(), "faces"), rows);
    expect(sent[0].id).toBe("faces");
  });

  it("keeps the Background category's negative (inverted) threshold", () => {
    expect(DEFAULT_CATEGORIES).toHaveLength(10);
    const background = DEFAULT_CATEGORIES.find((c) => c.key === "7");
    expect(background).toMatchObject({ id: "background", threshold: -0.25, include_caption: true });
    const fixes = tagsToFixes(toggleTag(emptyTags(), "background"), DEFAULT_CATEGORIES);
    expect(fixes).toHaveLength(1);
    expect(fixes[0]).toMatchObject({ id: "background", target: "yolo-person_yolov8m-seg.pt-1", threshold: -0.25 });
    // Stored settings from before the row existed get it back with its defaults.
    const older = reconcileCategories(DEFAULT_CATEGORIES.filter((c) => c.id !== "background"));
    expect(older.map((c) => c.key)).toEqual(["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]);
    expect(older[6]).toMatchObject({ id: "background", threshold: -0.25 });
    // A stored threshold below -1 clamps, but stays negative.
    const stored = reconcileCategories([{ ...older[6], threshold: -3 }]);
    expect(stored[6].threshold).toBe(-1);
  });
});

describe("parseSegmentString", () => {
  it("reads targets with and without numbers", () => {
    expect(parseSegmentString("<segment:face,0.9,0.25> <segment:hand> <segment:newspaper,1,0.2>")).toEqual([
      { target: "face", creativity: 0.9, threshold: 0.25 },
      { target: "hand", creativity: 1, threshold: 0.3 },
      { target: "newspaper", creativity: 1, threshold: 0.2 },
    ]);
    expect(parseSegmentString("no tags <lora:x:1>")).toEqual([]);
  });

  it("maps detected targets onto the categories", () => {
    const tags = categoriesFromDetected(
      parseSegmentString("<segment:hand,1,0.3> <segment:face> <segment:newspaper>"),
      DEFAULT_CATEGORIES,
    );
    expect([...tags.ids].sort()).toEqual(["faces", "hands", "other"]);
    expect(tags.others).toEqual([{ target: "newspaper", instruction: "", creativity: 1, threshold: 0.3 }]);
  });
});

describe("round two helpers", () => {
  it("finds a stem's cursor in the triage order and keeps a capped history", () => {
    const items = [item("a"), item("b", "fixed"), item("c", "tagged"), item("d")];
    const order = triageOrder(items, 3);
    expect(cursorForStem(order, "c")).toBe(order.findIndex((i) => i.stem === "c"));
    expect(cursorForStem(order, "b")).toBe(-1);
    expect(cursorForStem(order, "zzz")).toBe(-1);
    let history: string[] = [];
    for (const stem of ["p1", "p2", "p1", "p3"]) history = pushRecent(history, stem);
    expect(history).toEqual(["p3", "p1", "p2"]);
    for (let i = 0; i < 20; i++) history = pushRecent(history, `q${i}`);
    expect(history).toHaveLength(10);
    expect(history[0]).toBe("q19");
  });
});

describe("ledgerCounts", () => {
  it("counts statuses and undecided", () => {
    const counts = ledgerCounts([
      item("a"),
      item("b", "tagged"),
      item("c", "ready"),
      item("d", "fixed"),
      item("e", "deleted"),
    ]);
    expect(counts).toMatchObject({ undecided: 2, tagged: 1, ready: 1, fixed: 1, deleted: 1, total: 5 });
  });
});

describe("loadTriageFixSettings", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("layers swarm, regen and its own key, and reconciles categories", () => {
    window.localStorage.setItem("ot.swarmGenerate", JSON.stringify({ baseUrl: "http://swarm:1", token: "s" }));
    window.localStorage.setItem("ot.dpoRegen", JSON.stringify({ baseUrl: "http://regen:2" }));
    let settings = loadTriageFixSettings();
    expect(settings.baseUrl).toBe("http://regen:2");
    expect(settings.token).toBe("s");
    expect(settings.steps).toBe(4);
    expect(settings.maskParams).toEqual({ maskshrinkgrow: 32, maskblur: 8 });
    expect(settings.categories.map((c) => c.id)).toEqual(DEFAULT_CATEGORIES.map((c) => c.id));

    window.localStorage.setItem(
      TRIAGE_FIX_STORAGE_KEY,
      JSON.stringify({
        baseUrl: "http://own:3",
        model: "Flux/klein",
        categories: [{ id: "faces", target: "yolo-face.pt", threshold: 0.4 }, { id: "bogus" }],
        maskParams: { maskshrinkgrow: 64 },
      }),
    );
    settings = loadTriageFixSettings();
    expect(settings.baseUrl).toBe("http://own:3");
    expect(settings.model).toBe("Flux/klein");
    expect(settings.categories).toHaveLength(DEFAULT_CATEGORIES.length);
    expect(settings.categories[0]).toMatchObject({ id: "faces", target: "yolo-face.pt", threshold: 0.4 });
    expect(settings.categories[0].instruction).toBe(INSTRUCTIONS.faces);
    expect(settings.maskParams).toEqual({ maskshrinkgrow: 64, maskblur: 8 });

    // Rows stored with the old default wording move onto the new one (faces also drop the
    // caption); wording the user typed stays.
    window.localStorage.setItem(
      TRIAGE_FIX_STORAGE_KEY,
      JSON.stringify({
        categories: [
          { id: "faces", instruction: LEGACY_INSTRUCTIONS.faces[0], include_caption: true },
          { id: "text", instruction: LEGACY_INSTRUCTIONS.text[1] },
          { id: "hands", instruction: "Give her five fingers.", include_caption: true },
        ],
      }),
    );
    settings = loadTriageFixSettings();
    expect(settings.categories[0]).toMatchObject({ instruction: INSTRUCTIONS.faces, include_caption: false });
    expect(settings.categories[1]).toMatchObject({ instruction: "Give her five fingers.", include_caption: true });
    expect(settings.categories.find((c) => c.id === "text")?.instruction).toBe(INSTRUCTIONS.text);
  });

  it("migrates the old Background People phrase but keeps a custom target", () => {
    const people = DEFAULT_CATEGORIES.find((c) => c.id === "background_people") ?? DEFAULT_CATEGORIES[0];
    expect(people.id).toBe("background_people");
    expect(people.target).toBe(BACKGROUND_PEOPLE_TARGET);
    const migrated = reconcileCategories([{ ...people, target: "person in the background" }]);
    expect(migrated.find((c) => c.id === "background_people")?.target).toBe(BACKGROUND_PEOPLE_TARGET);
    const previousUnion = "yolo-person_yolov8m-seg.pt-2|yolo-person_yolov8m-seg.pt-3|yolo-person_yolov8m-seg.pt-4";
    const fromUnion = reconcileCategories([{ ...people, target: previousUnion }]);
    expect(fromUnion.find((c) => c.id === "background_people")?.target).toBe(BACKGROUND_PEOPLE_TARGET);
    expect(BACKGROUND_PEOPLE_TARGET.split("|")).toHaveLength(7);
    const custom = reconcileCategories([{ ...people, target: "yolo-person_yolov8m-seg.pt-2" }]);
    expect(custom.find((c) => c.id === "background_people")?.target).toBe("yolo-person_yolov8m-seg.pt-2");
  });

  it("fills the segment sort order when a stored session never set it", () => {
    expect(loadTriageFixSettings().segment).toEqual({ segmentsortorder: "largest-smallest" });
    expect(migrateSegment(undefined)).toEqual({ segmentsortorder: "largest-smallest" });
    expect(migrateSegment({ segmentmaskblur: "12", segmentsortorder: "" })).toEqual({
      segmentmaskblur: "12",
      segmentsortorder: "largest-smallest",
    });
    expect(migrateSegment({ segmentsortorder: "left-right" })).toEqual({ segmentsortorder: "left-right" });
    window.localStorage.setItem(TRIAGE_FIX_STORAGE_KEY, JSON.stringify({ segment: { segmentmaskblur: "12" } }));
    expect(loadTriageFixSettings().segment).toEqual({ segmentmaskblur: "12", segmentsortorder: "largest-smallest" });
  });
});

describe("caption keyword filter", () => {
  const withCaption = (stem: string, caption: string) => ({ stem, caption }) as TriageItem;
  const items = [
    withCaption("a", "A wide shot of a woman on a beach"),
    withCaption("b", "Close-up portrait of a man"),
    withCaption("c", "Full  body photo of a dancer, wide shot"),
    withCaption("d", ""),
  ];

  it("keeps everything when the box is empty", () => {
    const filter = parseCaptionFilter("  , ,");
    expect(captionFilterActive(filter)).toBe(false);
    expect(filterByCaption(items, filter)).toBe(items);
  });

  it("matches any plain phrase, case-insensitively, across collapsed whitespace", () => {
    const stems = (query: string) => filterByCaption(items, parseCaptionFilter(query)).map((i) => i.stem);
    expect(stems("WIDE shot")).toEqual(["a", "c"]);
    expect(stems("full body, portrait")).toEqual(["b", "c"]);
  });

  it("drops captions containing an excluded phrase, alone or with includes", () => {
    const stems = (query: string) => filterByCaption(items, parseCaptionFilter(query)).map((i) => i.stem);
    expect(stems("-close-up")).toEqual(["a", "c", "d"]);
    expect(stems("wide shot, -dancer")).toEqual(["a"]);
    expect(parseCaptionFilter("wide, - beach ,-")).toEqual({ include: ["wide"], exclude: ["beach"] });
  });
});

describe("fixer LoRAs", () => {
  it("keeps named rows once, with a usable weight", () => {
    expect(
      cleanLoras([
        { name: " faces-v2 ", weight: 0.8 },
        { name: "FACES-V2", weight: 1 },
        { name: "", weight: 1 },
        { name: "style", weight: Number.NaN },
        null,
      ]),
    ).toEqual([
      { name: "faces-v2", weight: 0.8 },
      { name: "style", weight: 1 },
    ]);
    expect(cleanLoras(undefined)).toEqual([]);
    expect(loadTriageFixSettings().loras).toEqual([]);
  });
});

describe("second round", () => {
  const items: TriageItem[] = [
    item("a"),
    item("b", "fixed"),
    item("c", "skipped"),
    item("d", "kept"),
    { ...item("e", "fixed"), second_pass: "skipped" },
    { ...item("f", "tagged"), second_pass: "open", second_from: "fixed" },
    { ...item("g", "queued"), second_pass: "open", second_from: "skipped" },
    item("h", "ready"),
  ];

  it("walks decided images not yet visited plus the ones taken up and awaiting fixes", () => {
    expect(new Set(secondRoundOrder(items, 5).map((i) => i.stem))).toEqual(new Set(["b", "c", "d", "f"]));
  });

  it("keeps images taken up by the second round out of the first triage walk", () => {
    expect(triageOrder(items, 5).map((i) => i.stem)).toEqual(["a"]);
  });

  it("reports what an image was before the second round took it up", () => {
    expect(items.map(secondKind)).toEqual([null, "fixed", "skipped", "kept", "fixed", "fixed", "skipped", null]);
  });

  it("counts only decided images the round has handled as clearable", () => {
    expect(items.filter(secondRoundHandled).map((i) => i.stem)).toEqual(["e"]);
  });

  it("filters by caption or stem", () => {
    const captioned = [
      { ...item("pair_0007"), caption: "a hand" },
      { ...item("pair_0008"), caption: "a foot" },
    ];
    const stems = (q: string) => filterByCaptionOrStem(captioned, parseCaptionFilter(q)).map((i) => i.stem);
    expect(stems("pair_0007")).toEqual(["pair_0007"]);
    expect(stems("foot")).toEqual(["pair_0008"]);
    expect(stems("-hand")).toEqual(["pair_0008"]);
  });
});

describe("colour correction setting", () => {
  beforeEach(() => window.localStorage.clear());

  it("defaults to Linear2, keeps a valid stored choice and repairs an invalid one", () => {
    expect(loadTriageFixSettings().colorCorrection).toBe("Linear2");
    window.localStorage.setItem(TRIAGE_FIX_STORAGE_KEY, JSON.stringify({ colorCorrection: "None" }));
    expect(loadTriageFixSettings().colorCorrection).toBe("None");
    window.localStorage.setItem(TRIAGE_FIX_STORAGE_KEY, JSON.stringify({ colorCorrection: "linear9" }));
    expect(loadTriageFixSettings().colorCorrection).toBe("Linear2");
    expect(cleanColorCorrection("Uniform")).toBe("Uniform");
  });
});

describe("crop reference setting", () => {
  beforeEach(() => window.localStorage.clear());

  it("defaults to 1 and clamps what is stored", () => {
    expect(loadTriageFixSettings().cropReference).toBe(1);
    window.localStorage.setItem(TRIAGE_FIX_STORAGE_KEY, JSON.stringify({ cropReference: 0.4 }));
    expect(loadTriageFixSettings().cropReference).toBe(0.4);
    expect(cleanCropReference(3)).toBe(1);
    expect(cleanCropReference(-2)).toBe(0);
    expect(cleanCropReference("")).toBe(1);
    expect(cleanCropReference("abc")).toBe(1);
  });
});

describe("sessions", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("makes folder-safe, unique ids from tab names", () => {
    expect(sessionIdFor("Anatomy", ["main"])).toBe("anatomy");
    expect(sessionIdFor("Flux: everything else!", ["main"])).toBe("flux-everything-else");
    expect(sessionIdFor("Anatomy", ["main", "anatomy", "anatomy-2"])).toBe("anatomy-3");
    expect(sessionIdFor("  ", ["main"])).toBe("session");
    expect(sessionIdFor("Main", ["main"])).toBe("main-2");
  });

  it("keeps the main tab first and drops bad or repeated ids", () => {
    const tabs = cleanTriageSessions([
      { id: "flux", name: "Flux" },
      { id: "../x", name: "bad" },
      { id: "flux", name: "again" },
      { id: "main", name: "Main" },
      { id: "anatomy", name: " " },
    ]);
    expect(tabs).toEqual([
      { id: "main", name: "Main" },
      { id: "flux", name: "Flux" },
      { id: "anatomy", name: "anatomy" },
    ]);
    expect(loadTriageSessions()).toEqual([{ id: "main", name: "Main" }]);
  });

  it("stores settings per session, a new one starting from the main session's", () => {
    const main = loadTriageFixSettings();
    saveTriageFixSettings({ ...main, model: "Flux/klein", conceptDir: "D:/ds" });
    const fresh = loadTriageFixSettings("anatomy");
    expect(fresh.model).toBe("Flux/klein");
    expect(fresh.conceptDir).toBe("D:/ds");
    saveTriageFixSettings({ ...fresh, model: "Anatomy/fixer" }, "anatomy");
    expect(loadTriageFixSettings("anatomy").model).toBe("Anatomy/fixer");
    expect(loadTriageFixSettings().model).toBe("Flux/klein");
  });
});
