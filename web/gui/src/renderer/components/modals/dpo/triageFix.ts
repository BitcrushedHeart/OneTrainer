/**
 * Pure helpers behind the Triage & Fix modal: the category table and its
 * number keys, the seeded triage order, tag sets, the detector-string parser
 * and the persisted settings. No React, no network.
 */

import {
  MAIN_TRIAGE_SESSION,
  type TriageCategory,
  type TriageCounts,
  type TriageFixSpec,
  type TriageItem,
  type TriageStatus,
} from "@/api/dpoTriageFixApi";

export const CATEGORY_OTHER = "other";
export const CATEGORY_MASK = "mask";
/** Whole-image Klein edit: no segment, the instruction is the whole prompt. */
export const CATEGORY_PHOTO = "photo";
export const DEFAULT_MASK_PARAMS: Record<string, number> = { maskshrinkgrow: 32, maskblur: 8 };
export const FIX_CREATIVITY = 1.0;
export const FIX_THRESHOLD = 0.3;
export const DEFAULT_PROMPT_TEMPLATE = "{instruction} Here is the original prompt: {caption}";
/** Earlier Background People defaults, migrated onto the current one. */
export const LEGACY_BACKGROUND_PEOPLE_TARGETS = [
  "person in the background",
  "yolo-person_yolov8m-seg.pt-2|yolo-person_yolov8m-seg.pt-3|yolo-person_yolov8m-seg.pt-4",
];
/** Every person except the largest (the subject), by SwarmUI's yolo-...-N index, up to eight. */
export const BACKGROUND_PEOPLE_TARGET = [2, 3, 4, 5, 6, 7, 8].map((i) => `yolo-person_yolov8m-seg.pt-${i}`).join("|");
/** Indexed yolo targets count people largest first, so -1 is the subject. */
export const DEFAULT_SEGMENT_SORT_ORDER = "largest-smallest";
/** The largest person; a negative threshold inverts it into "everything else". */
export const BACKGROUND_TARGET = "yolo-person_yolov8m-seg.pt-1";

/** YOLO detectors SwarmUI loads from Models/yolov8 (dpo_triage_fix_util). */
export const FACE_TARGET = "yolo-face_yolov8m.pt";
export const HAND_TARGET = "yolo-hand_yolov8s.pt";
export const FOOT_TARGET = "yolo-adetailerFootYolov8x_v20.pt";
export const YOLO_THRESHOLD = 0.25;
/** Second rung of the detection ladder: the YOLO target retried at this threshold. */
export const LOW_THRESHOLD = 0.1;
/** Earlier Faces default (CLIPSeg), migrated onto the YOLO detector. */
export const LEGACY_FACE_TARGET = "face";
/** The pre-split Hands/Feet category; a stored row feeds both new ones. */
export const LEGACY_HANDS_FEET_ID = "hands_feet";

/**
 * What each category tells Klein (dpo_triage_fix_util.INSTRUCTIONS). Klein gets the crop as its
 * reference, so say what good looks like, then what stays, in plain words: "sharp", "crisp" or
 * "detailed" made it paint an over-processed super-HD patch onto realistic but soft photos.
 */
export const INSTRUCTIONS = {
  faces:
    "Fix the face so it looks natural. Keep the same person, expression and lighting, and match the image's focus and softness.",
  hands:
    "Fix the hands so they have natural anatomy with five fingers each. Keep the same pose and lighting, and match the image's focus and softness.",
  feet: "Fix the feet so they have natural anatomy with five toes each. Keep the same pose and lighting, and match the image's focus and softness.",
  limbs:
    "Fix the arms, legs and body so the anatomy and proportions are natural. Keep the same pose, clothing and lighting, and match the image's focus and softness.",
  text: "Fix the text so it is legible and correctly spelled. Keep the same words, style and lighting, and match the image's focus and softness.",
  background_people:
    "Fix the people in the background so they look natural. Keep the same positions and lighting, and match the image's focus and softness.",
  background:
    "Fix the background so it looks natural and coherent. Keep the same scene and lighting, and match the image's focus and softness.",
  other:
    "Fix the {target} so it looks natural. Keep the same position and lighting, and match the image's focus and softness.",
  mask: "Fix this area so it looks natural. Keep it consistent with the rest of the image, and match its focus and softness.",
  photo: "Make this look like a natural photo. Keep the same people, poses, composition and scene.",
} as const;
/** Every earlier default; stored rows still carrying one move onto the current wording. */
export const LEGACY_INSTRUCTIONS: Record<string, readonly string[]> = {
  faces: [
    "Fix the face.",
    "Restore this face with natural skin texture, sharp symmetrical eyes and clean detailed features. Keep the same person, expression, head pose, hair and lighting.",
  ],
  hands: [
    "Fix the hands.",
    "Restore these hands with correct anatomy: five well-formed fingers on each hand, natural knuckles, joints and fingernails, realistic skin. Keep the same pose, grip, skin tone and lighting, and anything being held.",
  ],
  feet: [
    "Fix the feet.",
    "Restore these feet with correct anatomy: five well-formed toes on each foot, natural proportions, arches and toenails, realistic skin. Keep the same pose, footwear, skin tone and lighting.",
  ],
  limbs: [
    "Fix the arms, legs and body proportions on this image.",
    "Restore the arms, legs and body with correct anatomy and natural proportions: the right number of limbs, well-formed joints, limbs that connect naturally to the body. Keep the same person, pose, clothing and lighting.",
  ],
  text: [
    "Fix the text on this image so it is legible and correctly spelled.",
    "Redraw the text so every letter is crisp, evenly spaced, legible and correctly spelled. Keep the same words, font style, colour, surface and lighting.",
  ],
  background_people: [
    "Fix the people in the background of this image.",
    "Restore the people in the background with well-formed faces, natural bodies and correct proportions, at a level of detail that suits their distance. Keep the same positions, poses, clothing, focus blur and lighting.",
  ],
  background: [
    "Fix the background of this image.",
    "Restore the background with coherent, realistic detail: clean straight architecture, natural foliage and objects that make physical sense. Keep the same scene, layout, colours, depth of field and lighting.",
  ],
  other: [
    "Fix the {target} on this image.",
    "Restore the {target} so it is well-formed, realistic and correctly detailed. Keep the same position, shape, colours and lighting.",
  ],
  mask: [
    "Fix this area of the image.",
    "Restore this area so it is well-formed, realistic and correctly detailed. Keep it consistent with the rest of the image: same colours, texture and lighting.",
  ],
  photo: [
    "Make this look more like a photo.",
    "Make this look like a real photograph: natural skin texture, realistic lighting, true-to-life colours and fine photographic detail. Keep the same people, poses, composition and scene.",
  ],
};

/** The backend's default table (dpo_triage_fix_util.TRIAGE_CATEGORIES), key order. */
export const DEFAULT_CATEGORIES: TriageCategory[] = [
  {
    id: "faces",
    key: "1",
    label: "Fix Faces",
    target: FACE_TARGET,
    instruction: INSTRUCTIONS.faces,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: YOLO_THRESHOLD,
    fallback_target: "face",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: "hands",
    key: "2",
    label: "Fix Hands",
    target: HAND_TARGET,
    instruction: INSTRUCTIONS.hands,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: YOLO_THRESHOLD,
    fallback_target: "hands",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: "feet",
    key: "3",
    label: "Fix Feet",
    target: FOOT_TARGET,
    instruction: INSTRUCTIONS.feet,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: YOLO_THRESHOLD,
    fallback_target: "feet",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: "limbs",
    key: "4",
    label: "Fix Limbs/Body",
    target: "arms|legs|torso",
    instruction: INSTRUCTIONS.limbs,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: FIX_THRESHOLD,
    fallback_target: "",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: "text",
    key: "5",
    label: "Fix Text",
    target: "text",
    instruction: INSTRUCTIONS.text,
    include_caption: true,
    creativity: FIX_CREATIVITY,
    threshold: FIX_THRESHOLD,
    fallback_target: "",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: "background_people",
    key: "6",
    label: "Fix Background People",
    target: BACKGROUND_PEOPLE_TARGET,
    instruction: INSTRUCTIONS.background_people,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: FIX_THRESHOLD,
    fallback_target: "people in the background",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: "background",
    key: "7",
    label: "Fix Background",
    target: BACKGROUND_TARGET,
    instruction: INSTRUCTIONS.background,
    include_caption: true,
    creativity: FIX_CREATIVITY,
    // Negative: SwarmUI inverts the mask, so this is everything except the subject.
    threshold: -0.25,
    fallback_target: "",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: CATEGORY_OTHER,
    key: "8",
    label: "Fix Other",
    target: "",
    instruction: INSTRUCTIONS.other,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: FIX_THRESHOLD,
    fallback_target: "",
    low_threshold: LOW_THRESHOLD,
  },
  {
    id: CATEGORY_MASK,
    key: "9",
    label: "Quick Mask",
    target: "",
    instruction: INSTRUCTIONS.mask,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: FIX_THRESHOLD,
    fallback_target: "",
    low_threshold: LOW_THRESHOLD,
  },
  {
    // 0 is free in triage; on the compare screen it still means "keep the original".
    id: CATEGORY_PHOTO,
    key: "0",
    label: "More Photo-like",
    target: "",
    instruction: INSTRUCTIONS.photo,
    include_caption: false,
    creativity: FIX_CREATIVITY,
    threshold: FIX_THRESHOLD,
    fallback_target: "",
    low_threshold: LOW_THRESHOLD,
  },
];

/** Stored categories brought back to the current table: ids fixed, edits kept. */
export function reconcileCategories(stored: TriageCategory[] | undefined): TriageCategory[] {
  // Unknown ids are dropped; the pre-split Hands/Feet row lends its
  // creativity to the two rows that replaced it.
  const legacyHandsFeet = stored?.find((row) => row.id === LEGACY_HANDS_FEET_ID);
  return DEFAULT_CATEGORIES.map((cat) => {
    let own = stored?.find((row) => row.id === cat.id);
    if (!own && legacyHandsFeet && (cat.id === "hands" || cat.id === "feet")) {
      own = { ...cat, creativity: legacyHandsFeet.creativity };
    }
    if (!own) return { ...cat };
    const untargeted = cat.id === CATEGORY_OTHER || cat.id === CATEGORY_MASK || cat.id === CATEGORY_PHOTO;
    let target = untargeted ? "" : String(own.target ?? cat.target);
    if (cat.id === "faces" && target.trim() === LEGACY_FACE_TARGET) target = FACE_TARGET;
    // The old CLIPSeg default follows the tool onto the YOLO union; anything
    // the user typed themselves stays.
    if (cat.id === "background_people" && LEGACY_BACKGROUND_PEOPLE_TARGETS.includes(target.trim())) {
      target = BACKGROUND_PEOPLE_TARGET;
    }
    // A row still on the old default wording takes the new one; with it, faces also drop the
    // caption (on a face crop it paints a miniature of the whole scene). Typed wording stays.
    const storedInstruction = String(own.instruction ?? cat.instruction);
    const legacyWording = (LEGACY_INSTRUCTIONS[cat.id] ?? []).includes(storedInstruction.trim());
    const ownCaption = typeof own.include_caption === "boolean" ? own.include_caption : cat.include_caption;
    return {
      ...cat,
      target,
      instruction: legacyWording ? cat.instruction : storedInstruction,
      include_caption: legacyWording && cat.id === "faces" ? false : ownCaption,
      creativity: clamp01(Number(own.creativity), cat.creativity),
      threshold: clampThreshold(Number(own.threshold), cat.threshold),
      // Ladder fields arrived later; rows stored before them take the defaults.
      fallback_target: typeof own.fallback_target === "string" ? own.fallback_target : cat.fallback_target,
      low_threshold: clamp01(Number(own.low_threshold), cat.low_threshold),
    };
  });
}

function clamp01(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(0, Math.min(1, value));
}

/** Thresholds run -1..1: a negative value asks SwarmUI to invert the mask. */
function clampThreshold(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(-1, Math.min(1, value));
}

// --------------------------------------------------------------------------- //
// Order
// --------------------------------------------------------------------------- //

/** A small, fast PRNG with a 32-bit seed (deterministic across sessions). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fisher–Yates on a copy; the same seed always gives the same order. */
export function seededShuffle<T>(items: T[], seed: number): T[] {
  const out = [...items];
  const random = mulberry32(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

const TRIAGE_STATUSES = new Set<TriageStatus | null>([null, "tagged"]);

/**
 * The triage walk: every stem shuffled by the session seed, then only the
 * ones still undecided. Decided items drop out without moving the rest, so
 * a resumed session continues in the same order.
 */
export function triageOrder(items: TriageItem[], seed: number): TriageItem[] {
  return shuffledBySeed(items, seed).filter((item) => TRIAGE_STATUSES.has(item.status) && item.second_pass !== "open");
}

function shuffledBySeed(items: TriageItem[], seed: number): TriageItem[] {
  const sorted = [...items].sort((a, b) => (a.stem < b.stem ? -1 : a.stem > b.stem ? 1 : 0));
  return seededShuffle(sorted, seed);
}

/** What the second round can take up again. */
export type SecondKind = "fixed" | "skipped" | "kept";
export const SECOND_KINDS: SecondKind[] = ["fixed", "skipped", "kept"];
const SECOND_STATUSES = new Set<TriageStatus | null>(SECOND_KINDS);

/** What an image counts as in the second round: its decision, or what it was before being taken up. */
export function secondKind(item: TriageItem): SecondKind | null {
  if (SECOND_STATUSES.has(item.status)) return item.status as SecondKind;
  if (item.second_pass === "open" && SECOND_STATUSES.has(item.second_from)) return item.second_from as SecondKind;
  return null;
}

/** Waiting in the second round: decided and not visited yet, or taken up and still awaiting its fixes. */
export function inSecondRoundWalk(item: TriageItem): boolean {
  if (SECOND_STATUSES.has(item.status)) return item.second_pass === null;
  return TRIAGE_STATUSES.has(item.status) && item.second_pass === "open";
}

/** Decided images the second round has handled (skipped there, or confirmed there and decided since). */
export function secondRoundHandled(item: TriageItem): boolean {
  return SECOND_STATUSES.has(item.status) && item.second_pass !== null;
}

/** The second-round walk, in the same seeded order as triage. */
export function secondRoundOrder(items: TriageItem[], seed: number): TriageItem[] {
  return shuffledBySeed(items, seed).filter(inSecondRoundWalk);
}

/** Like {@link filterByCaption}, but a phrase may also match the stem ("pair_07"). */
export function filterByCaptionOrStem(items: TriageItem[], filter: CaptionFilter): TriageItem[] {
  if (!captionFilterActive(filter)) return items;
  return items.filter((item) => {
    const text = `${item.stem} ${item.caption ?? ""}`.toLowerCase().replace(/\s+/g, " ");
    if (filter.exclude.some((phrase) => text.includes(phrase))) return false;
    return filter.include.length === 0 || filter.include.some((phrase) => text.includes(phrase));
  });
}

/** A caption keyword filter: phrases any of which must appear, and phrases none of which may. */
export interface CaptionFilter {
  include: string[];
  exclude: string[];
}

/**
 * Parse the triage keyword box. Commas separate phrases; a phrase matches when the caption contains
 * it (case-insensitive, runs of whitespace collapsed). Any plain phrase matching keeps the image
 * ("wide shot, full body"); a phrase starting with "-" drops it ("-close-up").
 */
export function parseCaptionFilter(query: string): CaptionFilter {
  const filter: CaptionFilter = { include: [], exclude: [] };
  for (const raw of query.split(",")) {
    const phrase = raw.trim().toLowerCase().replace(/\s+/g, " ");
    if (phrase.startsWith("-")) {
      const rest = phrase.slice(1).trim();
      if (rest) filter.exclude.push(rest);
    } else if (phrase) {
      filter.include.push(phrase);
    }
  }
  return filter;
}

export function captionFilterActive(filter: CaptionFilter): boolean {
  return filter.include.length > 0 || filter.exclude.length > 0;
}

/** The items whose caption passes the filter; every item when the filter is empty. */
export function filterByCaption(items: TriageItem[], filter: CaptionFilter): TriageItem[] {
  if (!captionFilterActive(filter)) return items;
  return items.filter((item) => {
    const caption = (item.caption ?? "").toLowerCase().replace(/\s+/g, " ");
    if (filter.exclude.some((phrase) => caption.includes(phrase))) return false;
    return filter.include.length === 0 || filter.include.some((phrase) => caption.includes(phrase));
  });
}

/** Where a stem sits in the triage walk, or -1 when it is not in it. */
export function cursorForStem(order: TriageItem[], stem: string): number {
  return order.findIndex((item) => item.stem === stem);
}

/** A "recently fixed" history: newest first, no duplicates, capped. */
export function pushRecent(history: string[], stem: string, limit = 10): string[] {
  return [stem, ...history.filter((s) => s !== stem)].slice(0, limit);
}

// --------------------------------------------------------------------------- //
// Tags
// --------------------------------------------------------------------------- //

/** One "Fix Other" selection: its own target, instruction and strengths. */
export interface OtherFix {
  target: string;
  instruction: string;
  creativity: number;
  threshold: number;
}

export interface TagSet {
  ids: Set<string>;
  /** The Other selections, in the order they were added. */
  others: OtherFix[];
}

export function emptyTags(): TagSet {
  return { ids: new Set(), others: [] };
}

export function toggleTag(tags: TagSet, id: string): TagSet {
  const ids = new Set(tags.ids);
  if (ids.has(id)) ids.delete(id);
  else ids.add(id);
  return { ...tags, ids };
}

/** A blank Other row carrying the Other category's session defaults. */
export function newOtherFix(categories: TriageCategory[], target = ""): OtherFix {
  const cat = categories.find((row) => row.id === CATEGORY_OTHER);
  return {
    target,
    instruction: "",
    creativity: clamp01(Number(cat?.creativity), FIX_CREATIVITY),
    threshold: clampThreshold(Number(cat?.threshold), FIX_THRESHOLD),
  };
}

/** Rows as they are sent: trimmed, clamped, blank targets and repeats dropped. */
export function cleanOthers(others: OtherFix[]): OtherFix[] {
  const seen = new Set<string>();
  const out: OtherFix[] = [];
  for (const row of others) {
    const target = row.target.trim();
    // The backend keeps one fix per target, so a repeat would be lost there anyway.
    if (!target || seen.has(target.toLowerCase())) continue;
    seen.add(target.toLowerCase());
    out.push({
      target,
      instruction: row.instruction.trim(),
      creativity: clamp01(row.creativity, FIX_CREATIVITY),
      threshold: clampThreshold(row.threshold, FIX_THRESHOLD),
    });
  }
  return out;
}

/** Set the Other selections; with none left the tag is cleared. */
export function setOthers(tags: TagSet, others: OtherFix[]): TagSet {
  const ids = new Set(tags.ids);
  const clean = cleanOthers(others);
  if (clean.length > 0) ids.add(CATEGORY_OTHER);
  else ids.delete(CATEGORY_OTHER);
  return { ids, others: clean };
}

/** The fixes a tag set sends, in category order. Each Other row needs a target. */
export function tagsToFixes(tags: TagSet, categories: TriageCategory[]): TriageFixSpec[] {
  const fixes: TriageFixSpec[] = [];
  for (const cat of categories) {
    if (!tags.ids.has(cat.id)) continue;
    // A quick mask travels through its own endpoint with the painted mask.
    if (cat.id === CATEGORY_MASK) continue;
    if (cat.id === CATEGORY_OTHER) {
      for (const other of cleanOthers(tags.others)) {
        fixes.push({
          id: cat.id,
          target: other.target,
          creativity: other.creativity,
          threshold: other.threshold,
          instruction: other.instruction,
          include_caption: cat.include_caption,
        });
      }
      continue;
    }
    fixes.push({
      id: cat.id,
      target: cat.target,
      creativity: clamp01(cat.creativity, FIX_CREATIVITY),
      threshold: clampThreshold(cat.threshold, FIX_THRESHOLD),
      instruction: cat.instruction,
      include_caption: cat.include_caption,
    });
  }
  return fixes;
}

/** The tag set a stored fix list (a queued item, a detector) corresponds to. */
export function tagsFromFixes(fixes: TriageFixSpec[]): TagSet {
  const tags = emptyTags();
  for (const fix of fixes) {
    // A quick mask is not a toggle; it is repainted per send.
    if (fix.id === CATEGORY_MASK) continue;
    // Older ledger records still carry the pre-split Hands/Feet id.
    if (fix.id === LEGACY_HANDS_FEET_ID) {
      tags.ids.add("hands");
      tags.ids.add("feet");
      continue;
    }
    tags.ids.add(fix.id);
    if (fix.id === CATEGORY_OTHER) {
      tags.others.push({
        target: fix.target,
        instruction: fix.instruction ?? "",
        creativity: clamp01(Number(fix.creativity), FIX_CREATIVITY),
        threshold: clampThreshold(Number(fix.threshold), FIX_THRESHOLD),
      });
    }
  }
  return tags;
}

// --------------------------------------------------------------------------- //
// Detector strings
// --------------------------------------------------------------------------- //

export interface ParsedSegment {
  target: string;
  creativity: number;
  threshold: number;
}

const SEGMENT_TAG_RE = /<segment:\s*([^,>]*?)\s*(?:,\s*([0-9.]*)\s*)?(?:,\s*([0-9.]*)\s*)?>/gi;

function cleanTarget(target: string): string {
  return target.replace(/[<>,]/g, " ").replace(/\s+/g, " ").trim();
}

function numberOr(text: string | undefined, fallback: number): number {
  if (text === undefined || text === "") return fallback;
  const value = Number(text);
  return Number.isFinite(value) ? value : fallback;
}

/** Every `<segment:target[,creativity[,threshold]]>` in the detector's output. */
export function parseSegmentString(text: string): ParsedSegment[] {
  const out: ParsedSegment[] = [];
  for (const match of (text ?? "").matchAll(SEGMENT_TAG_RE)) {
    const target = cleanTarget(match[1] ?? "");
    if (!target) continue;
    out.push({
      target,
      creativity: numberOr(match[2], FIX_CREATIVITY),
      threshold: numberOr(match[3], FIX_THRESHOLD),
    });
  }
  return out;
}

function tokens(target: string): Set<string> {
  const out = new Set<string>();
  for (const part of cleanTarget(target).toLowerCase().split("|")) {
    const word = part.trim();
    if (!word) continue;
    const words = word.includes(" ") ? [word, ...word.split(" ")] : [word];
    for (const w of words) {
      out.add(w);
      if (w.endsWith("s") && w.length > 3) out.add(w.slice(0, -1));
    }
  }
  return out;
}

/** Which keys a detector string toggles; unclaimed targets become Other. */
export function categoriesFromDetected(parsed: ParsedSegment[], categories: TriageCategory[]): TagSet {
  const tags = emptyTags();
  for (const segment of parsed) {
    const wanted = tokens(segment.target);
    let matched = CATEGORY_OTHER;
    for (const cat of categories) {
      if (cat.id === CATEGORY_OTHER || cat.id === CATEGORY_MASK || cat.id === CATEGORY_PHOTO) continue;
      // YOLO targets carry no words, so the category id ("hands") counts too.
      const own = new Set([...tokens(cat.target), ...tokens(cat.id.replace(/_/g, " "))]);
      if ([...wanted].some((w) => own.has(w)) || segment.target.toLowerCase() === cat.id) {
        matched = cat.id;
        break;
      }
    }
    tags.ids.add(matched);
    // Every unclaimed target becomes its own Other row, with the detector's strengths.
    if (
      matched === CATEGORY_OTHER &&
      !tags.others.some((o) => o.target.toLowerCase() === segment.target.toLowerCase())
    ) {
      tags.others.push({
        target: segment.target,
        instruction: "",
        creativity: clamp01(segment.creativity, FIX_CREATIVITY),
        threshold: clampThreshold(segment.threshold, FIX_THRESHOLD),
      });
    }
  }
  return tags;
}

// --------------------------------------------------------------------------- //
// Counts
// --------------------------------------------------------------------------- //

export function ledgerCounts(items: TriageItem[]): TriageCounts {
  const counts: TriageCounts = {
    tagged: 0,
    queued: 0,
    generating: 0,
    ready: 0,
    fixed: 0,
    kept: 0,
    skipped: 0,
    failed: 0,
    deleted: 0,
    undecided: 0,
    total: items.length,
  };
  for (const item of items) {
    if (TRIAGE_STATUSES.has(item.status)) counts.undecided += 1;
    if (item.status) counts[item.status] += 1;
  }
  return counts;
}

// --------------------------------------------------------------------------- //
// Settings
// --------------------------------------------------------------------------- //

export interface FixerLora {
  name: string;
  weight: number;
}

/** Stored LoRA rows made safe: named, no repeats, a finite weight (default 1). */
export function cleanLoras(raw: unknown): FixerLora[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: FixerLora[] = [];
  for (const row of raw as Array<Partial<FixerLora> | null>) {
    const name = String(row?.name ?? "").trim();
    if (!name || seen.has(name.toLowerCase())) continue;
    seen.add(name.toLowerCase());
    const weight = Number(row?.weight);
    out.push({ name, weight: Number.isFinite(weight) ? weight : 1 });
  }
  return out;
}

export interface TriageFixSettings {
  conceptDir: string;
  baseUrl: string;
  token: string;
  model: string;
  steps: number;
  cfgscale: number;
  sampler: string;
  scheduler: string;
  /** Blank = SwarmUI's default for the model class. */
  sigmashift: string;
  /** SwarmUI "Color Correction Behavior" for the pasted-back crop (Quick Mask and segment fixes). */
  colorCorrection: ColorCorrection;
  /** SwarmUI "Crop Reference Strength" (local patch, Flux.2 / Klein): 1 = the original crop as reference throughout. */
  cropReference: number;
  /** LoRAs / adapters applied on top of the fixer model (e.g. a finetune's own adapter). */
  loras: FixerLora[];
  optionsPerImage: number;
  concurrency: number;
  segment: Record<string, string | boolean>;
  /** Quick Mask knobs: maskshrinkgrow (context margin px), maskblur. */
  maskParams: Record<string, number>;
  categories: TriageCategory[];
  promptTemplate: string;
  fixPairDir: string;
  /** null = a fresh random seed on start. */
  shuffleSeed: number | null;
  /** Queue confirmed fixes without sending them to SwarmUI (it is busy elsewhere); release from the header. */
  holdSending: boolean;
}

/** SwarmUI's colour-correction modes for a masked / segment crop pasted back into the image. */
export const COLOR_CORRECTIONS = ["None", "Uniform", "Linear", "Linear2"] as const;
export type ColorCorrection = (typeof COLOR_CORRECTIONS)[number];

export function cleanColorCorrection(value: unknown): ColorCorrection {
  return COLOR_CORRECTIONS.find((choice) => choice === value) ?? "Linear2";
}

/** 0–1; anything unusable is the SwarmUI default of 1. */
export function cleanCropReference(value: unknown): number {
  const number = typeof value === "number" ? value : Number(value);
  if (value === "" || value === null || !Number.isFinite(number)) return 1;
  return Math.max(0, Math.min(1, number));
}

export const TRIAGE_FIX_STORAGE_KEY = "ot.triageFix";
/** The triage screen's caption keyword filter (view only). */
export const KEYWORD_FILTER_KEY = "ot.triageFix.keywordFilter";
const REGEN_KEY = "ot.dpoRegen";
const SWARM_KEY = "ot.swarmGenerate";

export function defaultTriageFixSettings(): TriageFixSettings {
  return {
    conceptDir: "",
    baseUrl: "http://localhost:7801",
    token: "",
    model: "",
    steps: 4,
    cfgscale: 1,
    sampler: "",
    scheduler: "",
    sigmashift: "",
    // Keeps the repaired crop's colours in step with the untouched image around it, so a DPO pair
    // differs only in the repair (SwarmUI recommends Linear2 for the Flux VAE's colour drift).
    colorCorrection: "Linear2",
    cropReference: 1,
    loras: [],
    optionsPerImage: 2,
    concurrency: 2,
    segment: { segmentsortorder: DEFAULT_SEGMENT_SORT_ORDER },
    maskParams: { ...DEFAULT_MASK_PARAMS },
    categories: DEFAULT_CATEGORIES.map((cat) => ({ ...cat })),
    promptTemplate: DEFAULT_PROMPT_TEMPLATE,
    fixPairDir: "",
    shuffleSeed: null,
    holdSending: false,
  };
}

/** Where a session's setup is stored: the main one keeps the original key. */
export function triageSettingsKey(session: string = MAIN_TRIAGE_SESSION): string {
  return session === MAIN_TRIAGE_SESSION ? TRIAGE_FIX_STORAGE_KEY : `${TRIAGE_FIX_STORAGE_KEY}.session.${session}`;
}

/** A session's stored setup. One never set up starts from the main session's
 *  (same dataset and SwarmUI; pick its own model). */
export function loadTriageFixSettings(session: string = MAIN_TRIAGE_SESSION): TriageFixSettings {
  const settings = defaultTriageFixSettings();
  try {
    const swarm = window.localStorage.getItem(SWARM_KEY);
    if (swarm) {
      const s = JSON.parse(swarm) as { baseUrl?: string; token?: string };
      if (s.baseUrl) settings.baseUrl = s.baseUrl;
      if (s.token) settings.token = s.token;
    }
    const regen = window.localStorage.getItem(REGEN_KEY);
    if (regen) {
      const r = JSON.parse(regen) as { baseUrl?: string; token?: string };
      if (r.baseUrl) settings.baseUrl = r.baseUrl;
      if (r.token) settings.token = r.token;
    }
    const own =
      window.localStorage.getItem(triageSettingsKey(session)) ?? window.localStorage.getItem(TRIAGE_FIX_STORAGE_KEY);
    if (own) {
      const parsed = JSON.parse(own) as Partial<TriageFixSettings>;
      Object.assign(settings, parsed);
      settings.categories = reconcileCategories(parsed.categories);
      settings.segment = migrateSegment(parsed.segment);
      settings.maskParams = { ...DEFAULT_MASK_PARAMS, ...(parsed.maskParams ?? {}) };
      settings.loras = cleanLoras(parsed.loras);
      settings.colorCorrection = cleanColorCorrection(parsed.colorCorrection);
      settings.cropReference = cleanCropReference(parsed.cropReference);
      settings.holdSending = parsed.holdSending === true;
      if (!settings.promptTemplate) settings.promptTemplate = DEFAULT_PROMPT_TEMPLATE;
    }
  } catch {
    /* storage unavailable — defaults are fine */
  }
  return settings;
}

/** Stored segment params, with the sort order filled in when it was never set. */
export function migrateSegment(stored: Record<string, string | boolean> | undefined): Record<string, string | boolean> {
  const segment = { ...(stored ?? {}) };
  const order = segment.segmentsortorder;
  if (order === undefined || order === "") segment.segmentsortorder = DEFAULT_SEGMENT_SORT_ORDER;
  return segment;
}

export function saveTriageFixSettings(settings: TriageFixSettings, session: string = MAIN_TRIAGE_SESSION): void {
  try {
    window.localStorage.setItem(triageSettingsKey(session), JSON.stringify(settings));
  } catch {
    /* storage unavailable */
  }
}

// ---- sessions ----------------------------------------------------------------

/** One tab of the Triage & Fix window: a session with its own model and queue. */
export interface TriageSessionTab {
  id: string;
  name: string;
}

export const TRIAGE_SESSIONS_KEY = "ot.triageFix.sessions";

export function defaultTriageSessions(): TriageSessionTab[] {
  return [{ id: MAIN_TRIAGE_SESSION, name: "Main" }];
}

/** The stored tabs, the main session always first. */
export function loadTriageSessions(): TriageSessionTab[] {
  try {
    const raw = window.localStorage.getItem(TRIAGE_SESSIONS_KEY);
    if (!raw) return defaultTriageSessions();
    return cleanTriageSessions(JSON.parse(raw));
  } catch {
    return defaultTriageSessions();
  }
}

export function saveTriageSessions(tabs: TriageSessionTab[]): void {
  try {
    window.localStorage.setItem(TRIAGE_SESSIONS_KEY, JSON.stringify(tabs));
  } catch {
    /* storage unavailable */
  }
}

/** Valid, unique tabs with the main session first. */
export function cleanTriageSessions(raw: unknown): TriageSessionTab[] {
  const out: TriageSessionTab[] = [];
  const seen = new Set<string>();
  for (const entry of Array.isArray(raw) ? raw : []) {
    const id = typeof entry?.id === "string" ? entry.id : "";
    if (!SESSION_ID_RE.test(id) || seen.has(id)) continue;
    seen.add(id);
    const name = typeof entry?.name === "string" && entry.name.trim() ? entry.name.trim() : id;
    out.push({ id, name });
  }
  const main = out.find((tab) => tab.id === MAIN_TRIAGE_SESSION) ?? defaultTriageSessions()[0];
  return [main, ...out.filter((tab) => tab.id !== MAIN_TRIAGE_SESSION)];
}

/** Same rule as the backend: the id is a folder name under .triage_fix/sessions. */
const SESSION_ID_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/;

/** A folder-safe id for a new tab called ``name``, unique among ``taken``. */
export function sessionIdFor(name: string, taken: Iterable<string>): string {
  const used = new Set(taken);
  const base =
    name
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^[-_]+|[-_]+$/g, "")
      .slice(0, 32) || "session";
  let id = base;
  for (let n = 2; used.has(id); n++) id = `${base}-${n}`;
  return id;
}

/** Segment-param form values as the backend takes them (blanks dropped). */
export function segmentParamsFromForm(
  form: Record<string, string | boolean>,
): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(form)) {
    if (value === "" || value === null || value === undefined || value === false) continue;
    out[key] = value;
  }
  return out;
}
