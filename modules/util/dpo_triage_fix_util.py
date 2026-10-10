"""Triage & Fix: repair the chosen images of an exported DPO dataset through
SwarmUI's segment inpainting and turn each accepted repair into training data.

Everything here is pure: file layout, the ledger that remembers what was
tagged, generated and decided, the SwarmUI request builders, the guard that
tells a real repair from a no-op, the in-place replacement of a chosen image
(with a backup, like Replace Pairs) and the optional fix pair written into a
second dataset. Threads and HTTP live in the service.

The request shape (verified against SwarmUI's source, 2026-09-16):

- Mode A, *segment inpaint*: the chosen file is sent as ``initimage`` with
  ``initimagecreativity`` 0 and a prompt made only of ``<segment:...>`` tags,
  each followed by its own sub-prompt: an instruction ("Fix the face.") plus
  the original caption. That is the shape the user verified with Klein
  (2026-09-16). SwarmUI skips the base sampler entirely,
  hands the original pixels to the segment stage, samples only the detected
  crops and pastes them back. The fixer model (Flux.2 Klein) is the request
  model, not the segment-model override, so the init encode and the segment
  pass share one VAE and everything outside the mask stays byte-exact.
- Mode B, *edit*: for an "Other" fix that segmentation could not find, the
  chosen file goes in as a prompt image and the instruction as the prompt.
  Klein edits the whole image; no mask, no local composite.
- Mode C, *quick mask*: the user paints the region. ``initimage`` +
  ``maskimage`` + full creativity + "inpaint only masked": SwarmUI crops to
  the mask (grown for context), samples it, and pastes it back onto the
  untouched original. No segmentation, no edit model, no fallback.
"""

import base64
import io
import json
import os
import re
import shutil
import time
import zlib
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Protocol

from modules.util.dpo_curation_util import (
    UNCONDITIONAL_PROMPT,
    compute_file_sha256,
    compute_pixel_hash,
    export_single_pair,
    find_exported_file,
    load_manifest,
    save_manifest,
)
from modules.util.dpo_offpolicy_util import image_size, is_within
from modules.util.dpo_regen_util import (
    SEGMENT_PRESETS,
    FixSpec,
    _as_float,
    _clean_target,
    compose_prompt,
    sanitize_segment_params,
)
from modules.util.dpo_replace_util import (
    REPLACED_DIR,
    _snapshot,
    _timestamp,
    _unique,
    pair_caption_path,
    read_pair_caption,
)
from modules.util.image_metadata_util import strip_angle_bracket_segments
from modules.util.swarm_generate_util import _PARAM_METADATA_KEYS

LEDGER_DIR = ".triage_fix"
CANDIDATES_DIR = "candidates"
LEDGER_FILE = "ledger.json"
LEDGER_VERSION = 1
#: Several sessions can run side by side, each with its own fixer model and
#: queue. The main one keeps the original layout; the others live under
#: ``.triage_fix/sessions/<id>/``.
MAIN_SESSION = "main"
SESSIONS_DIR = "sessions"
SESSION_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")

STATUS_TAGGED = "tagged"
STATUS_QUEUED = "queued"
STATUS_GENERATING = "generating"
STATUS_READY = "ready"
STATUS_FIXED = "fixed"
STATUS_KEPT = "kept"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_DELETED = "deleted"
ALL_STATUSES = (
    STATUS_TAGGED,
    STATUS_QUEUED,
    STATUS_GENERATING,
    STATUS_READY,
    STATUS_FIXED,
    STATUS_KEPT,
    STATUS_SKIPPED,
    STATUS_FAILED,
    STATUS_DELETED,
)
#: Statuses (``None`` = no ledger record yet) that still show in triage.
TRIAGE_STATUSES = frozenset({None, STATUS_TAGGED})
DECIDED_STATUSES = frozenset({STATUS_FIXED, STATUS_KEPT, STATUS_SKIPPED, STATUS_DELETED})
#: Decided images the second round can take up again.
SECOND_ROUND_STATUSES = frozenset({STATUS_FIXED, STATUS_SKIPPED, STATUS_KEPT})
#: ``record["second_pass"]["state"]``: taken up again (tagged, queued or
#: decided since), or passed over in the second round.
SECOND_OPEN = "open"
SECOND_SKIPPED = "skipped"
#: Where a deleted pair's files go: a dot folder the trainer and every
#: scanner ignore, so the delete is reversible.
DELETED_DIR = "deleted"

CATEGORY_OTHER = "other"
CATEGORY_MASK = "mask"
#: Whole-image Klein edit ("make this look more like a photo"): no segment,
#: the image goes in as a prompt image and the instruction is the prompt.
CATEGORY_PHOTO = "photo"
#: Categories that are not a ``<segment:...>`` tag.
UNSEGMENTED_CATEGORIES = (CATEGORY_MASK, CATEGORY_PHOTO)
MODE_SEGMENT = "segment"
MODE_EDIT = "edit"
MODE_MASK = "mask"
MASKS_DIR = "masks"
#: Fix-pair chosen images superseded by a later round, kept for undo.
FIX_PAIR_REPLACED_DIR = "fix_pair_replaced"

#: SwarmUI's masked-inpaint knobs for Quick Mask. ``maskshrinkgrow`` is the
#: context margin (px) around the mask's bounding box that is cropped and
#: sampled with it; ``maskblur`` softens the paste-back edge.
MASK_PARAM_SPECS: dict[str, tuple] = {
    "maskshrinkgrow": (int, 0, 512),
    "maskblur": (int, 0, 64),
    "maskgrow": (int, 0, 256),
}
DEFAULT_MASK_PARAMS = {"maskshrinkgrow": 32, "maskblur": 8}

#: SwarmUI's "Color Correction Behavior": how a repaired crop's colours are
#: matched to the image it is pasted back into (masked inpaint and segment
#: passes alike). "None" leaves them as the fixer made them.
COLOR_CORRECTIONS = ("None", "Uniform", "Linear", "Linear2")

#: "No change" is decided by how much of the image moved, not by how much
#: on average: a repaired hand is a few percent of the pixels and barely
#: shifts the global mean, while an empty mask that only cost a VAE round
#: trip nudges every pixel by a little. A pixel counts as changed when any
#: channel moves by more than ``NO_CHANGE_PIXEL_DELTA``; the candidate is
#: "no change" when fewer than ``NO_CHANGE_FRACTION`` of the pixels did.
NO_CHANGE_PIXEL_DELTA = 12
NO_CHANGE_FRACTION = 0.0005
#: Kept for the ledger's informational ``diff`` field (mean |delta|).
NO_CHANGE_MAD = 0.75

#: PNG text chunk the fixed image carries about its own repair.
OT_CHUNK = "ot_triage_fix"

#: Klein (distilled) is a 4-step, CFG-1 model. Sampler, scheduler and sigma
#: shift stay unset unless the user sets them, so SwarmUI applies its own
#: defaults for the model class.
DEFAULT_KLEIN = {
    "model": "",
    "steps": 4,
    "cfgscale": 1.0,
    "sampler": "",
    "scheduler": "",
    "sigmashift": None,
    "loras": [],
}
HISTORY_ACTION = "triage_fix"

#: The user's Klein calibration: full creativity; 0.3 is the CLIPSeg cutoff,
#: YOLO targets use their own confidence (SwarmUI's default 0.25).
FIX_CREATIVITY = 1.0
FIX_THRESHOLD = 0.3
YOLO_THRESHOLD = 0.25
#: Second rung of the detection ladder: the same YOLO model, far more
#: permissive, for the distant or half-hidden faces and hands it scored low.
LOW_THRESHOLD = 0.1
#: Rungs of the ladder a first candidate climbs while the image stays
#: unchanged: as tagged, YOLO at the low threshold, CLIPSeg fallback word.
RUNG_TAGGED = 0
RUNG_LOW = 1
RUNG_CLIPSEG = 2
RUNG_LABELS = {RUNG_TAGGED: "", RUNG_LOW: "yolo low", RUNG_CLIPSEG: "clipseg"}
#: Detection models in SwarmUI's Models/yolov8 folder (adetailer set). One
#: model per body part, so each segment gets an instruction about what it
#: actually sees: a hand crop is told to fix hands, not "hands and feet".
FACE_MODEL = "face_yolov8m.pt"
HAND_MODEL = "hand_yolov8s.pt"
FOOT_MODEL = "adetailerFootYolov8x_v20.pt"
#: How a segment's sub-prompt is assembled from the category instruction and
#: the pair caption. Editable per session.
DEFAULT_PROMPT_TEMPLATE = "{instruction} Here is the original prompt: {caption}"

#: The adetailer person segmentation model (Bingsu/adetailer) in SwarmUI's
#: Models/yolov8 folder, indexed 2..8 under a largest-first sort order.
BACKGROUND_PEOPLE_MODEL = "person_yolov8m-seg.pt"
BACKGROUND_PEOPLE_TARGET = "|".join(f"yolo-{BACKGROUND_PEOPLE_MODEL}-{i}" for i in range(2, 9))
#: The subject itself, inverted: a negative threshold flips the mask.
BACKGROUND_TARGET = f"yolo-{BACKGROUND_PEOPLE_MODEL}-1"
BACKGROUND_THRESHOLD = -0.25
#: Segment parameters the tool applies unless the session overrides them.
#: The sort order only affects indexed (yolo-...-N) targets.
DEFAULT_SEGMENT_PARAMS = {"segmentsortorder": "largest-smallest"}

#: Words a detector may use for each category, so its tags map onto the
#: number keys whatever the category's segment target happens to be.
CATEGORY_ALIASES = {
    "faces": {"face", "faces", "head"},
    "hands": {"hand", "hands", "finger", "fingers"},
    "feet": {"feet", "foot", "toes", "toe"},
    "limbs": {"arm", "arms", "leg", "legs", "limb", "limbs", "torso", "body"},
    "text": {"text", "sign", "writing", "letters", "words"},
    "background_people": {"person", "people", "crowd", "bystander", "bystanders", "background people"},
    "background": {"background", "backdrop", "scenery", "environment"},
}


# ---------------------------------------------------------------------------
# Categories and fixes
# ---------------------------------------------------------------------------


#: What each category tells Klein. Inside a segment or mask pass Klein gets
#: the crop as its reference latent, so a bare "fix it" comes back almost
#: unchanged: say what good looks like, then what stays. Keep it plain. Words
#: like "sharp", "crisp" or "detailed" made Klein paint an over-processed,
#: super-HD patch onto photos that are realistic but soft (2026-09-19), hence
#: the closing clause about the image's own focus and softness.
INSTRUCTIONS = {
    "faces": (
        "Fix the face so it looks natural. Keep the same person, expression and lighting, and match the image's focus and softness."
    ),
    "hands": (
        "Fix the hands so they have natural anatomy with five fingers each. Keep the same pose and lighting, and match the image's focus and softness."
    ),
    "feet": (
        "Fix the feet so they have natural anatomy with five toes each. Keep the same pose and lighting, and match the image's focus and softness."
    ),
    "limbs": (
        "Fix the arms, legs and body so the anatomy and proportions are natural. Keep the same pose, clothing and lighting, and match the image's focus and softness."
    ),
    "text": (
        "Fix the text so it is legible and correctly spelled. Keep the same words, style and lighting, and match the image's focus and softness."
    ),
    "background_people": (
        "Fix the people in the background so they look natural. Keep the same positions and lighting, and match the image's focus and softness."
    ),
    "background": (
        "Fix the background so it looks natural and coherent. Keep the same scene and lighting, and match the image's focus and softness."
    ),
    "other": (
        "Fix the {target} so it looks natural. Keep the same position and lighting, and match the image's focus and softness."
    ),
    "mask": (
        "Fix this area so it looks natural. Keep it consistent with the rest of the image, and match its focus and softness."
    ),
    "photo": ("Make this look like a natural photo. Keep the same people, poses, composition and scene."),
}
#: Every earlier default, so stored sessions and ledgers move onto the current
#: wording while anything the user typed themselves stays.
LEGACY_INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "faces": (
        "Fix the face.",
        "Restore this face with natural skin texture, sharp symmetrical eyes and clean detailed features. Keep the same person, expression, head pose, hair and lighting.",
    ),
    "hands": (
        "Fix the hands.",
        "Restore these hands with correct anatomy: five well-formed fingers on each hand, natural knuckles, joints and fingernails, realistic skin. Keep the same pose, grip, skin tone and lighting, and anything being held.",
    ),
    "feet": (
        "Fix the feet.",
        "Restore these feet with correct anatomy: five well-formed toes on each foot, natural proportions, arches and toenails, realistic skin. Keep the same pose, footwear, skin tone and lighting.",
    ),
    "limbs": (
        "Fix the arms, legs and body proportions on this image.",
        "Restore the arms, legs and body with correct anatomy and natural proportions: the right number of limbs, well-formed joints, limbs that connect naturally to the body. Keep the same person, pose, clothing and lighting.",
    ),
    "text": (
        "Fix the text on this image so it is legible and correctly spelled.",
        "Redraw the text so every letter is crisp, evenly spaced, legible and correctly spelled. Keep the same words, font style, colour, surface and lighting.",
    ),
    "background_people": (
        "Fix the people in the background of this image.",
        "Restore the people in the background with well-formed faces, natural bodies and correct proportions, at a level of detail that suits their distance. Keep the same positions, poses, clothing, focus blur and lighting.",
    ),
    "background": (
        "Fix the background of this image.",
        "Restore the background with coherent, realistic detail: clean straight architecture, natural foliage and objects that make physical sense. Keep the same scene, layout, colours, depth of field and lighting.",
    ),
    "other": (
        "Fix the {target} on this image.",
        "Restore the {target} so it is well-formed, realistic and correctly detailed. Keep the same position, shape, colours and lighting.",
    ),
    "mask": (
        "Fix this area of the image.",
        "Restore this area so it is well-formed, realistic and correctly detailed. Keep it consistent with the rest of the image: same colours, texture and lighting.",
    ),
    "photo": (
        "Make this look more like a photo.",
        "Make this look like a real photograph: natural skin texture, realistic lighting, true-to-life colours and fine photographic detail. Keep the same people, poses, composition and scene.",
    ),
}


def _legacy_patterns(category_id: str) -> list:
    import re

    return [
        re.compile("^" + re.escape(text).replace(re.escape("{target}"), "(.+)") + "$")
        for text in LEGACY_INSTRUCTIONS.get(category_id, ())
    ]


def target_noun(target: str) -> str:
    """What an instruction calls a segment target. A CLIPSeg target already
    is the noun ("newspaper"); a YOLO target is a file name, so
    ``yolo-adetailerFootYolov8x_v20.pt-2`` reads as "foot" and a class filter
    (``yolo-model.pt:dog``) wins. Falls back to "area"."""
    import re

    text = (target or "").strip()
    if not text.lower().startswith("yolo-"):
        return text
    name = text[5:].split("|")[0]
    name, _, class_filter = name.partition(":")
    if class_filter.strip():
        return class_filter.split(",")[0].strip().lower() or "area"
    name = re.sub(r"-\d+$", "", name.strip())
    name = re.sub(r"\.(pt|pth|onnx|safetensors)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    words = []
    for word in re.split(r"[\s_\-.]+", name.lower()):
        if not word or word in ("seg", "adetailer", "detection", "detector", "model"):
            continue
        if re.fullmatch(r"yolo(v?\d+)?[a-z]?|v?\d+(\.\d+)*|\d+[a-z]?", word):
            continue
        words.append(re.sub(r"yolov?\d+[a-z]?$", "", word) or word)
    return " ".join(w for w in words if w) or "area"


def migrate_instruction(category_id: str, instruction: str) -> str:
    """``instruction`` unless it is one of that category's old defaults. An
    "Other" fix stores its wording with the target filled in, so that form is
    recognised too and keeps its target."""
    text = (instruction or "").strip()
    if not text or category_id not in INSTRUCTIONS:
        return text
    for pattern in _legacy_patterns(category_id):
        match = pattern.match(text)
        if match:
            current = INSTRUCTIONS[category_id]
            return current.replace("{target}", match.group(1)) if match.groups() else current
    return text


def migrate_ledger_instructions(ledger: dict) -> int:
    """Move the undecided fixes and the stored category table of ``ledger``
    onto the current default wording. Returns how many instructions changed.
    Decided items are history and stay as they were sent."""
    pending = (STATUS_TAGGED, STATUS_QUEUED, STATUS_GENERATING, STATUS_READY, STATUS_FAILED)
    changed = 0
    for record in ledger.get("items", {}).values():
        if not isinstance(record, dict) or record.get("status") not in pending:
            continue
        for fix in record.get("fixes") or []:
            new = migrate_instruction(str(fix.get("id") or ""), str(fix.get("instruction") or ""))
            if new != str(fix.get("instruction") or "").strip():
                fix["instruction"] = new
                changed += 1
    for cat in (ledger.get("settings") or {}).get("categories") or []:
        if isinstance(cat, dict):
            cat["instruction"] = migrate_instruction(str(cat.get("id") or ""), str(cat.get("instruction") or ""))
    return changed


@dataclass(frozen=True)
class TriageCategory:
    id: str
    key: str
    label: str
    target: str
    instruction: str
    threshold: float = FIX_THRESHOLD
    #: Whether the segment's sub-prompt carries the pair caption after the
    #: instruction. SwarmUI scales every crop up to the model's full canvas,
    #: so a small hand, foot or face given a scene caption becomes a tiny
    #: scene (A/B 2026-09-18, Klein with the crop as its reference latent:
    #: the caption painted a miniature of the subject over the face, the
    #: bare instruction repaired it in place). Only text (what it should
    #: say) and the background (the scene itself) keep it.
    include_caption: bool = True
    #: When a YOLO target finds nothing the first time, the ladder retries the
    #: detector at ``low_threshold`` and then this CLIPSeg word. Blank = no
    #: CLIPSeg rung for the category.
    fallback_target: str = ""
    low_threshold: float = LOW_THRESHOLD


def _preset_target(preset_id: str) -> str:
    for preset in SEGMENT_PRESETS:
        if preset.id == preset_id:
            return preset.target
    return ""


def _preset_label(preset_id: str, fallback: str) -> str:
    for preset in SEGMENT_PRESETS:
        if preset.id == preset_id:
            return preset.label
    return fallback


#: Number-key order in the triage screen. Targets come from the shared
#: presets so the two tools stay in step; "other" is typed per image.
TRIAGE_CATEGORIES: tuple[TriageCategory, ...] = (
    TriageCategory(
        "faces",
        "1",
        "Fix Faces",
        f"yolo-{FACE_MODEL}",
        INSTRUCTIONS["faces"],
        YOLO_THRESHOLD,
        fallback_target="face",
        include_caption=False,
    ),
    TriageCategory(
        "hands",
        "2",
        "Fix Hands",
        f"yolo-{HAND_MODEL}",
        INSTRUCTIONS["hands"],
        YOLO_THRESHOLD,
        fallback_target="hands",
        include_caption=False,
    ),
    TriageCategory(
        "feet",
        "3",
        "Fix Feet",
        f"yolo-{FOOT_MODEL}",
        INSTRUCTIONS["feet"],
        YOLO_THRESHOLD,
        fallback_target="feet",
        include_caption=False,
    ),
    TriageCategory(
        "limbs",
        "4",
        _preset_label("limbs", "Fix Limbs/Body"),
        _preset_target("limbs"),
        INSTRUCTIONS["limbs"],
        include_caption=False,
    ),
    TriageCategory(
        "text",
        "5",
        _preset_label("text", "Fix Text"),
        _preset_target("text"),
        INSTRUCTIONS["text"],
    ),
    # CLIPSeg cannot express "in the background"; a YOLO person model can:
    # sorted largest-first, match 1 is the subject and 2..8 are everyone else.
    TriageCategory(
        "background_people",
        "6",
        _preset_label("background_people", "Fix Background People"),
        BACKGROUND_PEOPLE_TARGET,
        INSTRUCTIONS["background_people"],
        fallback_target="people in the background",
        include_caption=False,
    ),
    # Everything except the subject: the largest person, mask inverted by the
    # negative threshold. The caption describes the scene, so it goes along.
    TriageCategory(
        "background",
        "7",
        "Fix Background",
        BACKGROUND_TARGET,
        INSTRUCTIONS["background"],
        BACKGROUND_THRESHOLD,
    ),
    TriageCategory(CATEGORY_OTHER, "8", "Fix Other", "", INSTRUCTIONS[CATEGORY_OTHER], include_caption=False),
    TriageCategory(CATEGORY_MASK, "9", "Quick Mask", "", INSTRUCTIONS[CATEGORY_MASK], include_caption=False),
    # Klein used as an edit model on the whole frame. 0 is free in triage (it
    # only means "keep the original" on the compare screen).
    TriageCategory(CATEGORY_PHOTO, "0", "More Photo-like", "", INSTRUCTIONS[CATEGORY_PHOTO], include_caption=False),
)


def default_categories() -> list[dict]:
    return [
        {
            "id": cat.id,
            "key": cat.key,
            "label": cat.label,
            "target": cat.target,
            "instruction": cat.instruction,
            "include_caption": cat.include_caption,
            "creativity": FIX_CREATIVITY,
            "threshold": cat.threshold,
            "fallback_target": cat.fallback_target,
            "low_threshold": cat.low_threshold,
        }
        for cat in TRIAGE_CATEGORIES
    ]


def normalize_categories(raw) -> list[dict]:
    """The session's category table: the defaults, with the request's target,
    creativity and threshold applied by id. Unknown ids are ignored so the
    number keys always mean the same thing."""
    table = {cat["id"]: cat for cat in default_categories()}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        cat = table.get(str(item.get("id") or ""))
        if cat is None:
            continue
        if cat["id"] not in (CATEGORY_OTHER, *UNSEGMENTED_CATEGORIES) and str(item.get("target") or "").strip():
            cat["target"] = _clean_target(str(item["target"]))
        if str(item.get("instruction") or "").strip():
            cat["instruction"] = migrate_instruction(cat["id"], str(item["instruction"]))
        if "include_caption" in item:
            cat["include_caption"] = bool(item["include_caption"])
        cat["creativity"] = _clamp01(_as_float(item.get("creativity"), cat["creativity"]))
        cat["threshold"] = _clamp_threshold(_as_float(item.get("threshold"), cat["threshold"]))
        if "fallback_target" in item:
            cat["fallback_target"] = _clean_target(str(item.get("fallback_target") or ""))
        cat["low_threshold"] = _clamp01(_as_float(item.get("low_threshold"), cat["low_threshold"]))
    return list(table.values())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_threshold(value: float) -> float:
    # Negative = inverted mask (SwarmUI), so keep the sign.
    return max(-1.0, min(1.0, float(value)))


@dataclass
class TriageFix:
    """One fix on one image: a category plus what SwarmUI needs for its tag.
    ``instruction`` leads the segment's sub-prompt in Mode A and is the whole
    prompt in Mode B; blank means the category's default."""

    id: str
    target: str
    creativity: float = FIX_CREATIVITY
    threshold: float = FIX_THRESHOLD
    instruction: str = ""
    include_caption: bool = True
    fallback_target: str = ""
    low_threshold: float = LOW_THRESHOLD

    @classmethod
    def from_dict(cls, raw: dict) -> "TriageFix":
        return cls(
            id=str(raw.get("id") or CATEGORY_OTHER),
            target=_clean_target(str(raw.get("target") or "")),
            creativity=_clamp01(_as_float(raw.get("creativity"), FIX_CREATIVITY)),
            threshold=_clamp_threshold(_as_float(raw.get("threshold"), FIX_THRESHOLD)),
            instruction=str(raw.get("instruction") or "").strip(),
            include_caption=bool(raw.get("include_caption", True)),
            fallback_target=_clean_target(str(raw.get("fallback_target") or "")),
            low_threshold=_clamp01(_as_float(raw.get("low_threshold"), LOW_THRESHOLD)),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": self.target,
            "creativity": self.creativity,
            "threshold": self.threshold,
            "instruction": self.instruction,
            "include_caption": self.include_caption,
            "fallback_target": self.fallback_target,
            "low_threshold": self.low_threshold,
        }

    @property
    def is_yolo(self) -> bool:
        return self.target.lower().startswith("yolo-")

    def at_rung(self, rung: int) -> "TriageFix":
        """This fix as the ladder would send it at ``rung``. Only YOLO targets
        climb; an inverted mask (negative threshold) and CLIPSeg targets are
        returned as they are, and a category without a fallback word stops
        at the low-threshold rung."""
        if not self.is_yolo or self.threshold < 0 or rung == RUNG_TAGGED:
            return self
        if rung == RUNG_LOW:
            return replace(self, threshold=min(self.threshold, self.low_threshold))
        if rung == RUNG_CLIPSEG and self.fallback_target:
            return replace(self, target=self.fallback_target, threshold=FIX_THRESHOLD)
        # No CLIPSeg word: the top of this fix's ladder is the low rung.
        return self.at_rung(RUNG_LOW)

    def spec(self) -> FixSpec:
        return FixSpec(target=self.target, creativity=self.creativity, threshold=self.threshold)


def ladder_rungs(fixes: list[TriageFix]) -> list[int]:
    """The rungs worth trying for this set of fixes, in order: each one must
    change at least one fix's tag compared with the rung before."""
    rungs = [RUNG_TAGGED]
    previous = [fix.spec() for fix in fixes]
    for rung in (RUNG_LOW, RUNG_CLIPSEG):
        current = [fix.at_rung(rung).spec() for fix in fixes]
        if current != previous:
            rungs.append(rung)
            previous = current
    return rungs


def normalize_fixes(raw: list, categories: list[dict] | None = None) -> list[TriageFix]:
    """Validated fixes for one image. A category fix with a blank target takes
    the session category's target; "Other" must bring its own. Raises
    ``ValueError`` on nothing usable."""
    table = {cat["id"]: cat for cat in (categories or default_categories())}
    fixes: list[TriageFix] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        fix = TriageFix.from_dict(item)
        cat = table.get(fix.id)
        if cat is None:
            raise ValueError(f"Unknown fix category: {fix.id}")
        if not fix.target:
            if fix.id == CATEGORY_OTHER:
                raise ValueError("The Other fix needs a target (what to segment)")
            # A quick mask or a whole-image edit has no segment target.
            fix.target = fix.id if fix.id in UNSEGMENTED_CATEGORIES else cat["target"]
        if "creativity" not in item:
            fix.creativity = cat["creativity"]
        if "threshold" not in item:
            fix.threshold = cat["threshold"]
        if not fix.instruction:
            fix.instruction = str(cat.get("instruction") or "").replace("{target}", target_noun(fix.target)).strip()
        if "include_caption" not in item:
            fix.include_caption = bool(cat.get("include_caption", True))
        if "fallback_target" not in item:
            fix.fallback_target = _clean_target(str(cat.get("fallback_target") or ""))
        if "low_threshold" not in item:
            fix.low_threshold = _clamp01(_as_float(cat.get("low_threshold"), LOW_THRESHOLD))
        key = fix.target.lower()
        if key in seen:
            continue
        seen.add(key)
        # The tag composer rejects targets that reduce to nothing.
        compose_prompt("", [fix.spec()])
        fixes.append(fix)
    if not fixes:
        raise ValueError("Pick at least one fix")
    return fixes


def _tokens(target: str) -> set[str]:
    out = set()
    for part in _clean_target(target).lower().split("|"):
        part = part.strip()
        if not part:
            continue
        words = [part, *part.split()] if " " in part else [part]
        for word in words:
            out.add(word)
            if word.endswith("s") and len(word) > 3:
                out.add(word[:-1])
    return out


def categorize(fixes: list[FixSpec], categories: list[dict]) -> list[TriageFix]:
    """Map detector tags onto the session categories by target word, so a
    detected ``hand`` toggles the Hands/Feet key. A target no category
    claims becomes an "Other" fix with that target. The detector's own
    creativity and threshold are kept: they are its calibration."""
    out: list[TriageFix] = []
    seen: set[str] = set()
    for fix in fixes:
        wanted = _tokens(fix.target)
        matched = CATEGORY_OTHER
        for cat in categories:
            if cat["id"] in (CATEGORY_OTHER, *UNSEGMENTED_CATEGORIES):
                continue
            aliases = CATEGORY_ALIASES.get(cat["id"], set())
            if wanted & (_tokens(cat["target"]) | aliases) or fix.target.lower() == cat["id"]:
                matched = cat["id"]
                break
        key = f"{matched}:{fix.target.lower()}"
        if key in seen:
            continue
        seen.add(key)
        cat = next((c for c in categories if c["id"] == matched), None)
        instruction = str((cat or {}).get("instruction") or "").replace("{target}", fix.target).strip()
        out.append(
            TriageFix(
                id=matched,
                target=fix.target,
                creativity=fix.creativity,
                threshold=fix.threshold,
                instruction=instruction,
                include_caption=bool((cat or {}).get("include_caption", True)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Detector interface (the model is still in training; only the contract exists)
# ---------------------------------------------------------------------------


class FixDetector(Protocol):
    def detect(self, image_path: str) -> str:
        """A string of ``<segment:...>`` tags for the image, "" when clean."""


def load_detector(spec: str) -> FixDetector | None:
    """``None`` for a blank spec. Anything else is refused until the detector
    exists: the ledger, triage and confirm-all-detected already understand
    its output, so wiring it is a matter of returning an object here."""
    if not (spec or "").strip():
        return None
    raise NotImplementedError("The fix detector is not wired yet")


# ---------------------------------------------------------------------------
# Layout and ledger
# ---------------------------------------------------------------------------


def ledger_dir(concept_dir: str) -> str:
    return os.path.join(concept_dir, LEDGER_DIR)


def valid_session_id(session: str) -> bool:
    return bool(SESSION_ID_RE.fullmatch(session or ""))


def session_dir(concept_dir: str, session: str = MAIN_SESSION) -> str:
    """Where one session keeps its ledger, candidates and masks. The main
    session uses the folder every ledger lived in before there were several,
    so an existing run resumes as it is."""
    if session == MAIN_SESSION:
        return ledger_dir(concept_dir)
    if not valid_session_id(session):
        raise ValueError(f"Invalid session id: {session!r}")
    return os.path.join(ledger_dir(concept_dir), SESSIONS_DIR, session)


def ledger_path(concept_dir: str, session: str = MAIN_SESSION) -> str:
    return os.path.join(session_dir(concept_dir, session), LEDGER_FILE)


def candidates_dir(concept_dir: str, session: str = MAIN_SESSION) -> str:
    return os.path.join(session_dir(concept_dir, session), CANDIDATES_DIR)


def candidate_path(concept_dir: str, stem: str, k: int, session: str = MAIN_SESSION) -> str:
    return os.path.join(candidates_dir(concept_dir, session), f"{stem}__v{int(k)}.png")


def mask_path(concept_dir: str, stem: str, session: str = MAIN_SESSION) -> str:
    return os.path.join(session_dir(concept_dir, session), MASKS_DIR, f"{stem}.png")


def deleted_dir(concept_dir: str) -> str:
    return os.path.join(ledger_dir(concept_dir), DELETED_DIR)


def save_mask(data: str | bytes, target_path: str, size: tuple[int, int]) -> None:
    """Store a painted mask as a binary PNG at the image's own size: white =
    repaint, black = keep. Accepts raw bytes or a ``data:`` URI. Raises
    ``ValueError`` when nothing was painted."""
    from PIL import Image

    if isinstance(data, str):
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        data = base64.b64decode(data)
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        # A painter that exports RGBA uses alpha as the brush; anything else
        # is read as luminance.
        mask = img.getchannel("A") if img.mode == "RGBA" else img.convert("L")
        if mask.size != tuple(size):
            mask = mask.resize(tuple(size), Image.NEAREST)
        mask = mask.point(lambda v: 255 if v > 127 else 0)
        if not mask.getbbox():
            raise ValueError("The mask is empty; paint the area to repair")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        mask.save(target_path, format="PNG")


def sanitize_mask_params(raw: dict | None) -> dict:
    """The masked-inpaint knobs with defaults applied, clamped."""
    out = dict(DEFAULT_MASK_PARAMS)
    for key, value in (raw or {}).items():
        spec = MASK_PARAM_SPECS.get(key)
        if spec is None or value is None or value == "":
            continue
        try:
            number = spec[0](float(value))
        except (TypeError, ValueError):
            continue
        out[key] = max(spec[1], min(spec[2], number))
    return out


def sanitize_crop_reference(value) -> float:
    """SwarmUI "Crop Reference Strength" (local patch, Flux.2 / Klein): the
    fraction of the sampling, from the end, during which a cropped pass sees
    the original crop as a reference. 1 = always (SwarmUI's default)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, number))


#: Per-image Quick Mask overrides of the session's fixer settings.
OVERRIDE_KEYS = ("color_correction", "crop_reference")


def sanitize_overrides(raw: dict | None) -> dict:
    """The Quick Mask per-send overrides that were given, sanitized."""
    out: dict = {}
    raw = raw or {}
    if raw.get("color_correction") is not None:
        out["color_correction"] = sanitize_color_correction(raw["color_correction"])
    if raw.get("crop_reference") is not None:
        out["crop_reference"] = sanitize_crop_reference(raw["crop_reference"])
    return out


def sanitize_color_correction(value) -> str:
    """One of :data:`COLOR_CORRECTIONS` (any case), else ``"None"``."""
    text = str(value or "").strip().lower()
    return next((choice for choice in COLOR_CORRECTIONS if choice.lower() == text), "None")


def new_ledger(shuffle_seed: int, settings: dict | None = None) -> dict:
    return {
        "version": LEDGER_VERSION,
        "created_at": time.time(),
        "shuffle_seed": int(shuffle_seed),
        "settings": dict(settings or {}),
        "items": {},
    }


def load_ledger(concept_dir: str, session: str = MAIN_SESSION) -> dict | None:
    """The ledger, or ``None`` when there is none. A file that does not parse
    is moved aside and reported: silently starting over would re-offer every
    decided image."""
    path = ledger_path(concept_dir, session)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as ex:
        aside = f"{path}.corrupt-{time.strftime('%Y%m%dT%H%M%S')}"
        with suppress(OSError):
            os.replace(path, aside)
        raise RuntimeError(
            f"The Triage & Fix ledger in {concept_dir} is corrupt ({ex}); it was moved to "
            f"{os.path.basename(aside)}. The dataset itself is untouched."
        ) from ex
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
        raise RuntimeError(f"The Triage & Fix ledger in {concept_dir} has an unexpected shape")
    data.setdefault("version", LEDGER_VERSION)
    data.setdefault("settings", {})
    return data


def save_ledger(concept_dir: str, ledger: dict, session: str = MAIN_SESSION) -> None:
    path = ledger_path(concept_dir, session)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)


def _replace_with_retry(tmp: str, path: str, attempts: int = 20) -> None:
    """Windows refuses the rename while a reader holds the old file open (the
    UI's inspect, a resume); a few short retries cover that window."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # noqa: PERF203 — the retry is the point
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def item_seed(shuffle_seed: int, stem: str) -> int:
    """A stable 31-bit seed per image, so re-running a session reproduces
    the same candidates and candidate ``k`` is ``seed + k``."""
    return (int(shuffle_seed) * 1000003 + zlib.crc32(stem.encode("utf-8"))) & 0x7FFFFFFF


def rel(concept_dir: str, path: str) -> str:
    try:
        return os.path.relpath(path, concept_dir).replace("\\", "/")
    except ValueError:
        return path


def absolute(concept_dir: str, path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(concept_dir, path))


def status_of(ledger: dict, stem: str) -> str | None:
    record = ledger["items"].get(stem)
    return record.get("status") if isinstance(record, dict) else None


def ledger_counts(ledger: dict, stems: list[str]) -> dict:
    counts = dict.fromkeys(ALL_STATUSES, 0)
    undecided = 0
    for stem in stems:
        status = status_of(ledger, stem)
        if status in TRIAGE_STATUSES:
            undecided += 1
        if status in counts:
            counts[status] += 1
    counts["undecided"] = undecided
    counts["total"] = len(stems)
    return counts


def ready_stems(ledger: dict) -> list[str]:
    """Stems awaiting a decision, oldest result first."""
    ready = [
        (record.get("ready_at") or 0.0, stem)
        for stem, record in ledger["items"].items()
        if record.get("status") == STATUS_READY
    ]
    ready.sort()
    return [stem for _at, stem in ready]


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def enumerate_items(concept_dir: str, manifest: dict) -> list[dict]:
    """One item per manifest pair whose chosen file is on disk.

    ``caption`` is what the SwarmUI prompt carries: the ORIGINAL generation
    prompt from the manifest (brackets stripped), because that is what
    describes the picture Klein is repairing. The pair's ``.txt`` is only
    the fallback here. A caption corrected for training lives in that
    ``.txt`` and is read again, from disk, at pick time for the fix pair
    (see :func:`current_caption`). Unconditional pairs carry an empty
    caption and are flagged so a fix pair keeps them unconditional."""
    items: list[dict] = []
    chosen_root = os.path.join(concept_dir, "chosen")
    for entry in manifest.get("pairs", []):
        chosen_file = str(entry.get("chosen_file") or "")
        if not chosen_file:
            continue
        chosen_path = find_exported_file(chosen_root, chosen_file)
        if not chosen_path:
            continue
        stem = os.path.splitext(chosen_file)[0]
        manifest_prompt = str(entry.get("prompt") or "")
        unconditional = manifest_prompt == UNCONDITIONAL_PROMPT
        original = strip_angle_bracket_segments(manifest_prompt) if not unconditional else ""
        if original.strip():
            caption, source = original, "manifest"
        elif unconditional:
            caption, source = "", "unconditional"
        else:
            caption, source = read_pair_caption(concept_dir, entry) or "", "pair"
        size = image_size(chosen_path)
        items.append(
            {
                "stem": stem,
                "pair_id": entry.get("pair_id"),
                "chosen_file": chosen_file,
                "chosen_path": chosen_path,
                "caption": caption,
                "caption_source": source,
                "unconditional": unconditional,
                "width": size[0] if size else None,
                "height": size[1] if size else None,
                "aspectratio": str(entry.get("aspectratio") or ""),
            }
        )
    return items


def current_caption(concept_dir: str, entry: dict, fallback: str) -> str:
    """The pair's caption as it is on disk right now: the chosen ``.txt``
    when present (it may have been corrected since the session started),
    else ``fallback``. This is what a transferred fix pair is captioned
    with, so a corrected caption travels with the repaired image."""
    disk = read_pair_caption(concept_dir, entry)
    return disk if disk else fallback


# ---------------------------------------------------------------------------
# SwarmUI requests
# ---------------------------------------------------------------------------

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def image_data_uri(path: str) -> str:
    """The file as a ``data:`` URI SwarmUI's image parameters accept. Formats
    Swarm may not decode are re-encoded to PNG."""
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext)
    if mime is None:
        from PIL import Image

        buffer = io.BytesIO()
        with Image.open(path) as img:
            img.convert("RGB").save(buffer, format="PNG")
        data = buffer.getvalue()
        mime = "image/png"
    else:
        with open(path, "rb") as f:
            data = f.read()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


MAX_FIXER_LORAS = 8


def sanitize_loras(raw) -> list[dict]:
    """The fixer's LoRAs / adapters as ``[{"name", "weight"}]``: named as
    SwarmUI lists them, no repeats, weight clamped to -4..4 (default 1)."""
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        weight = max(-4.0, min(4.0, _as_float(item.get("weight"), 1.0)))
        out.append({"name": name, "weight": weight})
        if len(out) >= MAX_FIXER_LORAS:
            break
    return out


def klein_settings(raw: dict | None) -> dict:
    raw = raw or {}
    steps = raw.get("steps")
    cfg = raw.get("cfgscale")
    try:
        steps = int(float(steps)) if steps not in (None, "") else DEFAULT_KLEIN["steps"]
    except (TypeError, ValueError):
        steps = DEFAULT_KLEIN["steps"]
    try:
        cfg = float(cfg) if cfg not in (None, "") else DEFAULT_KLEIN["cfgscale"]
    except (TypeError, ValueError):
        cfg = DEFAULT_KLEIN["cfgscale"]
    shift = raw.get("sigmashift")
    try:
        shift = float(shift) if shift not in (None, "") else None
    except (TypeError, ValueError):
        shift = None
    return {
        "model": str(raw.get("model") or "").strip(),
        "steps": max(1, min(200, steps)),
        "cfgscale": max(0.0, min(100.0, cfg)),
        "sampler": str(raw.get("sampler") or "").strip(),
        "scheduler": str(raw.get("scheduler") or "").strip(),
        "sigmashift": shift,
        "loras": sanitize_loras(raw.get("loras")),
        "color_correction": sanitize_color_correction(raw.get("color_correction")),
        "crop_reference": sanitize_crop_reference(raw.get("crop_reference")),
    }


#: Candidates come back inline and are never written to SwarmUI's output
#: folder: that tree is what the curation tool groups and what "last
#: generation settings" is read from, and a Klein repair belongs in neither.
NO_SAVE_PARAMS = {"donotsave": True}


def _base_kwargs(prompt: str, size: tuple[int, int], klein: dict, seed: int) -> dict:
    if not klein.get("model"):
        raise ValueError("No fixer model selected")
    kwargs = {
        "prompt": prompt,
        "width": int(size[0]),
        "height": int(size[1]),
        "model": klein["model"],
        "images": 1,
        "steps": int(klein["steps"]),
        "cfg_scale": float(klein["cfgscale"]),
        "seed": int(seed),
    }
    # The fixer need not be Klein: a finetune inpaints with its own adapter on
    # top. Sent as request parameters, so they apply to every segment pass.
    loras = sanitize_loras(klein.get("loras"))
    if loras:
        kwargs["loras"] = [entry["name"] for entry in loras]
        kwargs["lora_weights"] = [entry["weight"] for entry in loras]
    return kwargs


def _sampler_params(klein: dict) -> dict:
    # Left out when blank so SwarmUI applies the model class's own defaults.
    out = {}
    if klein.get("sampler"):
        out["sampler"] = klein["sampler"]
    if klein.get("scheduler"):
        out["scheduler"] = klein["scheduler"]
    if klein.get("sigmashift") is not None:
        out["sigmashift"] = float(klein["sigmashift"])
    out.update(_color_params(klein.get("color_correction")))
    # Sent only below 1: at 1 it is SwarmUI's default, and a SwarmUI without
    # the local patch does not know the parameter.
    reference = sanitize_crop_reference(klein.get("crop_reference"))
    if reference < 1:
        out["cropreferencestrength"] = reference
    return out


def _color_params(color_correction: str | None) -> dict:
    # SwarmUI's own default is no correction, so "None" is simply left out.
    value = sanitize_color_correction(color_correction)
    return {} if value == "None" else {"colorcorrectionbehavior": value}


def segment_sub_prompt(fix: TriageFix, caption: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    """What one segment is refined with: its instruction, then the caption,
    joined by the session template. Without a caption only the instruction."""
    instruction = edit_instruction(fix)
    caption = (caption or "").strip()
    if not caption or not fix.include_caption:
        return instruction
    text = (template or DEFAULT_PROMPT_TEMPLATE).replace("{instruction}", instruction).replace("{caption}", caption)
    return text.strip()


def segment_prompt(caption: str, fixes: list[TriageFix], template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    """The whole Mode A prompt: no global text, one tag per fix, each with
    its own sub-prompt. The global prompt would only feed the skipped base
    pass, and the caption reaches every segment through its sub-prompt."""
    specs = []
    for fix in fixes:
        spec = fix.spec()
        spec.sub_prompt = segment_sub_prompt(fix, caption, template)
        specs.append(spec)
    return compose_prompt("", specs)


def build_segment_kwargs(
    caption: str,
    fixes: list[TriageFix],
    size: tuple[int, int],
    klein: dict,
    segment_params: dict | None,
    seed: int,
    init_uri: str,
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> dict:
    """Mode A: keyword arguments for ``SwarmClient.generate``.
    ``initimagecreativity`` 0 skips the base pass; only the segments sample."""
    prompt = segment_prompt(caption, fixes, template)
    kwargs = _base_kwargs(prompt, size, klein, seed)
    extra = {"initimage": init_uri, "initimagecreativity": 0}
    extra.update(_sampler_params(klein))
    extra.update(NO_SAVE_PARAMS)
    extra.update(sanitize_segment_params(segment_params or {}))
    kwargs["extra_params"] = extra
    return kwargs


def edit_instruction(fix: TriageFix) -> str:
    if fix.instruction.strip():
        return fix.instruction.strip().replace("{target}", target_noun(fix.target))
    return INSTRUCTIONS[CATEGORY_OTHER].replace("{target}", target_noun(fix.target))


def edit_instructions(fixes: list[TriageFix]) -> str:
    """The whole-image edit prompt for several Other fixes: each one's
    instruction, in order, without repeats."""
    parts: list[str] = []
    for fix in fixes:
        text = edit_instruction(fix)
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts)


def build_edit_kwargs(
    instruction: str,
    size: tuple[int, int],
    klein: dict,
    seed: int,
    prompt_uri: str,
    creativity: float = FIX_CREATIVITY,
) -> dict:
    """Mode B: the image as a prompt image, the instruction as the prompt.
    Below full ``creativity`` the image is also the init image, so sampling
    starts part-way down from it instead of from pure noise; SwarmUI starts
    at ``round(steps * (1 - creativity))``, so at Klein's 4 steps only
    quarters differ. At 1 it stays a pure reference edit."""
    kwargs = _base_kwargs(instruction, size, klein, seed)
    extra = {"promptimages": prompt_uri}
    creativity = _clamp01(creativity)
    if creativity < 1:
        extra.update({"initimage": prompt_uri, "initimagecreativity": creativity})
    extra.update(_sampler_params(klein))
    extra.update(NO_SAVE_PARAMS)
    kwargs["extra_params"] = extra
    return kwargs


def build_mask_kwargs(
    prompt: str,
    size: tuple[int, int],
    klein: dict,
    mask_params: dict | None,
    seed: int,
    init_uri: str,
    mask_uri: str,
    creativity: float = FIX_CREATIVITY,
) -> dict:
    """Mode C: masked inpaint of a painted region. Full creativity re-samples
    the crop from noise; "inpaint only masked" (``maskshrinkgrow``) crops to
    the mask with a context margin and pastes the result back onto the
    original, which ``initimagerecompositemask`` keeps pixel-exact outside.
    Per-image overrides are merged into ``klein`` by the caller."""
    kwargs = _base_kwargs(prompt, size, klein, seed)
    extra = {
        "initimage": init_uri,
        "maskimage": mask_uri,
        "initimagecreativity": _clamp01(creativity),
        "initimagerecompositemask": True,
    }
    extra.update(sanitize_mask_params(mask_params))
    extra.update(_sampler_params(klein))
    extra.update(NO_SAVE_PARAMS)
    kwargs["extra_params"] = extra
    return kwargs


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def candidate_change(original_path: str, candidate: str) -> tuple[float, float]:
    """``(mean |delta|, fraction of pixels changed)`` between the original
    and a candidate, both on 0..255 after scaling to at most 512px so the
    numbers are comparable across image sizes. Pixel-identical images
    (text chunks do not count) are ``(0.0, 0.0)``."""
    if compute_pixel_hash(original_path) == compute_pixel_hash(candidate):
        return 0.0, 0.0
    import numpy as np
    from PIL import Image

    with Image.open(original_path) as a, Image.open(candidate) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        if b.size != a.size:
            b = b.resize(a.size, Image.LANCZOS)
        scale = min(1.0, 512 / max(a.size))
        if scale < 1.0:
            target = (max(1, round(a.size[0] * scale)), max(1, round(a.size[1] * scale)))
            a = a.resize(target, Image.BOX)
            b = b.resize(target, Image.BOX)
        diff = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    moved = diff.max(axis=2) > NO_CHANGE_PIXEL_DELTA
    return float(diff.mean()), float(moved.mean())


def candidate_diff(original_path: str, candidate: str) -> float:
    """Mean |delta| only; see :func:`candidate_change`."""
    return candidate_change(original_path, candidate)[0]


def is_no_change(changed_fraction: float) -> bool:
    return changed_fraction < NO_CHANGE_FRACTION


def write_candidate(data: bytes, original_path: str, target_path: str, provenance: dict) -> None:
    """Save SwarmUI's result as a PNG that still carries the original's
    parameter block (the curation tool groups by it, Replace Pairs reads the
    model from it) plus a chunk describing this repair."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    try:
        with Image.open(original_path) as original:
            for key in _PARAM_METADATA_KEYS:
                value = original.info.get(key)
                if isinstance(value, str) and value:
                    info.add_text(key, value)
    except OSError:
        pass
    info.add_text(OT_CHUNK, json.dumps(provenance))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.save(target_path, format="PNG", pnginfo=info)


def delete_candidate_files(concept_dir: str, record: dict) -> None:
    for candidate in record.get("candidates") or []:
        path = absolute(concept_dir, candidate.get("path"))
        if path:
            with suppress(OSError):
                os.remove(path)


def all_no_change(record: dict) -> bool:
    candidates = [c for c in record.get("candidates") or [] if not c.get("error")]
    return bool(candidates) and all(c.get("no_change") for c in candidates)


# ---------------------------------------------------------------------------
# Replacing the chosen image in place
# ---------------------------------------------------------------------------


def replace_chosen(concept_dir: str, manifest: dict, entry: dict, candidate: str, note: dict) -> dict:
    """Make ``candidate`` the pair's chosen image under the same stem. The old
    file moves to ``<concept>/.replaced/`` exactly as Replace Pairs does, so
    the same undo and the same cleanup apply. The caption ``.txt`` and the
    rejected side are untouched. Returns the record ``undo_replace_chosen``
    needs."""
    chosen_file = str(entry.get("chosen_file") or "")
    old_path = find_exported_file(os.path.join(concept_dir, "chosen"), chosen_file)
    if not old_path:
        raise FileNotFoundError(f"chosen image missing for {chosen_file}")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(candidate)
    stem = os.path.splitext(chosen_file)[0]
    side_dir = os.path.dirname(old_path)
    replaced_dir = os.path.join(concept_dir, REPLACED_DIR)
    os.makedirs(replaced_dir, exist_ok=True)
    backup = _unique(os.path.join(replaced_dir, f"{stem}__{_timestamp()}_chosen{os.path.splitext(old_path)[1]}"))
    placed = os.path.join(side_dir, stem + ".png")
    before = _snapshot(entry)
    history_len = len(entry.get("history") or [])

    shutil.move(old_path, backup)
    try:
        shutil.copy2(candidate, placed)
    except OSError:
        with suppress(OSError):
            if os.path.isfile(placed):
                os.remove(placed)
        shutil.move(backup, old_path)
        raise

    try:
        size: int | None = os.path.getsize(placed)
    except OSError:
        size = None
    entry["chosen_file"] = os.path.basename(placed)
    entry["chosen_sha256"] = compute_file_sha256(placed)
    entry["chosen_size"] = size
    entry.setdefault("history", []).append(
        {
            "at": time.time(),
            "action": HISTORY_ACTION,
            "fixes": note.get("fixes"),
            "mode": note.get("mode"),
            "seed": note.get("seed"),
            "candidate": rel(concept_dir, candidate),
            "retired": [rel(concept_dir, backup)],
        }
    )
    save_manifest(concept_dir, manifest)
    return {
        "stem": stem,
        "placed": rel(concept_dir, placed),
        "backup": rel(concept_dir, backup),
        "old_path": rel(concept_dir, old_path),
        "before": before,
        "history_len": history_len,
    }


def undo_replace_chosen(concept_dir: str, manifest: dict, entry: dict, record: dict) -> None:
    """Reverse :func:`replace_chosen`: the placed copy goes, the backup
    returns to its old path, the entry's side fields and history are as
    they were."""
    placed = absolute(concept_dir, record.get("placed"))
    backup = absolute(concept_dir, record.get("backup"))
    old_path = absolute(concept_dir, record.get("old_path"))
    if not backup or not os.path.isfile(backup):
        raise FileNotFoundError(f"backup missing: {record.get('backup')}")
    if placed and os.path.isfile(placed) and os.path.normcase(placed) != os.path.normcase(backup):
        os.remove(placed)
    os.makedirs(os.path.dirname(old_path), exist_ok=True)
    shutil.move(backup, old_path)
    for key in list(record.get("before") or {}):
        entry[key] = record["before"][key]
    for key in ("chosen_file", "chosen_sha256", "chosen_size"):
        if key not in (record.get("before") or {}):
            entry.pop(key, None)
    history = entry.get("history") or []
    entry["history"] = history[: int(record.get("history_len", len(history)))]
    if not entry["history"]:
        entry.pop("history", None)
    save_manifest(concept_dir, manifest)


# ---------------------------------------------------------------------------
# Deleting an unfixable pair
# ---------------------------------------------------------------------------


def delete_pair(concept_dir: str, manifest: dict, entry: dict) -> dict:
    """Take a pair out of the dataset: its chosen and rejected images and the
    caption move into ``.triage_fix/deleted/`` and the manifest entry goes.
    Nothing is destroyed; :func:`undo_delete_pair` puts it all back."""
    pairs = manifest.get("pairs", [])
    try:
        index = next(i for i, p in enumerate(pairs) if p is entry or p.get("pair_id") == entry.get("pair_id"))
    except StopIteration as ex:
        raise KeyError("pair is not in the manifest") from ex
    stem = os.path.splitext(str(entry.get("chosen_file") or ""))[0]
    sources: list[str] = []
    for side in ("chosen", "rejected"):
        name = str(entry.get(f"{side}_file") or "")
        found = find_exported_file(os.path.join(concept_dir, side), name) if name else None
        if found:
            sources.append(found)
    caption = pair_caption_path(concept_dir, entry)
    if caption:
        sources.append(caption)
    target_dir = os.path.join(deleted_dir(concept_dir), f"{stem}__{_timestamp()}")
    target_dir = _unique(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    moved: list[tuple[str, str]] = []
    try:
        for source in sources:
            target = os.path.join(target_dir, rel(concept_dir, source).replace("/", "__"))
            shutil.move(source, target)
            moved.append((rel(concept_dir, source), rel(concept_dir, target)))
    except OSError:
        for source_rel, target_rel in reversed(moved):
            with suppress(OSError):
                shutil.move(absolute(concept_dir, target_rel), absolute(concept_dir, source_rel))
        raise
    removed = pairs.pop(index)
    save_manifest(concept_dir, manifest)
    return {"stem": stem, "index": index, "entry": removed, "moved": moved, "folder": rel(concept_dir, target_dir)}


def undo_delete_pair(concept_dir: str, manifest: dict, record: dict) -> dict:
    """Reverse :func:`delete_pair`; returns the restored manifest entry."""
    for source_rel, target_rel in record.get("moved") or []:
        target = absolute(concept_dir, target_rel)
        source = absolute(concept_dir, source_rel)
        if not target or not os.path.isfile(target):
            raise FileNotFoundError(f"deleted file missing: {target_rel}")
        os.makedirs(os.path.dirname(source), exist_ok=True)
        shutil.move(target, source)
    folder = absolute(concept_dir, record.get("folder"))
    if folder:
        with suppress(OSError):
            os.rmdir(folder)
    entry = dict(record["entry"])
    pairs = manifest.setdefault("pairs", [])
    index = min(int(record.get("index", len(pairs))), len(pairs))
    pairs.insert(index, entry)
    save_manifest(concept_dir, manifest)
    return entry


# ---------------------------------------------------------------------------
# Fix pairs in a second dataset
# ---------------------------------------------------------------------------


def fix_pair_folder_error(fix_dir: str, concept_dir: str) -> str | None:
    """Why ``fix_dir`` cannot take fix pairs, or ``None``. A new or empty
    folder is fine (it becomes a dataset on the first pair); an existing
    folder must already be a pair folder."""
    if not fix_dir:
        return None
    if is_within(fix_dir, concept_dir) or is_within(concept_dir, fix_dir):
        return "The fix pair folder must not be inside the dataset, nor contain it"
    if os.path.isfile(fix_dir):
        return "The fix pair folder is a file"
    if os.path.isdir(fix_dir):
        try:
            entries = [e for e in os.listdir(fix_dir) if not e.startswith(".")]
        except OSError as ex:
            return f"Cannot read the fix pair folder: {ex}"
        if entries and not os.path.isfile(os.path.join(fix_dir, "manifest.json")):
            return "The fix pair folder is not empty and is not a DPO pair folder (no manifest.json)"
    return None


def load_fix_manifest(fix_dir: str) -> dict:
    return load_manifest(fix_dir) if os.path.isdir(fix_dir) else {"pairs": []}


def write_fix_pair(
    fix_dir: str,
    fix_manifest: dict,
    fixed_path: str,
    original_path: str,
    caption: str,
    aspectratio: str,
    unconditional: bool = False,
) -> int:
    """The repaired image as chosen and the untouched original as rejected,
    same caption: a minimal pair whose only difference is the repair."""
    os.makedirs(fix_dir, exist_ok=True)
    prompt = UNCONDITIONAL_PROMPT if unconditional else caption
    export_single_pair(fix_dir, fix_manifest, fixed_path, original_path, prompt, aspectratio)
    return int(fix_manifest["pairs"][-1]["pair_id"])


def undo_fix_pair(fix_dir: str, fix_manifest: dict, pair_id: int) -> None:
    pairs = fix_manifest.get("pairs", [])
    entry = next((p for p in pairs if p.get("pair_id") == pair_id), None)
    if entry is None:
        return
    for side in ("chosen", "rejected"):
        name = str(entry.get(f"{side}_file") or "")
        if not name:
            continue
        path = os.path.join(fix_dir, side, name)
        with suppress(OSError):
            os.remove(path)
        if side == "chosen":
            with suppress(OSError):
                os.remove(os.path.join(fix_dir, side, os.path.splitext(name)[0] + ".txt"))
    fix_manifest["pairs"] = [p for p in pairs if p.get("pair_id") != pair_id]
    save_manifest(fix_dir, fix_manifest)


_FIX_PAIR_CHOSEN_KEYS = ("chosen_file", "chosen_sha256", "chosen_size")


def fix_pair_backup_dir(concept_dir: str) -> str:
    return os.path.join(ledger_dir(concept_dir), FIX_PAIR_REPLACED_DIR)


def find_fix_pair(fix_manifest: dict, pair_id: int) -> dict | None:
    return next((p for p in fix_manifest.get("pairs", []) if p.get("pair_id") == pair_id), None)


def replace_fix_pair_chosen(fix_dir: str, fix_manifest: dict, pair_id: int, fixed_path: str, backup_dir: str) -> dict:
    """A later round supersedes an existing fix pair's chosen side in place:
    same pair id, same rejected original, same caption. The old chosen image
    moves to ``backup_dir`` so :func:`undo_replace_fix_pair_chosen` can put it
    back. Raises ``KeyError`` when the pair is gone from the manifest."""
    entry = find_fix_pair(fix_manifest, pair_id)
    if entry is None:
        raise KeyError(pair_id)
    chosen_dir = os.path.join(fix_dir, "chosen")
    old_name = str(entry.get("chosen_file") or "")
    old_path = os.path.join(chosen_dir, old_name) if old_name else ""
    stem = os.path.splitext(old_name)[0] or f"pair_{int(pair_id):04d}"
    new_path = os.path.join(chosen_dir, stem + os.path.splitext(fixed_path)[1].lower())
    before = {key: entry.get(key) for key in _FIX_PAIR_CHOSEN_KEYS}
    backup = None
    os.makedirs(chosen_dir, exist_ok=True)
    if old_path and os.path.isfile(old_path):
        os.makedirs(backup_dir, exist_ok=True)
        backup = _unique(os.path.join(backup_dir, f"{stem}__{_timestamp()}_chosen{os.path.splitext(old_name)[1]}"))
        shutil.move(old_path, backup)
    try:
        shutil.copy2(fixed_path, new_path)
    except OSError:
        if backup:
            shutil.move(backup, old_path)
        raise
    entry["chosen_file"] = os.path.basename(new_path)
    entry["chosen_sha256"] = compute_file_sha256(new_path)
    entry["chosen_size"] = os.path.getsize(new_path)
    save_manifest(fix_dir, fix_manifest)
    return {"dir": fix_dir, "pair_id": int(pair_id), "replaced": True, "backup": backup, "before": before}


def undo_replace_fix_pair_chosen(fix_manifest: dict, record: dict) -> None:
    """Reverse :func:`replace_fix_pair_chosen`."""
    fix_dir = record["dir"]
    entry = find_fix_pair(fix_manifest, int(record["pair_id"]))
    if entry is None:
        return
    chosen_dir = os.path.join(fix_dir, "chosen")
    before = record.get("before") or {}
    current = os.path.join(chosen_dir, str(entry.get("chosen_file") or ""))
    if os.path.isfile(current):
        os.remove(current)
    backup = record.get("backup")
    if backup and os.path.isfile(backup):
        shutil.move(backup, os.path.join(chosen_dir, str(before.get("chosen_file") or "")))
    for key in _FIX_PAIR_CHOSEN_KEYS:
        entry[key] = before.get(key)
    save_manifest(fix_dir, fix_manifest)


def first_original(record: dict) -> str | None:
    """The pre-repair chosen image of a fixed item (relative to the concept),
    however many rounds it has been through: round one's backup."""
    for decision in [*(record.get("rounds") or []), record.get("decision")]:
        if isinstance(decision, dict) and decision.get("kind") == "fixed" and decision.get("backup"):
            return decision["backup"]
    return None


def second_pass_state(record: dict) -> str | None:
    """Where an item stands in the second round: ``None`` (not taken up),
    ``SECOND_OPEN`` or ``SECOND_SKIPPED``. A fixed image sent back to triage
    before the second round existed (rounds, no marker) counts as open."""
    second = record.get("second_pass")
    if isinstance(second, dict) and second.get("state"):
        return second["state"]
    if record.get("rounds") and record.get("status") in TRIAGE_STATUSES:
        return SECOND_OPEN
    return None


def prior_fix_pair(record: dict) -> dict | None:
    """The fix pair an earlier round of this item wrote or updated, if any."""
    for decision in reversed(record.get("rounds") or []):
        if isinstance(decision, dict) and decision.get("fix_pair"):
            return decision["fix_pair"]
    return None


# ---------------------------------------------------------------------------
# Several sessions on one dataset
# ---------------------------------------------------------------------------

#: An image in one of these is claimed by its session: another session on the
#: same dataset holds its own copy back until this one is decided.
PENDING_STATUSES = frozenset({STATUS_QUEUED, STATUS_GENERATING, STATUS_READY, STATUS_FAILED})
#: Share of a painted mask that may fall on pixels another session already
#: changed before the two count as overlapping (blur fringes stay under it).
MASK_OVERLAP_FRACTION = 0.01


def _region_words(fix: dict) -> set[str]:
    """What a segment fix looks for, as words: a YOLO file reads as its noun
    (``yolo-face_yolov8m.pt`` -> "face"), so it meets a CLIPSeg "face"."""
    words: set[str] = set()
    for key in ("target", "fallback_target"):
        target = str(fix.get(key) or "")
        words |= _tokens(target) | _tokens(target_noun(target))
    return words - {"area"}


def _fix_label(fix: dict) -> str:
    fix_id = str(fix.get("id") or "")
    if fix_id == CATEGORY_OTHER:
        return target_noun(str(fix.get("target") or "")) or "other"
    return fix_id.replace("_", " ")


def _read_mask(path: str, size: tuple[int, int] | None = None):
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        mask = img.convert("L")
        if size is not None and mask.size != size:
            mask = mask.resize(size, Image.NEAREST)
        return np.asarray(mask) > 127


def masks_overlap(a_path: str | None, b_path: str | None) -> bool:
    """Whether two painted masks share any pixel. A missing file counts as
    overlapping: the region cannot be shown to be clear."""
    if not a_path or not b_path or not os.path.isfile(a_path) or not os.path.isfile(b_path):
        return True
    from PIL import Image

    with Image.open(a_path) as img:
        size = img.size
    return bool((_read_mask(a_path) & _read_mask(b_path, size)).any())


def fixes_overlap(a_fixes: list[dict], a_mask: str | None, b_fixes: list[dict], b_mask: str | None) -> str | None:
    """Why two sessions' fixes on the same image would repaint the same area,
    or ``None`` when they look separate. A whole-image edit overlaps
    everything; two painted masks overlap when they share a pixel; two
    segment fixes when they are the same category or their targets share a
    word. A painted mask against a segment fix cannot be judged before the
    segment is found, so it passes here and is checked at dispatch."""
    a_ids = {str(f.get("id") or "") for f in a_fixes}
    b_ids = {str(f.get("id") or "") for f in b_fixes}
    if CATEGORY_PHOTO in a_ids or CATEGORY_PHOTO in b_ids:
        return "a whole-image edit"
    a_is_mask = CATEGORY_MASK in a_ids
    b_is_mask = CATEGORY_MASK in b_ids
    if a_is_mask and b_is_mask:
        return "an overlapping painted mask" if masks_overlap(a_mask, b_mask) else None
    if a_is_mask or b_is_mask:
        return None
    for a in a_fixes:
        for b in b_fixes:
            if a.get("id") == b.get("id") and a.get("id") != CATEGORY_OTHER:
                return f"the same fix ({_fix_label(a)})"
            if _region_words(a) & _region_words(b):
                return f"an overlapping target ({_fix_label(a)} / {_fix_label(b)})"
    return None


def mask_overlaps_change(mask_file: str, before_path: str, after_path: str) -> bool:
    """Whether a painted mask falls on pixels that changed between two
    versions of the image (another session's accepted repair)."""
    import numpy as np
    from PIL import Image

    if not os.path.isfile(mask_file) or not os.path.isfile(before_path) or not os.path.isfile(after_path):
        return False
    with Image.open(before_path) as a, Image.open(after_path) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        if b.size != a.size:
            b = b.resize(a.size, Image.LANCZOS)
        diff = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
        size = a.size
    moved = diff.max(axis=2) > NO_CHANGE_PIXEL_DELTA
    mask = _read_mask(mask_file, size)
    area = int(mask.sum())
    if area == 0:
        return False
    return int((mask & moved).sum()) > area * MASK_OVERLAP_FRACTION
