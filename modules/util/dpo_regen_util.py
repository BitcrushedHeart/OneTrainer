"""Fix & Regenerate: resend a SwarmUI generation with ``<segment:...>`` fixes.

Pure helpers behind the curation tool's Shift+Right-click action. An image's
own SwarmUI parameter block is the source of truth for everything (seed, model,
size, sampler, LoRAs); this module only composes the prompt that carries the
refine tags and trims the block down to what can be resent.

SwarmUI's segment syntax, as verified in its source (``PromptRegion.cs`` and
``WorkflowGeneratorSteps.cs``):

- ``<segment:target,creativity,threshold>`` — ``creativity`` is the inpaint
  strength (SwarmUI default 0.6), ``threshold`` the mask cutoff (0.5, or 0.25
  for ``yolo-`` targets). ``target`` may join several queries with ``|`` (one
  merged mask) or name a YOLO model: ``yolo-<model>[-<index>][:<classes>:]``.
- The text after a tag up to the next tag is that segment's sub-prompt. When it
  is empty the segment inherits the global prompt, so bare tags appended at
  the end refine with the original prompt — the common case here.
- Segments run one after another; each derives its seed from the main seed, so
  the same request reproduces the same file.
"""

import os
import re
from dataclasses import dataclass

from modules.util.swarm_generate_util import _params_from_image

#: The user's calibration (2026-09-15), not SwarmUI's defaults (0.6 / 0.5).
DEFAULT_CREATIVITY = 0.95
DEFAULT_THRESHOLD = 0.3


@dataclass(frozen=True)
class SegmentPreset:
    id: str
    label: str
    target: str


#: The dialog's built-in fixes. Targets are CLIPSeg text queries and stay
#: editable in the dialog — a ``yolo-`` model name works in the same slot.
SEGMENT_PRESETS: tuple[SegmentPreset, ...] = (
    SegmentPreset("faces", "Fix Faces", "face"),
    SegmentPreset("background_people", "Fix Background People", "person in the background"),
    SegmentPreset("text", "Fix Text", "text"),
    SegmentPreset("hands_feet", "Fix Hands/Feet", "hands|feet"),
    SegmentPreset("limbs", "Fix Limbs/Body", "arms|legs|torso"),
)


@dataclass
class FixSpec:
    target: str
    creativity: float = DEFAULT_CREATIVITY
    threshold: float = DEFAULT_THRESHOLD
    #: Text SwarmUI uses for the refined region instead of the global prompt.
    #: Survives the curation tool's bracket stripping, so it changes the
    #: image's group on a future scan — the dialog warns about that.
    sub_prompt: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "FixSpec":
        return cls(
            target=str(raw.get("target") or ""),
            creativity=_as_float(raw.get("creativity"), DEFAULT_CREATIVITY),
            threshold=_as_float(raw.get("threshold"), DEFAULT_THRESHOLD),
            sub_prompt=str(raw.get("sub_prompt") or ""),
        )


def _as_float(raw, default: float) -> float:
    try:
        if raw is None or raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _as_int(raw) -> int | None:
    try:
        if raw is None or raw == "":
            return None
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _clean_target(target: str) -> str:
    # Commas and angle brackets are the tag's own syntax; anything else
    # (spaces, "|", ":", "." in a yolo model name) is legitimate.
    cleaned = re.sub(r"[<>,]", " ", target or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _fmt(value: float) -> str:
    return f"{max(0.0, min(1.0, float(value))):g}"


def _fmt_threshold(value: float) -> str:
    # SwarmUI inverts the mask for a negative threshold (everything except the
    # match), so the sign is meaningful and only the magnitude is clamped.
    return f"{max(-1.0, min(1.0, float(value))):g}"


def segment_tag(fix: FixSpec) -> str:
    """``<segment:face,0.95,0.3>`` for one fix. ``ValueError`` on an empty target."""
    target = _clean_target(fix.target)
    if not target:
        raise ValueError("A fix needs a target (what to segment)")
    return f"<segment:{target},{_fmt(fix.creativity)},{_fmt_threshold(fix.threshold)}>"


_SEGMENT_TAG_RE = re.compile(r"<segment:\s*([^,>]*?)\s*(?:,[^>]*)?>", re.IGNORECASE)

#: Same tag, with the two numeric arguments captured. ``remove_segment`` keeps
#: the looser pattern above so a malformed tag is still removed.
_SEGMENT_TAG_FULL_RE = re.compile(
    r"<segment:\s*([^,>]*?)\s*(?:,\s*(-?[0-9.]*)\s*)?(?:,\s*(-?[0-9.]*)\s*)?>", re.IGNORECASE
)


def parse_segment_string(text: str) -> list[FixSpec]:
    """Every ``<segment:target[,creativity[,threshold]]>`` tag in ``text`` as a
    :class:`FixSpec`, in order. Missing or unparsable numbers take this
    module's defaults. This is the wire format of the fix detector: one
    string of tags per image."""
    fixes: list[FixSpec] = []
    for match in _SEGMENT_TAG_FULL_RE.finditer(text or ""):
        target = _clean_target(match.group(1))
        if not target:
            continue
        fixes.append(
            FixSpec(
                target=target,
                creativity=_as_float(match.group(2), DEFAULT_CREATIVITY),
                threshold=_as_float(match.group(3), DEFAULT_THRESHOLD),
            )
        )
    return fixes


def remove_segment(prompt: str, target: str) -> str:
    """Drop every existing ``<segment:target...>`` tag for the same target
    (case-insensitive) together with its sub-prompt — the text up to the next
    ``<`` tag or the end. Other tags, ``<lora:...>`` included, are untouched."""
    wanted = _clean_target(target).lower()
    if not wanted:
        return prompt
    out = []
    pos = 0
    for match in _SEGMENT_TAG_RE.finditer(prompt):
        if _clean_target(match.group(1)).lower() != wanted:
            continue
        out.append(prompt[pos : match.start()])
        end = prompt.find("<", match.end())
        pos = len(prompt) if end < 0 else end
    out.append(prompt[pos:])
    return "".join(out)


def compose_prompt(prompt: str, fixes: list[FixSpec]) -> str:
    """The original prompt with one tag per fix appended.

    Same-target tags already in the prompt are replaced, so regenerating a
    fixed image with a stronger setting does not stack two face passes. Tags
    are separated by a single space: the prompt keeps its own punctuation and
    SwarmUI's parser ignores the whitespace."""
    result = prompt or ""
    tags = []
    for fix in fixes:
        result = remove_segment(result, fix.target)
        tag = segment_tag(fix)
        sub = re.sub(r"\s+", " ", fix.sub_prompt or "").strip()
        tags.append(f"{tag} {sub}" if sub else tag)
    result = re.sub(r"[ \t]{2,}", " ", result).strip()
    if tags:
        result = " ".join([result, *tags]).strip()
    return result


# ---------------------------------------------------------------------------
# Reusing the image's own parameter block
# ---------------------------------------------------------------------------

#: Keys that describe the output rather than the request, or that this tool
#: decides for itself. ``width``/``height`` are resent explicitly, so the
#: derived size keys go. Image-typed inputs cannot be reproduced from the
#: block (SwarmUI stores them as references, not data).
RESEND_DROP_KEYS = frozenset(
    {
        "swarm_version",
        "date",
        "generation_time",
        "prep_time",
        "images",
        "batchsize",
        "aspectratio",
        "rawresolution",
        "sidelength",
        "original_prompt",
        "scoring",
        "exactbackendid",
        "internalbackendtype",
        "initimage",
        "maskimage",
        "promptimages",
        "controlnetimageinput",
        "videoextendframeoverlap",
    }
)


def _image_size(path: str) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.size[0]), int(img.size[1])
    except Exception:
        return None


def resend_params_from_image(path: str) -> dict:
    """The image's SwarmUI parameter block, trimmed to what a regeneration
    resends verbatim. ``{}`` when the image carries no block (a photo, or a
    file re-encoded without its metadata).

    Everything that survives is sent back exactly as recorded — the block is
    SwarmUI's own account of what made the image, so a ``<lora:...>`` tag in
    the prompt next to a ``loras`` list is reproduced, not second-guessed.
    Segment-refining keys from an earlier fix run are dropped: this run's
    settings replace them."""
    raw = _params_from_image(path)
    if not raw:
        return {}
    params: dict = {}
    for key, value in raw.items():
        if key in RESEND_DROP_KEYS or value is None:
            continue
        if key.startswith(("segment", "savesegment")):
            continue
        if isinstance(value, str) and value.startswith("data:image"):
            continue
        params[key] = value
    size = _image_size(path)
    for axis, index in (("width", 0), ("height", 1)):
        value = _as_int(params.get(axis))
        if value is None and size is not None:
            value = size[index]
        if value is None:
            params.pop(axis, None)
        else:
            params[axis] = value
    if "seed" in params:
        try:
            seed = int(params["seed"])
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            params.pop("seed")
        else:
            params["seed"] = seed
    return params


#: SwarmUI's "Segment Refining" parameter group (API ids). ``(type, lo, hi)``
#: for numbers, ``(type,)`` for free values, a tuple of strings for choices.
SEGMENT_PARAM_SPECS: dict[str, tuple] = {
    "savesegmentmask": (bool,),
    "segmentmaskblur": (int, 0, 64),
    "segmentmaskgrow": (int, 0, 512),
    "segmentmaskoversize": (int, 0, 512),
    "segmentthresholdmax": (float, 0.0, 1.0),
    "segmentsortorder": (str,),
    "segmentapplyafter": ("Refiner", "Base"),
    "segmenttargetresolution": (str,),
    "segmentmodel": (str,),
    "segmentsteps": (int, 1, 200),
    "segmentcfgscale": (float, 0.0, 100.0),
}


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return None


def sanitize_segment_params(raw: dict) -> dict:
    """Coerce and clamp the segment-refining keys; drop blanks and unknowns."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, value in raw.items():
        spec = SEGMENT_PARAM_SPECS.get(key)
        if spec is None or value is None or value == "":
            continue
        kind = spec[0]
        if kind is bool:
            parsed = _as_bool(value)
            if parsed is not None:
                out[key] = parsed
        elif kind is int or kind is float:
            try:
                number = kind(float(value))
            except (TypeError, ValueError):
                continue
            out[key] = max(spec[1], min(spec[2], number))
        elif kind is str:
            text = str(value).strip()
            if text:
                out[key] = text
        else:
            # A choice list.
            text = str(value).strip()
            for choice in spec:
                if text.lower() == choice.lower():
                    out[key] = choice
                    break
    return out


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def build_generate_kwargs(params: dict, prompt: str, segment_params: dict, model_override: str = "") -> dict:
    """Keyword arguments for :meth:`SwarmClient.generate`.

    Lifts the keys the client takes explicitly (prompt, size, model, seed,
    negative prompt, LoRAs) and passes everything else flat through
    ``extra_params``, followed by the segment settings. Always one image: the
    request is deterministic, so more copies would be the same file."""
    model = (model_override or "").strip() or str(params.get("model") or "").strip()
    if not model:
        raise ValueError("No model recorded for this image")
    try:
        width = int(params["width"])
        height = int(params["height"])
    except (KeyError, TypeError, ValueError) as ex:
        raise ValueError("No size recorded for this image") from ex
    loras = [str(x) for x in _as_list(params.get("loras"))]
    weights = [_as_float(raw, 1.0) for raw in _as_list(params.get("loraweights"))]
    lifted = {"prompt", "width", "height", "model", "seed", "negativeprompt", "loras", "loraweights", "images"}
    extra = {key: value for key, value in params.items() if key not in lifted}
    extra.update(sanitize_segment_params(segment_params))
    seed = params.get("seed")
    kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "model": model,
        "images": 1,
        "seed": int(seed) if seed is not None else -1,
        "negative_prompt": str(params.get("negativeprompt") or ""),
        "loras": loras or None,
        "lora_weights": weights or None,
        "extra_params": extra,
    }
    return kwargs


def fix_output_path(source_path: str) -> str:
    """``<dir>/<stem>_fix<N>.png`` beside the source, first free N."""
    directory, name = os.path.split(source_path)
    stem = os.path.splitext(name)[0]
    n = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_fix{n}.png")
        if not os.path.exists(candidate):
            return candidate
        n += 1
