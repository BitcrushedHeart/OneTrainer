"""Pure logic for generating off-policy negatives through SwarmUI.

Two jobs: work out the size to generate a photo's caption at, and decide
which captions are worth generating for. Kept free of I/O and of the service
layer so both are testable without a server.
"""

import json
import os
import threading

from modules.util.caption_style import is_tag_line
from modules.util.dpo_offpolicy_util import RealPhoto, exact_size_for_small_photo

#: Trainer target resolution and bucket quantization these sizes are built
#: for. 64 is what every current OneTrainer image loader passes to
#: ``AspectBucketing`` (see ``ZImageBaseDataLoader``).
DEFAULT_TARGET_RESOLUTION = 1024
DEFAULT_QUANTIZATION = 64

_BUCKET_LOCK = threading.Lock()
_BUCKET_CACHE: dict[tuple[int, int], tuple[list[tuple[int, int]], list[float]]] = {}


def _bucket_grid(target_resolution: int, quantization: int) -> tuple[list[tuple[int, int]], list[float]]:
    """``([(h, w), ...], [h/w, ...])`` — the trainer's own bucket grid.

    Built by mgds' ``AspectBucketing`` itself rather than re-derived here, so
    a change to the trainer's bucketing cannot silently desync the sizes this
    tool asks SwarmUI for. The module is constructed with placeholder pipeline
    names; only its pure bucket construction is used."""
    key = (int(target_resolution), int(quantization))
    with _BUCKET_LOCK:
        cached = _BUCKET_CACHE.get(key)
    if cached is not None:
        return cached

    from mgds.pipelineModules.AspectBucketing import AspectBucketing

    module = AspectBucketing(
        quantization=key[1],
        resolution_in_name="resolution",
        target_resolution_in_name="target_resolution",
        enable_target_resolutions_override_in_name="enable_override",
        target_resolutions_override_in_name="override",
        target_frames_in_name="frames",
        frame_dim_enabled=False,
        scale_resolution_out_name="scale_resolution",
        crop_resolution_out_name="crop_resolution",
        possible_resolutions_out_name="possible_resolutions",
    )
    # Name-mangled: the construction is pure, but private to the module.
    resolutions, aspects = module._AspectBucketing__create_automatic_buckets([key[0]])
    grid = (list(resolutions[key[0]]), list(aspects[key[0]]))
    with _BUCKET_LOCK:
        _BUCKET_CACHE[key] = grid
    return grid


def bucket_sizes(
    target_resolution: int = DEFAULT_TARGET_RESOLUTION, quantization: int = DEFAULT_QUANTIZATION
) -> list[tuple[int, int]]:
    """Every trainer bucket as ``(width, height)``, widest first."""
    resolutions, _aspects = _bucket_grid(target_resolution, quantization)
    return sorted(((w, h) for h, w in resolutions), key=lambda size: -size[0] / size[1])


def bucket_size_for_photo(
    width: int,
    height: int,
    target_resolution: int = DEFAULT_TARGET_RESOLUTION,
    quantization: int = DEFAULT_QUANTIZATION,
) -> tuple[int, int]:
    """The ``(width, height)`` to generate this photo's caption at: exactly the
    trainer bucket the photo itself trains in.

    Generating at the bucket rather than at the photo's raw aspect is what
    removes cropping from the pair entirely — the rejected image already *is*
    bucket-shaped, so the loader's cover-crop is a no-op on it, while the
    chosen photo gets the same scale-crop it always would. The web UI's fixed
    aspect list does not apply to the API, so any bucket can be requested."""
    if width <= 0 or height <= 0:
        return target_resolution, target_resolution
    # Below the budget the photo's own dimensions are used, so the photo is
    # never upscaled to meet a bucket; OneTrainer's scale-crop handles the
    # small pair at train time.
    exact = exact_size_for_small_photo(width, height, target_resolution)
    if exact is not None:
        return exact
    resolutions, aspects = _bucket_grid(target_resolution, quantization)
    target = height / width
    best_index = min(range(len(aspects)), key=lambda i: abs(aspects[i] - target))
    bucket_h, bucket_w = resolutions[best_index]
    return int(bucket_w), int(bucket_h)


# ---------------------------------------------------------------------------
# Planning a batch
# ---------------------------------------------------------------------------


def line_is_usable(line: str, exclude_tag_style: bool = False, min_caption_words: int = 0) -> str | None:
    """Why one caption line should not be generated for — ``"tag_style"`` or
    ``"too_short"`` — or ``None`` when it is fine. Tag style is checked first
    so a short tag list is reported as tag style, the more specific reason."""
    if exclude_tag_style and is_tag_line(line):
        return "tag_style"
    if min_caption_words > 0 and len(line.split()) < min_caption_words:
        return "too_short"
    return None


def filter_pool_lines(
    pool: list[tuple[RealPhoto, list[str]]],
    exclude_tag_style: bool = False,
    min_caption_words: int = 0,
) -> tuple[list[tuple[RealPhoto, list[str]]], dict[str, int]]:
    """Narrow a suggestion pool to the caption *lines* worth generating for,
    the way :func:`~modules.util.dpo_offpolicy_util.filter_pool_by_keywords`
    narrows by keyword.

    A multi-line sidecar is several captions, and the trainer draws one at
    random per step, so the filters apply per line, not per file: a photo
    whose sidecar mixes a short line with a long one keeps the long line
    rather than being judged by whichever line a draw happens to pick. A
    photo stays while at least one of its eligible lines survives.

    ``stats`` counts the photos dropped entirely, once each: under
    ``too_short`` when any of its lines fell to the word minimum, otherwise
    under ``tag_style``."""
    stats = {"tag_style": 0, "too_short": 0}
    if not exclude_tag_style and min_caption_words <= 0:
        return pool, stats
    narrowed: list[tuple[RealPhoto, list[str]]] = []
    for photo, lines in pool:
        kept: list[str] = []
        reasons: set[str] = set()
        for line in lines:
            reason = line_is_usable(line, exclude_tag_style, min_caption_words)
            if reason is None:
                kept.append(line)
            else:
                reasons.add(reason)
        if kept:
            narrowed.append((photo, kept))
        elif reasons:
            stats["too_short" if "too_short" in reasons else "tag_style"] += 1
    return narrowed, stats


def plan_generations(
    entries: list[dict],
    images_per_caption: int = 1,
    exclude_tag_style: bool = False,
    min_caption_words: int = 0,
    target_resolution: int = DEFAULT_TARGET_RESOLUTION,
    quantization: int = DEFAULT_QUANTIZATION,
) -> tuple[list[dict], dict[str, int]]:
    """Turn suggester entries into concrete generation jobs.

    Each entry needs ``caption`` and ``real_size``; ``real_path`` is carried
    through for reporting. Returns ``(jobs, stats)`` where a job has the
    caption, the exact bucket ``width``/``height`` for its photo, and the
    number of images to make. ``stats`` counts what was dropped and why.

    ``exclude_tag_style`` drops booru-style tag lists (see
    :mod:`modules.util.caption_style`) — as a prompt they under-specify the
    scene, so the generation shares little with the photo beyond keywords and
    the pair teaches composition rather than realism. ``min_caption_words``
    additionally drops very short captions for the same reason."""
    jobs: list[dict] = []
    stats = {"tag_style": 0, "too_short": 0, "unusable": 0}
    for entry in entries:
        caption = (entry.get("caption") or "").strip()
        size = entry.get("real_size") or (0, 0)
        try:
            width, height = int(size[0]), int(size[1])
        except (TypeError, ValueError, IndexError):
            width = height = 0
        if not caption or width <= 0 or height <= 0:
            stats["unusable"] += 1
            continue
        reason = line_is_usable(caption, exclude_tag_style, min_caption_words)
        if reason is not None:
            stats[reason] += 1
            continue
        gen_w, gen_h = bucket_size_for_photo(width, height, target_resolution, quantization)
        jobs.append(
            {
                "caption": caption,
                "real_path": entry.get("real_path"),
                "real_size": [width, height],
                "width": gen_w,
                "height": gen_h,
                # Small photos generate at their own size rather than a bucket.
                "exact_size": exact_size_for_small_photo(width, height, target_resolution) is not None,
                "images": max(1, int(images_per_caption)),
            }
        )
    return jobs, stats


# ---------------------------------------------------------------------------
# Reusing the settings the user is already generating with
# ---------------------------------------------------------------------------

#: Keys this tool always decides for itself, whatever the source says. The
#: prompt is the caption, the size is the photo's bucket, the seed must vary
#: across a batch of negatives, and the image count is the user's choice here.
OVERRIDDEN_PARAM_KEYS = frozenset(
    {"prompt", "width", "height", "aspectratio", "rawresolution", "seed", "images", "batchsize", "swarm_version"}
)

#: PNG text chunks SwarmUI writes its parameter block into.
_PARAM_METADATA_KEYS = ("sui_image_params", "parameters")


def sanitize_source_params(params: dict) -> dict:
    """Everything from a settings source that is safe to reuse verbatim.

    Keeps sampler, scheduler, sigma shift, CFG variants, VAE, LoRAs and any
    other model-specific switch — the parts that make a generation match what
    the user is doing in the web UI — and drops what this tool sets itself
    (:data:`OVERRIDDEN_PARAM_KEYS`)."""
    if not isinstance(params, dict):
        return {}
    return {key: value for key, value in params.items() if key not in OVERRIDDEN_PARAM_KEYS and value is not None}


def _params_from_image(path: str) -> dict:
    """SwarmUI's parameter block from one generated image, or ``{}``."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            info = dict(img.info or {})
    except Exception:
        return {}
    for key in _PARAM_METADATA_KEYS:
        raw = info.get(key)
        if not isinstance(raw, str):
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data.get("sui_image_params", data) if isinstance(data.get("sui_image_params"), dict) else data
    return {}


def read_last_generation_params(generated_folder: str, search_limit: int = 40) -> dict:
    """Parameters of the most recent image in the generated tree.

    This is the closest thing to "what the user is generating with right now":
    SwarmUI stamps the full parameter set into every image it writes, so the
    newest one carries their live sampler, sigma shift, LoRAs and model. Only
    the newest few files are opened, and dot-folders (pending, discarded) are
    skipped so a held-back image never becomes the template."""
    from modules.util.path_util import supported_image_extensions

    supported = supported_image_extensions()
    candidates: list[tuple[float, str]] = []
    for root, dirs, files in os.walk(generated_folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if os.path.splitext(name)[1].lower() not in supported:
                continue
            path = os.path.join(root, name)
            try:
                candidates.append((os.path.getmtime(path), path))
            except OSError:
                continue
    if not candidates:
        return {}
    candidates.sort(reverse=True)
    for _mtime, path in candidates[:search_limit]:
        params = _params_from_image(path)
        if params:
            return params
    return {}


def merge_params(base: dict, overrides: dict) -> dict:
    """Source parameters with the caller's explicit choices layered on top.

    An override is only applied when it is actually set — a blank steps field
    means "whatever the source said", not "zero"."""
    merged = sanitize_source_params(base)
    for key, value in (overrides or {}).items():
        if value is None or value == "" or value == []:
            continue
        merged[key] = value
    return merged
