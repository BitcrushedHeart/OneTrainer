"""Off-policy DPO pairs: a real photo as *chosen* against the model's own
generation of one of that photo's captions as *rejected*.

The generated side comes from the same metadata-scanned trees the curation
tool uses; the real side is an ordinary OneTrainer concept folder with ``.txt``
sidecars. A sidecar may hold several independent captions, one per line
(``LoadMultipleTexts`` picks one at random at train time), so matching is
per-line and exact: a generation counts only if its prompt is one of the
photo's caption lines in full.

Everything here is pure bookkeeping over paths, strings and sizes so it can be
tested without a model; the service layers scoring and file I/O on top.
"""

import json
import math
import os
import random
import shutil
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field

from modules.util.dpo_curation_util import (
    UNCONDITIONAL_PROMPT,
    _normalize_source_path,
    normalize_prompt_for_grouping,
    trainer_bucket_key,
    walk_skipping_dotted,
)
from modules.util.path_util import supported_image_extensions

#: Default allowed aspect mismatch between the photo and the generation, as a
#: fraction: 0.15 accepts a generation whose aspect is within 15% of the
#: photo's.
#:
#: What the mismatch costs at train time is a *crop*, never a stretch: the
#: loader (``CoverScaleCropImage``) scales the rejected image uniformly so it
#: covers the chosen photo's crop box and trims the overshoot with the photo's
#: relative crop offset. A 15% aspect gap therefore loses ~13% of the
#: generation along one axis. Generate at the photo's own aspect (see
#: :func:`suggested_generation_size`) to make it zero.
DEFAULT_ASPECT_TOLERANCE = 0.15


def caption_sidecar_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + ".txt"


def read_caption_lines(txt_path: str) -> list[str]:
    """The caption lines the trainer would draw from — the exact split
    ``LoadMultipleTexts`` performs (strip each line, drop empties). A missing
    or unreadable sidecar reads as no captions, never as an error."""
    try:
        with open(txt_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f]
    except (OSError, UnicodeDecodeError):
        return []
    return [line for line in lines if line != ""]


def caption_match_key(text: str) -> str:
    """Key under which a caption line and a generation prompt compare equal.

    Runs the same normalization the curation scan applies to generation
    prompts (``<...>`` metadata segments dropped, whitespace runs collapsed) so
    a caption never fails to match its own generation over a ``<lora:...>``
    tag or a doubled space the generator introduced. Content must still match
    in full — this is not a fuzzy or prefix comparison.

    A UTF-8 BOM (which ``open(encoding="utf-8")`` leaves on the first line of
    a sidecar) and the NFC/NFD form of accented characters are not content
    either: neither survives into the generator's prompt, so both are folded
    away rather than causing a silent miss on the first caption of a file."""
    text = unicodedata.normalize("NFC", (text or "").lstrip("\ufeff"))
    normalized = normalize_prompt_for_grouping(text)
    if normalized == UNCONDITIONAL_PROMPT:
        return ""
    return " ".join(normalized.split()).casefold()


def aspect_mismatch(real_size: tuple[int, int], generated_size: tuple[int, int]) -> float | None:
    """Relative aspect difference between two images as a fraction, e.g. 0.12
    when one is 12% wider-per-height than the other. Symmetric (measured in
    log-aspect, so 1.15x and 1/1.15x both read 0.15). For a DPO pair this is
    the share of the generation's long axis the loader's cover-crop trims
    away (as a fraction of the crop box). ``None`` for a non-positive
    dimension."""
    rw, rh = real_size
    gw, gh = generated_size
    if min(rw, rh, gw, gh) <= 0:
        return None
    return abs(math.exp(abs(math.log((rw / rh) / (gw / gh)))) - 1.0)


#: How much larger than the photo a generation may be, per axis, before the
#: photo is excluded: 1.10 lets a photo be upscaled by up to 10% relative to
#: its generation. Generators emit fixed sizes (832x1216, 1024x1024, ...) that
#: are exactly the training crop, and a large share of real photos sit a few
#: percent under that — they get the same small upscale in SFT anyway, and a
#: 10% resample is far below the realism gap the pair is meant to teach.
DEFAULT_MAX_UPSCALE = 1.10


def real_covers_generation(
    real_size: tuple[int, int], generated_size: tuple[int, int], max_upscale: float = DEFAULT_MAX_UPSCALE
) -> bool:
    """The photo must be at least ``1/max_upscale`` of the generation on both
    axes (``max_upscale=1.0`` ⇒ at least as large).

    Both halves of a pair are scaled to the same crop box at train time. A
    photo much smaller than its generation is *upscaled* relative to it, so
    the pair would teach "soft/upsampled = real, crisp = AI" — a resampling
    cue, not realism. Photos beyond the allowance are excluded, not paired."""
    scale = max(1.0, float(max_upscale))
    return real_size[0] * scale >= generated_size[0] and real_size[1] * scale >= generated_size[1]


def aspect_within_tolerance(
    real_size: tuple[int, int], generated_size: tuple[int, int], tolerance: float = DEFAULT_ASPECT_TOLERANCE
) -> bool:
    mismatch = aspect_mismatch(real_size, generated_size)
    return mismatch is not None and mismatch <= tolerance


@dataclass
class RealPhoto:
    path: str
    caption_path: str
    lines: list[str]
    width: int
    height: int


@dataclass
class GeneratedImage:
    path: str
    prompt_key: str
    width: int
    height: int


@dataclass
class Candidate:
    """One proposed pair, before scoring picked the rejected image."""

    real: RealPhoto
    caption: str
    #: Generations of ``caption`` whose aspect is within tolerance of the photo.
    generated: list[GeneratedImage] = field(default_factory=list)
    #: The chosen hardest negative (most photorealistic generation), once scored.
    rejected: GeneratedImage | None = None
    rejected_score: float | None = None
    #: P(AI) of the real photo itself, when the head scored it. Shown next to
    #: the rejected score: a photo the head finds *less* real than the
    #: generation is worth a second look before it becomes a chosen sample.
    real_score: float | None = None
    #: Whether ``rejected`` was picked on a realism score or by fallback order.
    scored: bool = False
    #: Caption line each generation was made from, so the exported caption is
    #: the one the chosen negative actually answers.
    lines_by_path: dict[str, str] = field(default_factory=dict)
    #: Generations already shown for this photo (re-roll history), in order.
    shown_paths: list[str] = field(default_factory=list)

    def rejected_index(self) -> int:
        """Position of ``rejected`` in ``generated`` (-1 when nothing is picked)."""
        if self.rejected is None:
            return -1
        return next((i for i, g in enumerate(self.generated) if g.path == self.rejected.path), -1)

    def to_dict(self) -> dict:
        data = {
            "real_path": self.real.path,
            "caption_path": self.real.caption_path,
            "real_size": [self.real.width, self.real.height],
            "caption": self.caption,
            "caption_lines": list(self.real.lines),
            "generated_count": len(self.generated),
            "rejected_path": self.rejected.path if self.rejected else None,
            "rejected_size": [self.rejected.width, self.rejected.height] if self.rejected else None,
            "rejected_score": self.rejected_score,
            "real_score": self.real_score,
            "scored": self.scored,
            "shown_count": len(self.shown_paths),
            # Every usable generation, so a picker can show them side by side
            # and name the one it wants as the rejected image.
            "generated_paths": [g.path for g in self.generated],
            "rejected_index": self.rejected_index(),
        }
        if self.rejected is not None:
            data["aspect_mismatch"] = aspect_mismatch(
                (self.real.width, self.real.height), (self.rejected.width, self.rejected.height)
            )
        return data


def image_size(path: str) -> tuple[int, int] | None:
    """(width, height) from the image header, or None when it can't be read.
    ``Image.open`` is lazy, so this costs one small read per file."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def index_real_photos(folder: str, size_of=image_size) -> list[RealPhoto]:
    """Every supported image under ``folder`` (dot-dirs pruned) that has at
    least one caption line and a readable size. Sorted by path so a session
    walks the dataset in a stable order across runs."""
    supported = supported_image_extensions()
    photos: list[RealPhoto] = []
    for root, files in walk_skipping_dotted(folder):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() not in supported:
                continue
            path = os.path.join(root, filename)
            caption_path = caption_sidecar_path(path)
            lines = read_caption_lines(caption_path)
            if not lines:
                continue
            size = size_of(path)
            if size is None:
                continue
            photos.append(RealPhoto(path, caption_path, lines, size[0], size[1]))
    photos.sort(key=lambda p: p.path)
    return photos


def used_caption_keys_from_manifest(manifest: dict) -> set[str]:
    return {caption_match_key(entry.get("prompt", "")) for entry in manifest.get("pairs", [])} - {""}


def used_caption_keys_from_chosen_dir(chosen_dir: str) -> set[str]:
    """Caption keys of every exported pair under a DPO concept's chosen folder.

    Exported sidecars are single-line by construction (``export_single_pair``),
    but a hand-edited one is read the way the trainer would — every line is a
    caption that already has a pair."""
    keys: set[str] = set()
    if not os.path.isdir(chosen_dir):
        return keys
    for root, _dirs, files in os.walk(chosen_dir):
        for filename in files:
            if not filename.lower().endswith(".txt"):
                continue
            for line in read_caption_lines(os.path.join(root, filename)):
                key = caption_match_key(line)
                if key:
                    keys.add(key)
    return keys


def group_generated_by_key(generated: list[GeneratedImage]) -> dict[str, list[GeneratedImage]]:
    by_key: dict[str, list[GeneratedImage]] = {}
    for gen in generated:
        if gen.prompt_key:
            by_key.setdefault(gen.prompt_key, []).append(gen)
    return by_key


def usable_generations(
    photo: RealPhoto,
    by_key: dict[str, list[GeneratedImage]],
    *,
    excluded_caption_keys: set[str],
    used_sources: set[str],
    aspect_tolerance: float,
    max_upscale: float = DEFAULT_MAX_UPSCALE,
    stats: dict[str, int] | None = None,
) -> list[tuple[str, GeneratedImage]]:
    """(caption line, generation) for every generation of one of the photo's
    caption lines that is free, aspect-compatible, no larger than the photo,
    and whose caption is not already paired. A generation appears once even
    when two sidecar lines normalize to the same key. With ``stats`` given,
    ``stats["too_small"]`` counts generations the photo could not cover."""
    real_size = (photo.width, photo.height)
    matched: list[tuple[str, GeneratedImage]] = []
    seen_paths: set[str] = set()
    for line in photo.lines:
        key = caption_match_key(line)
        if not key or key in excluded_caption_keys:
            continue
        for gen in by_key.get(key, ()):
            if gen.path in seen_paths or _normalize_source_path(gen.path) in used_sources:
                continue
            gen_size = (gen.width, gen.height)
            if not aspect_within_tolerance(real_size, gen_size, aspect_tolerance):
                continue
            if not real_covers_generation(real_size, gen_size, max_upscale):
                if stats is not None:
                    stats["too_small"] = stats.get("too_small", 0) + 1
                continue
            seen_paths.add(gen.path)
            matched.append((line, gen))
    return matched


def build_candidates(
    photos: list[RealPhoto],
    generated: list[GeneratedImage],
    *,
    excluded_caption_keys: set[str],
    used_sources: set[str],
    skipped_sources: set[str],
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    max_upscale: float = DEFAULT_MAX_UPSCALE,
    stats: dict[str, int] | None = None,
) -> list[Candidate]:
    """One candidate per real photo that has a caption line with at least one
    aspect-compatible generation no larger than the photo, where that caption
    is not already paired. ``stats["too_small"]`` (when given) counts photos
    that matched a generation on caption and aspect but were smaller than
    every one of them.

    A photo whose sidecar holds several lines may match several generation
    groups; every matching line's generations are pooled into one candidate
    so the hardest negative is chosen across all of them and the caption the
    pair is exported under is the one that generation was made from."""
    by_key = group_generated_by_key(generated)

    candidates: list[Candidate] = []
    for photo in photos:
        source = _normalize_source_path(photo.path)
        if source in used_sources or source in skipped_sources:
            continue
        photo_stats: dict[str, int] = {}
        matched = usable_generations(
            photo,
            by_key,
            excluded_caption_keys=excluded_caption_keys,
            used_sources=used_sources,
            aspect_tolerance=aspect_tolerance,
            max_upscale=max_upscale,
            stats=photo_stats,
        )
        if not matched:
            if stats is not None and photo_stats.get("too_small"):
                stats["too_small"] = stats.get("too_small", 0) + 1
            continue
        candidates.append(
            Candidate(
                real=photo,
                caption=matched[0][0],
                generated=[gen for _line, gen in matched],
                lines_by_path={gen.path: line for line, gen in matched},
            )
        )
    return candidates


def refilter_candidate(
    candidate: Candidate,
    *,
    excluded_caption_keys: set[str],
    used_sources: set[str],
    is_content_used=None,
) -> bool:
    """Drop generations that became unusable after the candidate was built —
    their caption got a pair elsewhere, or the file (or a byte-identical copy
    of it, via ``is_content_used(path)``) was committed for another photo — and
    report whether anything is left. When the previously picked ``rejected``
    was dropped it is cleared, so the caller knows to re-pick."""
    keep: list[GeneratedImage] = []
    for gen in candidate.generated:
        line = candidate.lines_by_path.get(gen.path, candidate.caption)
        if caption_match_key(line) in excluded_caption_keys:
            continue
        if _normalize_source_path(gen.path) in used_sources:
            continue
        if is_content_used is not None and is_content_used(gen.path):
            continue
        keep.append(gen)
    candidate.generated = keep
    if candidate.rejected is not None and candidate.rejected.path not in {g.path for g in keep}:
        candidate.rejected = None
        candidate.rejected_score = None
        candidate.scored = False
    return bool(keep)


def pick_hardest_negative(candidate: Candidate, score_of, rng: random.Random | None = None) -> Candidate:
    """Set ``candidate.rejected`` to the most photorealistic generation —
    lowest P(AI-generated) — so the pair teaches "even your best isn't a
    photo". ``score_of(path)`` returns a *cached* score or None; the model is
    never run here. When no generation has a score the pick is random (the
    UI shows it as UNSCORED); scored generations always beat unscored ones.
    Ties break by path for determinism."""
    scored = [(score, gen) for gen in candidate.generated if (score := score_of(gen.path)) is not None]
    if scored:
        best_score, best = min(scored, key=lambda item: (item[0], item[1].path))
    else:
        best_score, best = None, (rng or random).choice(candidate.generated)
    candidate.rejected = best
    candidate.rejected_score = best_score
    candidate.scored = best_score is not None
    candidate.caption = candidate.lines_by_path.get(best.path, candidate.caption)
    return candidate


def reroll_negative(candidate: Candidate, score_of, rng: random.Random | None = None) -> bool:
    """Swap ``candidate.rejected`` for a generation not shown yet — the next
    hardest by cached score, or a random one when nothing is scored — so a
    negative that happens to be a different composition can be traded for
    one that isn't. Cycles back to the start once every generation has been
    shown. Returns False when the photo has only one generation."""
    if len(candidate.generated) <= 1:
        return False
    current = candidate.rejected.path if candidate.rejected else None
    if current and current not in candidate.shown_paths:
        candidate.shown_paths.append(current)
    remaining = [g for g in candidate.generated if g.path not in candidate.shown_paths]
    if not remaining:
        candidate.shown_paths = [current] if current else []
        remaining = [g for g in candidate.generated if g.path != current]
    trial = Candidate(
        real=candidate.real, caption=candidate.caption, generated=remaining, lines_by_path=candidate.lines_by_path
    )
    pick_hardest_negative(trial, score_of, rng)
    candidate.rejected = trial.rejected
    candidate.rejected_score = trial.rejected_score
    candidate.scored = trial.scored
    candidate.caption = trial.caption
    candidate.shown_paths.append(trial.rejected.path)
    return True


def _same_file(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def choose_negative(candidate: Candidate, path: str, score_of) -> bool:
    """Make the generation at ``path`` (one of ``candidate.generated``) the
    rejected image — the picker's explicit choice, where :func:`reroll_negative`
    cycles. The previous pick joins the shown history the same way. Returns
    False when ``path`` is not one of this photo's generations."""
    chosen = next((g for g in candidate.generated if _same_file(g.path, path)), None)
    if chosen is None:
        return False
    current = candidate.rejected.path if candidate.rejected else None
    if current and current not in candidate.shown_paths:
        candidate.shown_paths.append(current)
    score = score_of(chosen.path)
    candidate.rejected = chosen
    candidate.rejected_score = score
    candidate.scored = score is not None
    candidate.caption = candidate.lines_by_path.get(chosen.path, candidate.caption)
    if chosen.path not in candidate.shown_paths:
        candidate.shown_paths.append(chosen.path)
    return True


#: Photos the user never wants offered again, per output folder. Session
#: skips ("not now") live in memory and are gone at the end of the session;
#: this file is the opt-in permanent form.
PERMANENT_SKIP_FILENAME = "offpolicy_permanent_skips.json"

#: Earlier builds wrote the very same ``{"skipped": [...]}`` document under a
#: different name. Renaming the file must not silently forget what the user
#: marked — a skip list that quietly empties itself is indistinguishable from
#: "never again doesn't work" — so the old names are still read, folded into
#: the live list, and retired (renamed aside, never deleted) on the next save.
#: Skip files written by an earlier build. ``offpolicy_state.json`` is NOT
#: among them on purpose: it was written when a plain skip was permanent, so
#: its entries are "not now" decisions, not "never again" ones. Adopting them
#: would re-hide the photos the move to session-only skips deliberately
#: brought back. It is retired (renamed aside) rather than read.
LEGACY_PERMANENT_SKIP_FILENAMES: tuple[str, ...] = ()

#: Retired without being adopted — see above.
OBSOLETE_SKIP_FILENAMES = ("offpolicy_state.json",)

#: Suffix a legacy skip file gets once its entries are in the current file.
MIGRATED_SUFFIX = ".migrated"


def _read_skip_file(path: str) -> set[str]:
    """Normalized entries of one ``{"skipped": [...]}`` document. A missing or
    malformed file reads as "none", never as an error."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    entries = data.get("skipped") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return set()
    return {_normalize_source_path(entry) for entry in entries if isinstance(entry, str) and entry}


def legacy_permanent_skip_files(output_dir: str) -> list[str]:
    """Skip files written under a previous name that are still present."""
    return [
        path
        for path in (os.path.join(output_dir, name) for name in LEGACY_PERMANENT_SKIP_FILENAMES)
        if os.path.isfile(path)
    ]


def load_permanent_skips(output_dir: str) -> set[str]:
    """Normalized source paths the user permanently skipped for this output
    folder, including any still-unmigrated file from an earlier build whose
    entries meant "never again" (see :data:`LEGACY_PERMANENT_SKIP_FILENAMES`)."""
    skips = _read_skip_file(os.path.join(output_dir, PERMANENT_SKIP_FILENAME))
    for path in legacy_permanent_skip_files(output_dir):
        skips |= _read_skip_file(path)
    return skips


def save_permanent_skips(output_dir: str, skips: set[str]) -> None:
    """Persist the permanent skip list (atomic replace, sorted for a stable
    diff)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, PERMANENT_SKIP_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"skipped": sorted(skips)}, f, indent=2)
    os.replace(tmp, path)


def retire_obsolete_skip_files(output_dir: str) -> list[str]:
    """Rename aside any skip file this build no longer honours, returning the
    files retired. Nothing is deleted and nothing is adopted: an
    ``offpolicy_state.json`` from the build where a plain skip persisted holds
    "not now" decisions, and those photos are meant to come back."""
    retired: list[str] = []
    for name in OBSOLETE_SKIP_FILENAMES:
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            os.replace(path, path + MIGRATED_SUFFIX)
        except OSError:
            continue
        retired.append(path)
    return retired


def migrate_permanent_skips(output_dir: str) -> int:
    """Fold any legacy *permanent* skip file into the current one and retire it
    (renamed aside, never deleted), returning how many entries that adopted.

    Call this once per session start, before anything can lift a skip: while a
    legacy file is still on disk :func:`load_permanent_skips` keeps re-reading
    it, so an undone skip would be resurrected at the next start. A failure to
    write is not fatal — the loader still unions both files, so nothing is
    lost, and the migration is simply retried next time."""
    legacy_files = legacy_permanent_skip_files(output_dir)
    if not legacy_files:
        return 0
    current = _read_skip_file(os.path.join(output_dir, PERMANENT_SKIP_FILENAME))
    adopted: set[str] = set()
    for path in legacy_files:
        adopted |= _read_skip_file(path) - current
    try:
        save_permanent_skips(output_dir, current | adopted)
    except OSError:
        return len(adopted)
    for path in legacy_files:
        with suppress(OSError):
            os.replace(path, path + MIGRATED_SUFFIX)
    return len(adopted)


def is_within(path: str, folder: str) -> bool:
    """True when ``path`` is ``folder`` or lies underneath it (case-folded,
    absolute comparison, so ``C:/Data`` and ``c:\\data\\x.jpg`` agree)."""
    try:
        path_n = os.path.normcase(os.path.abspath(path))
        folder_n = os.path.normcase(os.path.abspath(folder))
        return os.path.commonpath([path_n, folder_n]) == folder_n
    except ValueError:  # different drives
        return False


def sidecar_has_other_images(caption_path: str, except_image: str) -> bool:
    """True when another supported image shares the sidecar's stem — ``a.jpg``
    and ``a.png`` both read ``a.txt``, so emptying the sidecar for one of them
    also un-captions the other."""
    stem = os.path.splitext(caption_path)[0]
    except_n = os.path.normcase(os.path.abspath(except_image))
    for ext in supported_image_extensions():
        for candidate in (stem + ext, stem + ext.upper()):
            if os.path.normcase(os.path.abspath(candidate)) == except_n:
                continue
            if os.path.isfile(candidate):
                return True
    return False


TRASH_DIRNAME = ".offpolicy_trash"


def purge_trash(root: str) -> int:
    """Permanently delete everything in ``<root>/.offpolicy_trash``. Returns
    the number of files removed."""
    trash_root = os.path.join(root, TRASH_DIRNAME)
    if not os.path.isdir(trash_root):
        return 0
    count = 0
    for _walk_root, _dirs, files in os.walk(trash_root):
        count += len(files)
    shutil.rmtree(trash_root, ignore_errors=True)
    return count


def remove_caption_line(caption_path: str, line: str, root: str) -> dict:
    """Remove every line of the sidecar whose stripped text equals ``line``,
    keeping the other lines byte-for-byte (line endings and BOM included).
    When nothing usable remains the sidecar is deleted, so the photo simply
    drops out of the captioned set. The original file is copied into
    ``<root>/.offpolicy_trash`` first and its text returned, so the change
    can be undone. Refuses paths outside ``root``."""
    if not is_within(caption_path, root):
        raise ValueError(f"Refusing to edit outside the real photo folder: {caption_path}")
    with open(caption_path, encoding="utf-8", newline="") as f:
        raw = f.read()
    bom = raw.startswith("\ufeff")
    body = raw[1:] if bom else raw
    nl = "\r\n" if "\r\n" in body else "\n"
    lines = body.split(nl)
    target = line.strip()
    kept = [entry for entry in lines if entry.strip() != target]
    removed = len(lines) - len(kept)
    if removed == 0:
        return {"removed": 0, "removed_file": False, "original": raw, "backup": None, "remaining": []}
    # Backup copy for undo / inspection; purged with the rest of the trash.
    trash_root = os.path.join(root, TRASH_DIRNAME)
    backup = os.path.join(trash_root, os.path.relpath(caption_path, root))
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    if os.path.exists(backup):
        stem, ext = os.path.splitext(backup)
        i = 1
        while os.path.exists(f"{stem}_{i}{ext}"):
            i += 1
        backup = f"{stem}_{i}{ext}"
    with open(backup, "w", encoding="utf-8", newline="") as f:
        f.write(raw)
    remaining = [entry.strip() for entry in kept if entry.strip()]
    if remaining:
        with open(caption_path, "w", encoding="utf-8", newline="") as f:
            f.write(("\ufeff" if bom else "") + nl.join(kept))
        removed_file = False
    else:
        os.remove(caption_path)
        removed_file = True
    return {"removed": removed, "removed_file": removed_file, "original": raw, "backup": backup, "remaining": remaining}


def restore_caption_file(caption_path: str, original: str) -> None:
    """Write a sidecar back exactly as :func:`remove_caption_line` found it."""
    os.makedirs(os.path.dirname(caption_path) or ".", exist_ok=True)
    with open(caption_path, "w", encoding="utf-8", newline="") as f:
        f.write(original)


# ---------------------------------------------------------------------------
# Persistent real-photo index: size + caption lines per photo, validated by
# (size, mtime_ns) of both the image and its sidecar. A warm pass over 96k
# photos then costs directory listings and dict lookups only — on Windows
# ``os.DirEntry.stat()`` is served from the listing, so no per-file syscall —
# instead of an open() of every sidecar and a stat() of every image.
# ---------------------------------------------------------------------------

REAL_INDEX_FILENAME = ".offpolicy_real_index.json"


def walk_entries_skipping_dotted(folder: str):
    """Like :func:`walk_skipping_dotted` but yields ``(root, {name: DirEntry})``
    so callers can use the entries' cached ``stat()`` (free on Windows) and
    find a sidecar in the same listing without touching the filesystem
    again. Dot-prefixed subdirectories are pruned; directories are visited in
    sorted order."""
    stack = [folder]
    while stack:
        root = stack.pop()
        try:
            it = os.scandir(root)
        except OSError:
            continue
        files: dict[str, os.DirEntry] = {}
        dirs: list[str] = []
        with it:
            for entry in it:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    if not entry.name.startswith("."):
                        dirs.append(entry.path)
                else:
                    files[entry.name] = entry
        yield root, files
        stack.extend(sorted(dirs, reverse=True))


def entry_stamp(entry) -> tuple[int, int] | None:
    """(size, mtime_ns) of a DirEntry, or None when it can't be read."""
    try:
        st = entry.stat()
    except OSError:
        return None
    return int(st.st_size), int(st.st_mtime_ns)


def sidecar_map(files: dict) -> dict[str, object]:
    """``{casefolded name: DirEntry}`` of the ``.txt`` files in one directory
    listing. Built once per directory so sidecar lookup is O(1): a linear
    case-insensitive scan per image is O(n²) in a flat folder and stalls for
    minutes on a 96k-photo dataset."""
    return {name.casefold(): entry for name, entry in files.items() if name.lower().endswith(".txt")}


def find_sidecar_entry(files: dict, image_name: str, sidecars: dict[str, object] | None = None):
    """The ``.txt`` sidecar entry for ``image_name`` from one directory
    listing — exact name first, then case-insensitive so ``A.JPG`` +
    ``a.txt`` pair up on Windows the way the trainer pairs them. Pass the
    directory's :func:`sidecar_map` when looking up many images."""
    stem = os.path.splitext(image_name)[0]
    exact = files.get(stem + ".txt")
    if exact is not None:
        return exact
    if sidecars is None:
        sidecars = sidecar_map(files)
    return sidecars.get((stem + ".txt").casefold())


class RealPhotoIndex:
    """JSON-backed index of the real photo folder for the off-policy builder.

    One entry per image path: the image's and sidecar's (size, mtime_ns), the
    pixel size, and the caption lines. Lines are stored so a rescan never
    opens a sidecar; a photo whose sidecar has no usable line is stored with
    ``lines == []`` so it is not re-read either. ``save`` keeps only the keys
    seen in the latest walk, which prunes deleted files without a stat
    sweep."""

    _VERSION = 1

    def __init__(self, path: str):
        self._path = path
        self._entries: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and data.get("version") == self._VERSION
                and isinstance(data.get("entries"), dict)
            ):
                self._entries = data["entries"]
        except (OSError, json.JSONDecodeError):
            pass

    @staticmethod
    def key(path: str) -> str:
        return _normalize_source_path(path)

    def get(self, key: str, image_stamp: tuple[int, int], sidecar_stamp: tuple[int, int]) -> dict | None:
        entry = self._entries.get(key)
        if (
            entry is not None
            and tuple(entry.get("img", ())) == tuple(image_stamp)
            and tuple(entry.get("txt", ())) == tuple(sidecar_stamp)
        ):
            self.hits += 1
            return entry
        self.misses += 1
        return None

    def put(
        self,
        key: str,
        image_stamp: tuple[int, int],
        sidecar_stamp: tuple[int, int],
        size: tuple[int, int] | None,
        lines: list[str],
    ) -> dict:
        entry = {
            "img": list(image_stamp),
            "txt": list(sidecar_stamp),
            "w": int(size[0]) if size else 0,
            "h": int(size[1]) if size else 0,
            "lines": list(lines),
        }
        self._entries[key] = entry
        return entry

    def save(self, keep_keys: set[str] | None = None) -> None:
        if keep_keys is not None:
            self._entries = {k: v for k, v in self._entries.items() if k in keep_keys}
        tmp = self._path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": self._VERSION, "entries": self._entries}, f)
            os.replace(tmp, self._path)
        except OSError:
            pass


def photo_from_entry(path: str, caption_path: str, entry: dict) -> RealPhoto | None:
    """A RealPhoto from an index entry, or None when the entry records a
    photo that is unusable (no caption lines, unreadable size)."""
    lines = entry.get("lines") or []
    w, h = int(entry.get("w", 0)), int(entry.get("h", 0))
    if not lines or w <= 0 or h <= 0:
        return None
    return RealPhoto(path, caption_path, list(lines), w, h)


# ---------------------------------------------------------------------------
# "What should I generate next?" — a caption from the real set that has no
# usable generation yet, with the aspect to generate it at.
# ---------------------------------------------------------------------------


#: Aspect ratios the generator can actually produce, landscape form; each is
#: also offered inverted (portrait). Suggestions only ever name one of these —
#: advising a photo's exact aspect is useless when it can't be generated.
GENERATABLE_ASPECTS: tuple[tuple[int, int], ...] = ((1, 1), (4, 3), (3, 2), (8, 5), (16, 9), (21, 9))


def generatable_aspect_labels() -> list[str]:
    """Every generatable label, landscape and portrait, widest first."""
    labels: list[str] = []
    for num, den in sorted(GENERATABLE_ASPECTS, key=lambda r: -r[0] / r[1]):
        labels.append(f"{num}:{den}")
        if num != den:
            labels.append(f"{den}:{num}")
    return labels


def parse_aspect_label(label: str) -> tuple[int, int] | None:
    try:
        num, den = (int(part) for part in str(label).split(":"))
    except (ValueError, AttributeError):
        return None
    return (num, den) if num > 0 and den > 0 else None


def nearest_generatable_aspect(width: int, height: int) -> str:
    """The generatable label (``3:2``, ``9:16``, ...) closest to this image's
    aspect, measured in log-aspect so a ratio and its inverse are equidistant.
    Falls back to ``1:1`` for a non-positive dimension."""
    if width <= 0 or height <= 0:
        return "1:1"
    target = math.log(width / height)
    best_label, best_gap = "1:1", None
    for label in generatable_aspect_labels():
        num, den = parse_aspect_label(label)
        gap = abs(math.log(num / den) - target)
        if best_gap is None or gap < best_gap - 1e-12:
            best_label, best_gap = label, gap
    return best_label


def aspect_size_for_label(label: str, base_resolution: int = 1024, multiple: int = 16) -> tuple[int, int]:
    """Pixel size for a ratio label at the ``base_resolution``² budget, snapped
    to ``multiple`` (16 is Z-Image's latent patch: 8x VAE times a 2-wide patch)."""
    parsed = parse_aspect_label(label)
    if parsed is None:
        return base_resolution, base_resolution
    num, den = parsed
    aspect = num / den
    area = float(base_resolution * base_resolution)
    gen_w = math.sqrt(area * aspect)
    gen_h = gen_w / aspect
    snap = lambda v: max(multiple, int(round(v / multiple)) * multiple)  # noqa: E731
    return snap(gen_w), snap(gen_h)


#: Generation sizes must be a multiple of this: the VAE downsamples by 8
#: and the transformer takes 2-wide patches, so 16 is the smallest step a
#: model can actually render.
GENERATION_SIZE_MULTIPLE = 16


def snap_down(value: float, multiple: int = GENERATION_SIZE_MULTIPLE) -> int:
    """Largest multiple of ``multiple`` at or below ``value`` (never zero)."""
    return max(multiple, int(value // multiple) * multiple)


def exact_size_for_small_photo(
    width: int, height: int, base_resolution: int = 1024, multiple: int = GENERATION_SIZE_MULTIPLE
) -> tuple[int, int] | None:
    """The photo's own size, for a photo smaller than the pixel budget.

    A photo below ``base_resolution``² has nothing to gain from a
    budget-sized generation: it would have to be upscaled to meet it, and
    the pair would then carry a resampling difference on top of the realism
    one. Generating at the photo's own dimensions keeps both sides native
    and leaves the trainer's scale-crop to handle the rest.

    Snapped *down* to a renderable multiple, so the generation is never
    larger than the photo. ``None`` when the photo is big enough for the
    normal budget-sized path."""
    if width <= 0 or height <= 0:
        return None
    if width * height >= base_resolution * base_resolution:
        return None
    return snap_down(width, multiple), snap_down(height, multiple)


def suggested_generation_size(
    width: int, height: int, base_resolution: int = 1024, multiple: int = 16
) -> tuple[int, int]:
    """Size to generate for this photo: the pixel budget of
    ``base_resolution``² at the **nearest generatable aspect**
    (:data:`GENERATABLE_ASPECTS`), snapped to ``multiple``.

    The photo's exact aspect would be ideal — the loader cover-crops the
    generation with the photo's own framing, so a matching aspect loses
    nothing — but the generator only emits a fixed set of ratios, so the
    nearest one is the actionable advice. Whatever gap remains shows up as
    the pair's aspect delta and is checked against the session tolerance."""
    # A photo below the budget generates at its own size instead.
    exact = exact_size_for_small_photo(width, height, base_resolution, multiple)
    if exact is not None:
        return exact
    return aspect_size_for_label(nearest_generatable_aspect(width, height), base_resolution, multiple)


def suggestion_pool(
    photos: list[RealPhoto],
    generated: list[GeneratedImage] | dict[str, list[GeneratedImage]],
    *,
    excluded_caption_keys: set[str],
    used_sources: set[str],
    skipped_sources: set[str],
    base_resolution: int = 1024,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    max_upscale: float = DEFAULT_MAX_UPSCALE,
) -> list[tuple[RealPhoto, list[str]]]:
    """(photo, caption lines worth generating for) for every photo that could
    still become a pair.

    Effort goes where it can still produce a pair: a photo is eligible only
    while it has *no* usable generation at all (one pair per photo, so a second
    generation for a photo that already has a candidate adds nothing), and a
    line is eligible while it has no aspect-compatible generation — a line
    whose only generations are the wrong orientation is offered again, with
    the right size this time. Photos already paired or skipped, captions
    paired elsewhere, and photos smaller than the size that would be
    suggested for them (they could never cover such a generation) are passed
    over."""
    by_key = generated if isinstance(generated, dict) else group_generated_by_key(generated)
    pool: list[tuple[RealPhoto, list[str]]] = []
    for photo in photos:
        source = _normalize_source_path(photo.path)
        if source in used_sources or source in skipped_sources:
            continue
        real_size = (photo.width, photo.height)
        if not real_covers_generation(
            real_size, suggested_generation_size(photo.width, photo.height, base_resolution), max_upscale
        ):
            continue
        lines: list[str] = []
        covered = False
        for line in photo.lines:
            key = caption_match_key(line)
            if not key or key in excluded_caption_keys:
                continue
            gens = by_key.get(key, ())
            if any(aspect_within_tolerance(real_size, (g.width, g.height), aspect_tolerance) for g in gens):
                covered = True
                break
            lines.append(line)
        if covered or not lines:
            continue
        pool.append((photo, lines))
    return pool


def filter_pool_by_keywords(
    pool: list[tuple[RealPhoto, list[str]]], keywords: list[str]
) -> list[tuple[RealPhoto, list[str]]]:
    """Narrow a suggestion pool to caption lines containing every keyword
    (case-insensitive substring, the curation tool's ``parse_caption_keywords``
    convention: comma-separated terms, all required). A photo stays only
    while at least one of its eligible lines matches; no keywords keeps the
    pool as is."""
    if not keywords:
        return pool
    narrowed: list[tuple[RealPhoto, list[str]]] = []
    for photo, lines in pool:
        matching = [line for line in lines if all(k in line.casefold() for k in keywords)]
        if matching:
            narrowed.append((photo, matching))
    return narrowed


def bucket_generation_size(label: str, base_resolution: int = 1024) -> tuple[int, int]:
    """Generation size for a generatable ratio label (``3:2`` ⇒ width:height)
    at the ``base_resolution``² budget — one size to batch that whole group
    of captions at."""
    return aspect_size_for_label(label, base_resolution)


def bucket_summary(pool: list[tuple[RealPhoto, list[str]]], base_resolution: int = 1024) -> dict[str, dict]:
    """Per **generatable** aspect ratio: how many pool photos are nearest to
    it and the one size to generate them all at. Grouping by what the
    generator can emit (rather than by trainer bucket) means every row is a
    batch you can actually run. Sorted by count, largest first."""
    counts: dict[str, int] = {}
    for photo, _lines in pool:
        label = nearest_generatable_aspect(photo.width, photo.height)
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        label: {"photos": count, "generate_size": list(bucket_generation_size(label, base_resolution))}
        for label, count in ordered
    }


def _suggestion_entry(
    photo: RealPhoto, line: str, base_resolution: int, aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE
) -> dict:
    gen_w, gen_h = suggested_generation_size(photo.width, photo.height, base_resolution)
    delta = aspect_mismatch((photo.width, photo.height), (gen_w, gen_h))
    exact = exact_size_for_small_photo(photo.width, photo.height, base_resolution) is not None
    return {
        "real_path": photo.path,
        "caption": line,
        "caption_lines": list(photo.lines),
        "real_size": [photo.width, photo.height],
        "aspect_bucket": trainer_bucket_key(photo.width, photo.height),
        "generate_aspect": nearest_generatable_aspect(photo.width, photo.height),
        "generate_size": [gen_w, gen_h],
        # True when the size is the photo's own rather than a budget-sized
        # bucket, because the photo is smaller than the budget.
        "exact_size": exact,
        # How far the nearest generatable ratio still is from the photo, and
        # whether a generation at that size would pass the session's check.
        "aspect_delta": delta,
        "within_tolerance": delta is not None and delta <= aspect_tolerance,
    }


def suggest_captions(
    pool: list[tuple[RealPhoto, list[str]]],
    rng: random.Random,
    count: int = 1,
    aspect_bucket: str | None = None,
    base_resolution: int = 1024,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
) -> list[dict]:
    """Up to ``count`` random (photo, caption line) entries from the pool —
    distinct photos, one random eligible line each — optionally restricted to
    photos whose nearest generatable ratio is ``aspect_bucket`` (``"3:2"``).
    Re-rollable: every call is a fresh draw."""
    eligible = pool
    if aspect_bucket:
        eligible = [(p, lines) for p, lines in pool if nearest_generatable_aspect(p.width, p.height) == aspect_bucket]
    if not eligible or count <= 0:
        return []
    picked = rng.sample(eligible, min(count, len(eligible)))
    return [_suggestion_entry(photo, rng.choice(lines), base_resolution, aspect_tolerance) for photo, lines in picked]


def suggest_caption(
    photos: list[RealPhoto],
    generated: list[GeneratedImage] | dict[str, list[GeneratedImage]],
    *,
    excluded_caption_keys: set[str],
    used_sources: set[str],
    skipped_sources: set[str],
    rng: random.Random,
    base_resolution: int = 1024,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    max_upscale: float = DEFAULT_MAX_UPSCALE,
) -> dict | None:
    """One random suggestion (see :func:`suggestion_pool`) with the size to
    generate at, plus ``remaining`` = eligible photos; ``None`` when nothing
    is left to generate for."""
    pool = suggestion_pool(
        photos,
        generated,
        excluded_caption_keys=excluded_caption_keys,
        used_sources=used_sources,
        skipped_sources=skipped_sources,
        base_resolution=base_resolution,
        aspect_tolerance=aspect_tolerance,
        max_upscale=max_upscale,
    )
    entries = suggest_captions(pool, rng, 1, base_resolution=base_resolution, aspect_tolerance=aspect_tolerance)
    if not entries:
        return None
    return entries[0] | {"remaining": len(pool)}


# ---------------------------------------------------------------------------
# "What do I generate for this photo?" — look a photo up by name in the
# already-built index and report the size to generate at.
# ---------------------------------------------------------------------------

LOOKUP_LIMIT = 20


def _lookup_key(path: str) -> str:
    return path.replace("\\", "/").casefold()


def find_photos_by_name(photos: list[RealPhoto], query: str, limit: int = LOOKUP_LIMIT) -> list[RealPhoto]:
    """Photos whose file name matches ``query`` — the bare name, the name
    without its extension, or a trailing path fragment (``sub/dir/name.jpg``).

    Matching is against the in-memory index only, by string comparison: the
    query is never joined onto a folder or handed to the filesystem, so a
    crafted name cannot reach outside the indexed tree. Case-insensitive, with
    either slash accepted. Results come back in path order, capped at
    ``limit``."""
    q = _lookup_key((query or "").strip().strip("/"))
    if not q:
        return []
    q_stem = os.path.splitext(q)[0]
    matches: list[RealPhoto] = []
    for photo in photos:
        key = _lookup_key(photo.path)
        base = os.path.basename(key)
        if base == q or os.path.splitext(base)[0] == q_stem and "/" not in q or key.endswith("/" + q):
            matches.append(photo)
            if len(matches) >= limit:
                break
    return matches


def photo_generation_brief(
    photo: RealPhoto,
    by_key: dict[str, list[GeneratedImage]],
    *,
    excluded_caption_keys: set[str],
    used_sources: set[str],
    skipped_sources: set[str],
    base_resolution: int = 1024,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    max_upscale: float = DEFAULT_MAX_UPSCALE,
) -> dict:
    """Everything the user needs to generate for one photo: the size to
    generate at (photo's own aspect, ``base_resolution``² budget), its trainer
    bucket, its caption lines, and where it stands — already paired, skipped,
    already covered by a usable generation, or too small to ever be paired at
    the suggested size."""
    source = _normalize_source_path(photo.path)
    gen_w, gen_h = suggested_generation_size(photo.width, photo.height, base_resolution)
    matched = usable_generations(
        photo,
        by_key,
        excluded_caption_keys=excluded_caption_keys,
        used_sources=used_sources,
        aspect_tolerance=aspect_tolerance,
        max_upscale=max_upscale,
    )
    covered_lines = {line for line, _gen in matched}
    return {
        "real_path": photo.path,
        "real_size": [photo.width, photo.height],
        "aspect_bucket": trainer_bucket_key(photo.width, photo.height),
        "generate_aspect": nearest_generatable_aspect(photo.width, photo.height),
        "generate_size": [gen_w, gen_h],
        "exact_size": exact_size_for_small_photo(photo.width, photo.height, base_resolution) is not None,
        "aspect_delta": aspect_mismatch((photo.width, photo.height), (gen_w, gen_h)),
        "caption_lines": list(photo.lines),
        "lines_needing_generation": [
            line
            for line in photo.lines
            if line not in covered_lines
            and caption_match_key(line)
            and caption_match_key(line) not in excluded_caption_keys
        ],
        "paired": source in used_sources,
        "skipped": source in skipped_sources,
        "has_usable_generation": bool(matched),
        "too_small_for_suggested_size": not real_covers_generation(
            (photo.width, photo.height), (gen_w, gen_h), max_upscale
        ),
    }
