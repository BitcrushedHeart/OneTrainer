"""Text Alignment dataset: pick the generations whose on-image text came out
*wrong* and package them for the offline text reward.

The reward step re-noises a finished image and pushes the adapter to correct
the spelling from there, so an image only earns its place when there is
something to correct: the caption asked for text, the OCR found that text
rendered but misspelt, and the training-time detector can box at least one
of the target words (no box, no gradient). Verbatim renderings teach nothing;
absent ones give the reward nothing to push on.

Layout of the output folder, one sample per stem (the trainer's contract):

* ``<stem>.png``  the image (converted to PNG when the source is not one)
* ``<stem>.txt``  first line = the full prompt the image was generated with
* ``<stem>.json`` ``prompt``, ``text``, ``words``, ``ocr_reading``,
  ``text_score``, ``boxed_words``, ``source_image``, ``sha256``

Plus ``registry.json`` (sha256 of every source ever admitted) so a rescan of
the same SwarmUI folder never re-admits an image.

The model halves are injected: ``read_text(path) -> str | None`` (the
curation OCR, PaddleOCR-VL / GOT-OCR2) and ``count_boxed(path, words) -> int``
(the training-time docTR detector + recognizer). Everything else is pure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from modules.util.swarm_generate_util import _params_from_image
from modules.util.text_adherence_util import caption_text_spans, score_text

REGISTRY_FILE = "registry.json"
#: Per-source-image scan results keyed by sha256, so a cancelled or repeated
#: scan never re-runs the OCR for an image it has already classified.
SCAN_CACHE_FILE = "scan_cache.json"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

#: Decision reasons, in the order the pipeline checks them.
REASON_KEEP = "keep"
REASON_NO_METADATA = "no_metadata"
REASON_NO_TEXT_SPAN = "no_text_span"
REASON_DUPLICATE = "duplicate"
REASON_ALREADY_IN_DATASET = "already_in_dataset"
REASON_UNREADABLE = "unreadable"
REASON_ABSENT = "absent"
REASON_VERBATIM = "verbatim"
REASON_NO_BOXES = "no_boxes"
REASON_READER_DISAGREE = "reader_disagree"
REASONS = (
    REASON_KEEP,
    REASON_NO_METADATA,
    REASON_NO_TEXT_SPAN,
    REASON_DUPLICATE,
    REASON_ALREADY_IN_DATASET,
    REASON_UNREADABLE,
    REASON_ABSENT,
    REASON_VERBATIM,
    REASON_NO_BOXES,
    REASON_READER_DISAGREE,
)

_TEXT_READING_RE = re.compile(r'text reading\s+"([^"]+)"', re.IGNORECASE)


@dataclass
class Candidate:
    source_image: str
    sha256: str = ""
    prompt: str = ""
    text: str = ""
    words: list[str] = field(default_factory=list)
    ocr_reading: str | None = None
    text_score: float | None = None
    verdicts: list[str] = field(default_factory=list)
    boxed_words: int = 0
    reward_reading: str | None = None  #: what the training-time reader (PARSeq) read in the boxes
    reader_agreement: float | None = None  #: curation OCR vs training reader, mean word edit ratio
    reason: str = REASON_NO_METADATA

    @property
    def keep(self) -> bool:
        return self.reason == REASON_KEEP

    def to_record(self) -> dict:
        d = asdict(self)
        d["keep"] = self.keep
        return d


@dataclass
class ScanPlan:
    source_folder: str
    output_folder: str
    candidates: list[Candidate] = field(default_factory=list)
    cancelled: bool = False

    @property
    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(REASONS, 0)
        for c in self.candidates:
            out[c.reason] = out.get(c.reason, 0) + 1
        return out

    @property
    def kept(self) -> list[Candidate]:
        return [c for c in self.candidates if c.keep]


# --------------------------------------------------------------------------- #
# Pure pieces
# --------------------------------------------------------------------------- #


def list_images(folder: str, recursive: bool = True) -> list[str]:
    """Image files under ``folder`` (dot-folders skipped), sorted by path."""
    out: list[str] = []
    if not os.path.isdir(folder):
        return out
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        out.extend(os.path.join(root, name) for name in files if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS)
        if not recursive:
            break
    out.sort()
    return out


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_from_image(path: str) -> str | None:
    """The positive prompt SwarmUI stamped into the image, or ``None``."""
    params = _params_from_image(path)
    prompt = params.get("prompt") if isinstance(params, dict) else None
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def target_text(prompt: str) -> str | None:
    """The string the caption asks to be rendered. Prefers the curation
    extractor (quoted strings, "reads"/"printed with" cues); falls back to the
    reward generator's own ``text reading "..."`` template."""
    spans = caption_text_spans(prompt)
    if spans:
        return " ".join(s.strip() for s in spans if s.strip()) or None
    m = _TEXT_READING_RE.search(prompt or "")
    if m:
        return m.group(1).strip() or None
    return None


def decide(
    verdicts: Sequence[str],
    boxed_words: int,
    *,
    keep_absent: bool = False,
    keep_verbatim: bool = False,
    min_boxed_words: int = 1,
    reader_agreement: float | None = None,
    min_agreement: float = 0.0,
) -> str:
    """The decision rule. Default: keep only "rendered but wrong" images that
    the training-time detector can box.

    * every span verbatim -> nothing to learn (unless ``keep_verbatim``)
    * every span absent -> nothing to push on (unless ``keep_absent``)
    * fewer than ``min_boxed_words`` boxed -> no gradient would flow
    * the two readers disagree (``reader_agreement`` below ``min_agreement``)
      -> the gradient would point at what the small reader hallucinated, not
      at the misspelling the judge saw. ``None`` (not measured) never rejects.
    """
    if not verdicts:
        return REASON_UNREADABLE
    if all(v == "verbatim" for v in verdicts) and not keep_verbatim:
        return REASON_VERBATIM
    if all(v == "absent" for v in verdicts) and not keep_absent:
        return REASON_ABSENT
    if "wrong" not in verdicts and not (keep_absent or keep_verbatim):
        # mixed absent/verbatim with no wrong span: still nothing to correct
        return REASON_ABSENT if "absent" in verdicts else REASON_VERBATIM
    if boxed_words < min_boxed_words:
        return REASON_NO_BOXES
    if min_agreement > 0.0 and reader_agreement is not None and reader_agreement < min_agreement:
        return REASON_READER_DISAGREE
    return REASON_KEEP


def load_registry(output_folder: str) -> set[str]:
    path = os.path.join(output_folder, REGISTRY_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return set()
    hashes = data.get("sha256", []) if isinstance(data, dict) else data
    return {h for h in hashes if isinstance(h, str)}


def save_registry(output_folder: str, hashes: set[str]) -> None:
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, REGISTRY_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sha256": sorted(hashes)}, f, indent=2)


def existing_stems(output_folder: str) -> set[str]:
    if not os.path.isdir(output_folder):
        return set()
    return {
        os.path.splitext(n)[0]
        for n in os.listdir(output_folder)
        if n.lower().endswith(".json") and n not in (REGISTRY_FILE, SCAN_CACHE_FILE)
    }


# --------------------------------------------------------------------------- #
# Scan (dry run) and apply
# --------------------------------------------------------------------------- #


def scan_dataset(
    source_folder: str,
    output_folder: str,
    *,
    read_text: Callable[[str], str | None],
    count_boxed: Callable[[str, Sequence[str]], int],
    recursive: bool = True,
    keep_absent: bool = False,
    keep_verbatim: bool = False,
    min_boxed_words: int = 1,
    min_agreement: float = 0.0,
    limit: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ScanPlan:
    """Classify every image under ``source_folder``. Nothing is written.

    The expensive halves run only for images that passed the cheap checks
    (metadata, text span, duplicates), and the detector only when the OCR
    verdict already says "wrong"."""
    plan = ScanPlan(source_folder=source_folder, output_folder=output_folder)
    registry = load_registry(output_folder)
    stems = existing_stems(output_folder)
    cache = load_scan_cache(output_folder)
    cache_dirty = 0
    seen: set[str] = set()
    paths = list_images(source_folder, recursive=recursive)
    if limit > 0:
        paths = paths[:limit]
    total = len(paths)
    for i, path in enumerate(paths):
        if should_stop is not None and should_stop():
            plan.cancelled = True
            break
        if progress is not None:
            progress(i, total, path)
        cand = Candidate(source_image=path)
        plan.candidates.append(cand)

        prompt = prompt_from_image(path)
        if prompt is None:
            cand.reason = REASON_NO_METADATA
            continue
        cand.prompt = prompt
        text = target_text(prompt)
        if not text:
            cand.reason = REASON_NO_TEXT_SPAN
            continue
        cand.text = text
        cand.words = [w for w in text.split(" ") if w]

        try:
            cand.sha256 = sha256_of_file(path)
        except OSError:
            cand.reason = REASON_UNREADABLE
            continue
        if cand.sha256 in seen:
            cand.reason = REASON_DUPLICATE
            continue
        seen.add(cand.sha256)
        if cand.sha256 in registry or dataset_stem(path, cand.sha256) in stems:
            cand.reason = REASON_ALREADY_IN_DATASET
            continue

        cached = cache.get(cand.sha256)
        if cached is not None and cached.get("text") == text:
            cand.ocr_reading = cached.get("ocr_reading")
            cand.text_score = cached.get("text_score")
            cand.verdicts = list(cached.get("verdicts") or [])
            cand.boxed_words = int(cached.get("boxed_words") or 0)
            cand.reward_reading = cached.get("reward_reading")
            cand.reader_agreement = cached.get("reader_agreement")
            if cand.ocr_reading is None:
                cand.reason = REASON_UNREADABLE
                continue
            cand.reason = decide(
                cand.verdicts,
                cand.boxed_words,
                keep_absent=keep_absent,
                keep_verbatim=keep_verbatim,
                min_boxed_words=min_boxed_words,
                reader_agreement=cand.reader_agreement,
                min_agreement=min_agreement,
            )
            continue

        try:
            reading = read_text(path)
        except Exception:
            reading = None
        if reading is None:
            cand.reason = REASON_UNREADABLE
            cache[cand.sha256] = {"text": text, "ocr_reading": None}
            cache_dirty += 1
            continue
        cand.ocr_reading = reading
        scored = score_text([text], reading)
        cand.text_score = scored["text"]
        cand.verdicts = [m.verdict for m in scored["matches"]]

        reason = decide(
            cand.verdicts,
            min_boxed_words,  # provisional: boxes only counted when needed
            keep_absent=keep_absent,
            keep_verbatim=keep_verbatim,
            min_boxed_words=min_boxed_words,
        )
        if reason == REASON_KEEP:
            try:
                boxed = count_boxed(path, cand.words)
            except Exception:
                boxed = 0
            if isinstance(boxed, tuple):
                cand.boxed_words = int(boxed[0])
                cand.reward_reading = " ".join(boxed[1]) if boxed[1] else ""
                cand.reader_agreement = reader_agreement(reading, list(boxed[1]))
            else:
                cand.boxed_words = int(boxed)
            reason = decide(
                cand.verdicts,
                cand.boxed_words,
                keep_absent=keep_absent,
                keep_verbatim=keep_verbatim,
                min_boxed_words=min_boxed_words,
                reader_agreement=cand.reader_agreement,
                min_agreement=min_agreement,
            )
        cand.reason = reason
        cache[cand.sha256] = {
            "text": text,
            "ocr_reading": cand.ocr_reading,
            "text_score": cand.text_score,
            "verdicts": list(cand.verdicts),
            "boxed_words": cand.boxed_words,
            "reward_reading": cand.reward_reading,
            "reader_agreement": cand.reader_agreement,
        }
        cache_dirty += 1
        if cache_dirty >= 25:
            save_scan_cache(output_folder, cache)
            cache_dirty = 0
    if cache_dirty:
        save_scan_cache(output_folder, cache)
    if progress is not None and not plan.cancelled:
        progress(total, total, "")
    return plan


def load_scan_cache(output_folder: str) -> dict[str, dict]:
    path = os.path.join(output_folder, SCAN_CACHE_FILE) if output_folder else ""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_scan_cache(output_folder: str, cache: dict[str, dict]) -> None:
    if not output_folder:
        return
    os.makedirs(output_folder, exist_ok=True)
    tmp = os.path.join(output_folder, SCAN_CACHE_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(output_folder, SCAN_CACHE_FILE))


def reader_agreement(ocr_reading: str | None, reward_words: Sequence[str]) -> float | None:
    """How much the curation OCR (one string for the whole image) and the
    training-time reader (one string per detected box) agree: for every
    reward word, the best edit ratio against any token of the OCR reading,
    averaged. 1.0 = the two readers saw the same words; low values flag the
    stylised surfaces where the gradient reader is the weak link."""
    from modules.util.caption_text_extract import levenshtein_ratio, normalize_for_match

    if not ocr_reading or not reward_words:
        return None
    tokens = [normalize_for_match(t) for t in ocr_reading.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    scores = []
    for w in reward_words:
        nw = normalize_for_match(w)
        if not nw:
            continue
        scores.append(max(levenshtein_ratio(nw, t) for t in tokens))
    return sum(scores) / len(scores) if scores else None


def dataset_stem(source_path: str, sha256: str) -> str:
    """Stable, collision-free stem: source basename plus a hash prefix."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", base)[:60]
    return f"{base}_{sha256[:10]}"


def apply_plan(
    plan: ScanPlan,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Copy every kept candidate into the output folder and update the
    registry. Returns ``{"written": n, "skipped": n, "cancelled": bool}``."""
    from PIL import Image

    out_dir = plan.output_folder
    os.makedirs(out_dir, exist_ok=True)
    registry = load_registry(out_dir)
    kept = plan.kept
    written = 0
    skipped = 0
    cancelled = False
    for i, cand in enumerate(kept):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        if progress is not None:
            progress(i, len(kept), cand.source_image)
        stem = dataset_stem(cand.source_image, cand.sha256)
        png_path = os.path.join(out_dir, stem + ".png")
        if os.path.exists(png_path) or cand.sha256 in registry:
            skipped += 1
            continue
        try:
            if cand.source_image.lower().endswith(".png"):
                shutil.copy2(cand.source_image, png_path)
            else:
                with Image.open(cand.source_image) as img:
                    img.convert("RGB").save(png_path)
        except OSError:
            skipped += 1
            continue
        with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(cand.prompt + "\n")
        record = {
            "prompt": cand.prompt,
            "text": cand.text,
            "words": list(cand.words),
            "ocr_reading": cand.ocr_reading or "",
            "text_score": cand.text_score,
            "boxed_words": cand.boxed_words,
            "reward_reading": cand.reward_reading or "",
            "reader_agreement": cand.reader_agreement,
            "source_image": cand.source_image,
            "sha256": cand.sha256,
        }
        with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        registry.add(cand.sha256)
        written += 1
    save_registry(out_dir, registry)
    if progress is not None and not cancelled:
        progress(len(kept), len(kept), "")
    return {"written": written, "skipped": skipped, "cancelled": cancelled}


# --------------------------------------------------------------------------- #
# Default model halves
# --------------------------------------------------------------------------- #


def make_read_text(device: str = "auto") -> Callable[[str], str | None]:
    """The curation OCR (PaddleOCR-VL, GOT-OCR2 fallback) as a reader callable."""
    from modules.util.text_adherence_util import load_ocr_backend, read_image_text

    reader = load_ocr_backend(device=device)

    def read(path: str) -> str | None:
        readings = read_image_text([path], backend=reader)
        return readings.get(path)

    return read


def make_count_boxed(device: str = "auto") -> Callable[[str, Sequence[str]], tuple[int, list[str]]]:
    """The training-time detector + recognizer: how many target words would
    receive a gradient."""
    from modules.util.text_reward_model import TextRewardModel

    import torch

    import numpy as np
    from PIL import Image

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextRewardModel(torch.device(device), augment=False)

    def count(path: str, words: Sequence[str]) -> tuple[int, list[str]]:
        with Image.open(path) as handle:
            pil = handle.convert("RGB")
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float()[None] / 255.0
        detected = model.detect(x.to(model.device))[0]
        alignment = model.align(detected, list(words))
        return len(alignment.pairs), [detected.readings[bi] for bi, _ in alignment.pairs]

    return count


def write_wildcard_file(directory: str, name: str, prompts: Sequence[str]) -> str:
    """One prompt per line, SwarmUI wildcard format. Returns the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(p.strip() for p in prompts if p.strip()) + "\n")
    return path
