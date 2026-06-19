"""
Discard DPO pairs that would trigger

    "RLHF DPO paired samples must have matching crop resolutions in
     chosen and rejected concepts."

at AspectBatchSorting time. Two independent failure modes are checked:

(A) Source aspect mismatch — chosen and rejected images have aspects that
    fall in different ``AspectBucketing`` buckets at the configured target.
    Bucket math is replicated locally (no MGDS imports).

(B) SmartDiskCache variant mismatch — when ``--cache-json`` is supplied,
    the cache index is consulted: a pair is flagged when one side has a
    cached variant at the target's bucket sizes and the other side doesn't.
    The cache then returns differently-bucketed crop_resolution values for
    the two sides at training time. This catches the case where a previous
    run cached items at a different target (e.g. 512) and the current run
    is at 768 — entries lacking a 768 variant fall back to ``_any_variant``
    which yields a 512-bucket crop_resolution.

Flagged pairs (image + .txt sidecar) are moved to a ``.discard`` subdir
inside each concept folder, preserving relative sub-path. CollectPaths
skips dotted subdirectories during recursion, so the trainer ignores them.

Use ``--dry-run`` to count without moving. Only counts are printed —
filenames and captions are never written to stdout.

Usage:
    python scripts/util/dpo_discard_resolution_mismatches.py \\
        --concepts F:\\Datasets\\RLHF\\concepts.json \\
        --resolution 768 --quantization 64 \\
        --cache-json F:\\workspace\\SoReal!\\cache\\image\\cache.json
"""

import argparse
import json
import math
import os
import shutil
import sys

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif"}
EXCLUDE_POSTFIX = ("-masklabel", "-condlabel")

# Mirror of AspectBucketing.all_possible_input_aspects.
ALL_POSSIBLE_INPUT_ASPECTS = [
    (1.0, 1.0),
    (1.0, 1.25),
    (1.0, 1.5),
    (1.0, 1.75),
    (1.0, 2.0),
    (1.0, 2.5),
    (1.0, 3.0),
    (1.0, 3.5),
    (1.0, 4.0),
]


def quantize_resolution(resolution, quantization):
    return (
        round(resolution[0] / quantization) * quantization,
        round(resolution[1] / quantization) * quantization,
    )


def build_buckets(target_resolution, quantization):
    """Return (bucket_resolutions, bucket_aspects) for a single target.

    Mirrors AspectBucketing.__create_automatic_buckets but for one target at
    a time (sufficient because bucket_for_aspect is keyed per-target).
    """
    res = [
        (
            h / math.sqrt(h * w) * target_resolution,
            w / math.sqrt(h * w) * target_resolution,
        )
        for (h, w) in ALL_POSSIBLE_INPUT_ASPECTS
    ]
    res = res + [(w, h) for (h, w) in res]
    res = [quantize_resolution(r, quantization) for r in res]
    res = list(set(res))
    aspects = [h / w for (h, w) in res]
    return res, aspects


def bucket_for_aspect(aspect, bucket_resolutions, bucket_aspects):
    best_idx = 0
    best_delta = abs(bucket_aspects[0] - aspect)
    for i in range(1, len(bucket_aspects)):
        d = abs(bucket_aspects[i] - aspect)
        if d < best_delta:
            best_delta = d
            best_idx = i
    return bucket_resolutions[best_idx]


def parse_resolutions(resolution_str):
    """Parse a target-resolution string the same way AspectBucketing does.

    Returns a list of integer targets (the AxB fixed-resolution syntax is
    out of scope for this fixer; bail loudly so the user notices).
    """
    s = (resolution_str or "").strip()
    if not s:
        return []
    if "x" in s and "," not in s:
        raise ValueError(
            f"Fixed-resolution syntax {s!r} is not supported by this fixer; "
            "pass numeric targets like '1024' or '512,1024'."
        )
    return [int(part.strip()) for part in s.split(",") if part.strip()]


def canonical_path(p):
    return os.path.normcase(os.path.abspath(p))


def dpo_pair_key(image_path, concept_path):
    try:
        relative = os.path.relpath(image_path, concept_path)
    except ValueError:
        relative = os.path.basename(image_path)
    return os.path.splitext(relative.replace("\\", "/"))[0]


def list_image_files(path, include_subdirectories):
    """Replicates CollectPaths.__list_files: skip dotted subdirs on recursion."""
    out = []
    try:
        entries = os.listdir(path)
    except OSError:
        return out
    for name in entries:
        full = os.path.join(path, name)
        if os.path.isfile(full):
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                stem = os.path.splitext(name)[0]
                if not any(stem.endswith(p) for p in EXCLUDE_POSTFIX):
                    out.append(full)
        elif include_subdirectories and os.path.isdir(full):
            if not name.startswith("."):
                out.extend(list_image_files(full, True))
    return out


def read_image_size(path):
    """Return (height, width), honoring EXIF orientation.

    LoadImage decodes with ``apply_exif_orientation=True``, so an image stored
    landscape with ``Orientation=6`` is presented to AspectBucketing as
    portrait. PIL's lazy ``.size`` returns the *stored* dimensions, so we
    swap H/W when the EXIF tag indicates a 90°/270° rotation (values 5–8).
    Reading just the orientation tag does not force pixel decoding.
    """
    from PIL import Image

    with Image.open(path) as img:
        w, h = img.size
        try:
            exif = img.getexif()
            orientation = exif.get(0x0112, 1) if exif is not None else 1
        except Exception:
            orientation = 1
    if orientation in (5, 6, 7, 8):
        h, w = w, h
    return h, w


def concept_targets(concept, default_targets):
    """Resolve a concept's effective integer target resolutions."""
    image_cfg = concept.get("image") or {}
    if image_cfg.get("enable_resolution_override"):
        override = image_cfg.get("resolution_override") or ""
        try:
            return parse_resolutions(override)
        except ValueError as e:
            print(f"  warning: skipping concept override {override!r}: {e}")
            return default_targets
    return default_targets


def precompute_bucket_tables(targets, quantization):
    return {t: build_buckets(t, quantization) for t in targets}


def target_bucket_set(targets, quantization):
    """Set of (h, w) tuples a SmartDiskCache variant would use for these targets."""
    out = set()
    for t in targets:
        res, _aspects = build_buckets(t, quantization)
        out.update(res)
    return out


def parse_variant_key(key):
    """Variant keys are stored as ``"{h}x{w}"`` strings (see SmartDiskCache._get_resolution_string)."""
    try:
        a, b = key.split("x")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return None


def entry_has_target_variant(entry, target_buckets):
    variants = entry.get("variants") or {}
    for k in variants:
        parsed = parse_variant_key(k)
        if parsed is not None and parsed in target_buckets:
            return True
    return False


def entry_first_variant_bucket(entry):
    """Bucket (h, w) for the *first* variant key in dict order — what
    ``_populate_active_keys`` picks as the active variant when the active
    key isn't otherwise found. Drift recovery reorders so the current
    target's variant is first; if it isn't, the cache returns a stale
    bucket for this entry."""
    variants = entry.get("variants") or {}
    for k in variants:
        return parse_variant_key(k)
    return None


def load_cache_entries(cache_json_path):
    """Return a dict of normcase(filepath) -> entry. Empty if path is None or unreadable."""
    if not cache_json_path:
        return None
    with open(cache_json_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    entries = cache.get("entries", {})
    return {os.path.normcase(fp): e for fp, e in entries.items()}


def discard_pair(chosen_image, chosen_concept, rejected_image, rejected_concept, dry_run):
    moved = 0
    for image_path, concept_path in (
        (chosen_image, chosen_concept),
        (rejected_image, rejected_concept),
    ):
        rel = os.path.relpath(image_path, concept_path)
        target_dir = os.path.join(concept_path, ".discard", os.path.dirname(rel))
        if not dry_run:
            os.makedirs(target_dir, exist_ok=True)
        for src in (image_path, os.path.splitext(image_path)[0] + ".txt"):
            if not os.path.isfile(src):
                continue
            dst = os.path.join(target_dir, os.path.basename(src))
            if not dry_run:
                shutil.move(src, dst)
            moved += 1
    return moved


def process_pair(chosen_concept, rejected_concept, default_targets, quantization, cache_lookup, dry_run):
    cp = chosen_concept["path"]
    rp = rejected_concept["path"]
    c_inc = chosen_concept.get("include_subdirectories", False)
    r_inc = rejected_concept.get("include_subdirectories", False)

    chosen_files = {dpo_pair_key(f, cp): f for f in list_image_files(cp, c_inc)}
    rejected_files = {dpo_pair_key(k, rp): k for k in list_image_files(rp, r_inc)}
    matched_keys = set(chosen_files) & set(rejected_files)

    c_targets = concept_targets(chosen_concept, default_targets)
    r_targets = concept_targets(rejected_concept, default_targets)
    if c_targets != r_targets:
        print(
            f"  warning: chosen targets {c_targets} != rejected targets {r_targets}; "
            f"using union for the bucket comparison"
        )
    union_targets = sorted(set(c_targets) | set(r_targets))
    if not union_targets:
        print("  no target resolutions resolved; skipping pair")
        return {"scanned": 0, "aspect_mismatch": 0, "cache_xor": 0, "cache_only_one_missing": 0, "unreadable": 0}

    tables = precompute_bucket_tables(union_targets, quantization)
    target_buckets = target_bucket_set(union_targets, quantization)

    counts = {
        "scanned": 0,
        "aspect_mismatch": 0,
        "cache_xor": 0,
        "cache_active_variant_mismatch": 0,
        "cache_only_one_missing": 0,
        "unreadable": 0,
    }

    for key in matched_keys:
        c_img = chosen_files[key]
        r_img = rejected_files[key]
        counts["scanned"] += 1

        try:
            ch, cw = read_image_size(c_img)
            rh, rw = read_image_size(r_img)
        except Exception:
            counts["unreadable"] += 1
            continue

        c_aspect = ch / cw
        r_aspect = rh / rw

        # (A) Source aspect-bucket check.
        aspect_mismatch = False
        for t in union_targets:
            bres, basp = tables[t]
            if bucket_for_aspect(c_aspect, bres, basp) != bucket_for_aspect(r_aspect, bres, basp):
                aspect_mismatch = True
                break

        # (B) SmartDiskCache variant alignment check.
        cache_xor = False
        cache_active_mismatch = False
        cache_one_missing = False
        if cache_lookup is not None:
            ce = cache_lookup.get(os.path.normcase(c_img))
            re_ = cache_lookup.get(os.path.normcase(r_img))
            c_present = ce is not None
            r_present = re_ is not None
            if c_present and r_present:
                c_has = entry_has_target_variant(ce, target_buckets)
                r_has = entry_has_target_variant(re_, target_buckets)
                if c_has != r_has:
                    cache_xor = True
                else:
                    # Both have a target variant (or neither does). Now check
                    # the *first* variant in dict order — that's what the
                    # cache will return when active_key isn't found, and what
                    # _populate_active_keys picks as the active key.
                    c_first = entry_first_variant_bucket(ce)
                    r_first = entry_first_variant_bucket(re_)
                    if c_first is not None and r_first is not None and c_first != r_first:
                        cache_active_mismatch = True
            elif c_present != r_present:
                # One side cached, the other not yet. The cached side returns
                # whatever variant it has; the uncached side gets (re)built
                # against the current target. Conservatively flag.
                cache_one_missing = True

        if aspect_mismatch:
            counts["aspect_mismatch"] += 1
            discard_pair(c_img, cp, r_img, rp, dry_run)
        elif cache_xor:
            counts["cache_xor"] += 1
            discard_pair(c_img, cp, r_img, rp, dry_run)
        elif cache_active_mismatch:
            counts["cache_active_variant_mismatch"] += 1
            discard_pair(c_img, cp, r_img, rp, dry_run)
        elif cache_one_missing:
            counts["cache_only_one_missing"] += 1
            discard_pair(c_img, cp, r_img, rp, dry_run)

    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", default=r"F:\Datasets\RLHF\concepts.json")
    p.add_argument(
        "--resolution",
        default="1024",
        help="Default target resolution(s) — single int or comma-separated list, "
        'e.g. "1024" or "512,1024". Per-concept resolution_override is honored.',
    )
    p.add_argument(
        "--quantization",
        type=int,
        default=64,
        help="Bucket quantization. SDXL/Flux use 64; some pipelines use 8 or 16. Match what your trainer uses.",
    )
    p.add_argument("--mode", choices=["train", "val", "both"], default="both")
    p.add_argument(
        "--cache-json",
        default=None,
        help="Path to SmartDiskCache cache.json (e.g. <cache_dir>/image/cache.json). "
        "When supplied, also discards pairs where chosen/rejected variants in "
        "the cache disagree about whether a target-resolution bucket is present.",
    )
    p.add_argument("--dry-run", action="store_true", help="Report what would be moved without modifying disk.")
    args = p.parse_args()

    try:
        default_targets = parse_resolutions(args.resolution)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    if not default_targets:
        print("error: --resolution must contain at least one numeric target", file=sys.stderr)
        sys.exit(2)

    with open(args.concepts, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    enabled = [c for c in concepts if c.get("enabled", True)]

    cache_lookup = None
    if args.cache_json:
        if not os.path.isfile(args.cache_json):
            print(f"error: --cache-json {args.cache_json!r} not found", file=sys.stderr)
            sys.exit(2)
        cache_lookup = load_cache_entries(args.cache_json)
        print(f"loaded cache index: {len(cache_lookup)} entries")

    modes = []
    if args.mode in ("train", "both"):
        modes.append((False, "TRAIN", "DPO_CHOSEN", "DPO_REJECTED"))
    if args.mode in ("val", "both"):
        modes.append((True, "VAL", "DPO_CHOSEN_VAL", "DPO_REJECTED_VAL"))

    grand = {
        "scanned": 0,
        "aspect_mismatch": 0,
        "cache_xor": 0,
        "cache_active_variant_mismatch": 0,
        "cache_only_one_missing": 0,
        "unreadable": 0,
    }

    for _is_val, label, chosen_type, rejected_type in modes:
        chosen_cs = [c for c in enabled if c.get("type") == chosen_type]
        rejected_cs = [c for c in enabled if c.get("type") == rejected_type]
        if not chosen_cs and not rejected_cs:
            continue
        if len(chosen_cs) != len(rejected_cs):
            print(f"\n=== {label} === skipped: {len(chosen_cs)} chosen vs {len(rejected_cs)} rejected concepts")
            continue
        print(f"\n=== {label} === {len(chosen_cs)} concept pair(s)")
        for idx, (cc, rc) in enumerate(zip(chosen_cs, rejected_cs)):
            print(f"  pair {idx + 1}/{len(chosen_cs)}")
            counts = process_pair(
                cc,
                rc,
                default_targets,
                args.quantization,
                cache_lookup,
                args.dry_run,
            )
            verb = "would discard" if args.dry_run else "discarded"
            print(
                f"    scanned={counts['scanned']}  "
                f"aspect_mismatch={counts['aspect_mismatch']}  "
                f"cache_xor={counts['cache_xor']}  "
                f"cache_active_mismatch={counts['cache_active_variant_mismatch']}  "
                f"cache_one_missing={counts['cache_only_one_missing']}  "
                f"unreadable={counts['unreadable']}  ({verb})"
            )
            for k in grand:
                grand[k] += counts[k]

    discarded = (
        grand["aspect_mismatch"]
        + grand["cache_xor"]
        + grand["cache_active_variant_mismatch"]
        + grand["cache_only_one_missing"]
    )
    verb = "would discard" if args.dry_run else "discarded"
    print(f"\nTOTAL: scanned={grand['scanned']}  {verb}={discarded}")
    print(
        f"  by reason — aspect={grand['aspect_mismatch']}  "
        f"cache_xor={grand['cache_xor']}  "
        f"cache_active_mismatch={grand['cache_active_variant_mismatch']}  "
        f"cache_one_missing={grand['cache_only_one_missing']}  "
        f"unreadable={grand['unreadable']}"
    )
    if args.dry_run:
        print("(dry run — nothing moved; rerun without --dry-run to apply)")


if __name__ == "__main__":
    main()
