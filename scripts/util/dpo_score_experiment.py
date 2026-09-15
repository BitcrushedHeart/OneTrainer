"""
Replay curated DPO pairs through the machine scorers and report how often each
one agrees with the human pick.

For every pair in one or more Pair Tool exports (``manifest.json`` + chosen/
rejected trees) the script computes, for both sides:

* prompt adherence (SigLIP 2 over the caption's propositions, plus the Engine
  age head when the caption claims an age),
* realism (P(AI) from the realism head),
* scalar quality (Studio's Simple formula),
* the two combiners (gate-then-rank, weighted z).

A scorer "agrees" with a pair when it ranks the human's chosen image above the
rejected one; a tie counts half. Accuracy is reported per manifest and per
scorer, per adherence axis (which propositions actually discriminate), and as
a realism-vs-adherence agreement table (where adherence adds information the
realism head does not have). Every per-pair number also lands in a CSV.

Only counts and scores are printed — no prompts, no file names.

Usage:
    python scripts/util/dpo_score_experiment.py --manifest "F:/Datasets/RLHF_3" \\
        --manifest "F:/Datasets/RLHF_4" --out scores.csv [--limit 200] [--device auto|cpu|cuda]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.util import dpo_combined_score as combined  # noqa: E402
from modules.util.adherence_score_util import (  # noqa: E402
    AGE_MODEL_AXIS,
    claimed_age_range,
    load_siglip_embedder,
    score_images_adherence,
)
from modules.util.age_estimate_util import estimate_ages  # noqa: E402
from modules.util.dpo_curation_util import UNCONDITIONAL_PROMPT, DpoScanCache, load_manifest  # noqa: E402
from modules.util.realism_detect_util import score_images_realism  # noqa: E402
from modules.util.scalar_quality_util import score_images_quality  # noqa: E402


def _resolve(output_dir: str, side: str, entry: dict) -> str | None:
    name = entry.get(f"{side}_file")
    if name:
        for sub in ("train", "val", ""):
            candidate = os.path.join(output_dir, side, sub, name) if sub else os.path.join(output_dir, side, name)
            if os.path.isfile(candidate):
                return candidate
    source = entry.get(f"{side}_source")
    return source if source and os.path.isfile(source) else None


def _pairs(output_dir: str, limit: int | None) -> list[dict]:
    manifest = load_manifest(output_dir)
    pairs = []
    for entry in manifest.get("pairs", []):
        prompt = entry.get("prompt") or ""
        if prompt == UNCONDITIONAL_PROMPT:
            continue
        chosen, rejected = _resolve(output_dir, "chosen", entry), _resolve(output_dir, "rejected", entry)
        if not chosen or not rejected:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "pair_id": entry.get("pair_id")})
        if limit and len(pairs) >= limit:
            break
    return pairs


def _win(chosen: float | None, rejected: float | None, higher_is_better: bool = True) -> float | None:
    if chosen is None or rejected is None:
        return None
    if chosen == rejected:
        return 0.5
    return float((chosen > rejected) == higher_is_better)


def _accuracy(wins: list[float]) -> str:
    n = len(wins)
    return f"{(sum(wins) / n * 100):5.1f}% (n={n})" if n else "   n/a (n=0)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True, help="Pair Tool export dir (repeatable)")
    parser.add_argument("--out", default="dpo_score_experiment.csv")
    parser.add_argument("--cache", default=".dpo_score_experiment_cache.json")
    parser.add_argument("--limit", type=int, default=None, help="max pairs per manifest")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-patches", type=int, default=784)
    parser.add_argument("--no-realism", action="store_true")
    parser.add_argument("--no-age", action="store_true")
    parser.add_argument("--no-quality", action="store_true")
    args = parser.parse_args()

    cache = DpoScanCache(args.cache)
    # The ONNX heads only go to the GPU when asked outright: "auto" puts the
    # SigLIP tower there (it is the expensive one) and keeps the two small
    # heads on the CPU so a generation job sharing the card is not squeezed.
    onnx_device = "auto" if args.device == "cuda" else "cpu"
    embedder = load_siglip_embedder(device=args.device, max_patches=args.max_patches)
    # The handle is lazy; the experiment always runs the model, so load it now.
    embedder.load_model()
    print(f"SigLIP: {embedder.model_version} on {embedder.device}")

    rows: list[dict] = []
    per_manifest: dict[str, list[dict]] = {}
    started = time.time()
    for output_dir in args.manifest:
        pairs = _pairs(output_dir, args.limit)
        label = os.path.basename(os.path.normpath(output_dir))
        print(f"\n[{label}] {len(pairs)} pairs")
        all_paths = sorted({p for pair in pairs for p in (pair["chosen"], pair["rejected"])})

        realism: dict[str, float] = {}
        if not args.no_realism:
            t = time.time()
            for start in range(0, len(all_paths), 200):
                realism.update(score_images_realism(all_paths[start : start + 200], cache=cache, device=onnx_device))
            cache.save()
            print(f"  realism: {len(realism)} scored in {time.time() - t:.0f}s")

        quality: dict[str, float] = {}
        if not args.no_quality:
            t = time.time()
            quality = score_images_quality(all_paths)
            print(f"  quality: {len(quality)} scored in {time.time() - t:.0f}s")

        ages: dict[str, dict] = {}
        if not args.no_age:
            need = sorted(
                {p for pair in pairs if claimed_age_range(pair["prompt"]) for p in (pair["chosen"], pair["rejected"])}
            )
            if need:
                t = time.time()
                for start in range(0, len(need), 100):
                    ages.update(estimate_ages(need[start : start + 100], cache=cache, device=onnx_device))
                cache.save()
                print(f"  age: {len(ages)}/{len(need)} estimated in {time.time() - t:.0f}s")

        by_prompt: dict[str, list[dict]] = defaultdict(list)
        for pair in pairs:
            by_prompt[pair["prompt"]].append(pair)
        t = time.time()
        adherence: dict[tuple[str, str], dict] = {}
        for i, (prompt, group) in enumerate(by_prompt.items(), 1):
            paths = sorted({p for pair in group for p in (pair["chosen"], pair["rejected"])})
            scored = score_images_adherence(paths, prompt, embedder=embedder, cache=cache, age_estimates=ages or None)
            for path, result in scored.items():
                adherence[(prompt, path)] = result
            if i % 25 == 0:
                cache.save()
                print(f"  adherence: {i}/{len(by_prompt)} prompts, {time.time() - t:.0f}s")
        cache.save()
        print(f"  adherence: {len(by_prompt)} prompts in {time.time() - t:.0f}s")

        manifest_rows = []
        for pair in pairs:
            c, r = pair["chosen"], pair["rejected"]
            ac, ar = adherence.get((pair["prompt"], c)), adherence.get((pair["prompt"], r))
            images = {
                c: {
                    "adherence": ac and ac["adherence"],
                    "age": ac and ac["age_agreement"],
                    "realism": realism.get(c),
                    "quality": quality.get(c),
                },
                r: {
                    "adherence": ar and ar["adherence"],
                    "age": ar and ar["age_agreement"],
                    "realism": realism.get(r),
                    "quality": quality.get(r),
                },
            }
            z = combined.weighted_z(images)
            gate = combined.gate_then_rank(images)
            axis_wins = {}
            if ac and ar:
                pc = {p["axis"]: p["prob"] for p in ac["propositions"] if p["axis"]}
                pr = {p["axis"]: p["prob"] for p in ar["propositions"] if p["axis"]}
                for axis in pc.keys() & pr.keys():
                    axis_wins[axis] = _win(pc[axis], pr[axis])
            row = {
                "manifest": label,
                "pair_id": pair["pair_id"],
                "adherence": _win(images[c]["adherence"], images[r]["adherence"]),
                "axis_mean": _win(ac and ac["axis_mean"], ar and ar["axis_mean"]),
                "axis_min": _win(ac and ac["axis_min"], ar and ar["axis_min"]),
                "phrase_mean": _win(ac and ac["phrase_mean"], ar and ar["phrase_mean"]),
                "age": _win(images[c]["age"], images[r]["age"]),
                "realism": _win(realism.get(c), realism.get(r), higher_is_better=False),
                "quality": _win(quality.get(c), quality.get(r)),
                "weighted_z": _win(z[c], z[r]),
                "gate_then_rank": None if gate[0] == gate[1] else float(gate[0] == c),
                "chosen_adherence": images[c]["adherence"],
                "rejected_adherence": images[r]["adherence"],
                "chosen_realism": realism.get(c),
                "rejected_realism": realism.get(r),
                "chosen_quality": quality.get(c),
                "rejected_quality": quality.get(r),
                "n_axis": ac and ac["n_axis"],
                "n_phrase": ac and ac["n_phrase"],
                "axis_wins": json.dumps(axis_wins),
            }
            manifest_rows.append(row)
        per_manifest[label] = manifest_rows
        rows.extend(manifest_rows)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["manifest"])
        writer.writeheader()
        writer.writerows(rows)

    metrics = (
        "adherence",
        "axis_mean",
        "axis_min",
        "phrase_mean",
        "age",
        "realism",
        "quality",
        "weighted_z",
        "gate_then_rank",
    )
    print(f"\n=== Agreement with the human pick (chosen ranked above rejected) — {time.time() - started:.0f}s ===")
    for label, manifest_rows in per_manifest.items():
        print(f"\n[{label}] {len(manifest_rows)} pairs")
        for metric in metrics:
            wins = [r[metric] for r in manifest_rows if r[metric] is not None]
            print(f"  {metric:15s} {_accuracy(wins)}")
        axis_acc: dict[str, list[float]] = defaultdict(list)
        for r in manifest_rows:
            for axis, win in json.loads(r["axis_wins"]).items():
                if win is not None:
                    axis_acc[axis].append(win)
        print("  per axis:")
        for axis, wins in sorted(axis_acc.items(), key=lambda kv: -len(kv[1])):
            marker = " (age head)" if axis == AGE_MODEL_AXIS else ""
            print(f"    {axis:18s} {_accuracy(wins)}{marker}")
        both = [r for r in manifest_rows if r["realism"] is not None and r["adherence"] is not None]
        if both:
            rr = sum(1 for r in both if r["realism"] == 1 and r["adherence"] == 1)
            ra = sum(1 for r in both if r["realism"] == 1 and r["adherence"] == 0)
            ar_ = sum(1 for r in both if r["realism"] == 0 and r["adherence"] == 1)
            aa = sum(1 for r in both if r["realism"] == 0 and r["adherence"] == 0)
            print(
                f"  realism vs adherence (n={len(both)}): both right {rr}, realism only {ra}, adherence only {ar_}, both wrong {aa}"
            )
    print(f"\nCSV: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
