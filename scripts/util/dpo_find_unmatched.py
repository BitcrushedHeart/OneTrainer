"""
Standalone DPO pair-mismatch diagnostic.

Two layers of investigation, in order of faithfulness to what the running
trainer actually does:

1. Static: replicates `CollectPaths + PairByFilename` purely from disk
   (no MGDS imports). Confirms whether the FILESYSTEM itself has the mismatch.

2. Runtime: spins up a minimal MGDS pipeline with the real CollectPaths +
   PairByFilename modules (monkey-patched to print offending paths).
   This matches what the trainer sees at epoch start.

Usage:
    python scripts/util/dpo_find_unmatched.py \\
        --concepts F:\\Datasets\\RLHF\\concepts.json [--mgds]
"""

import argparse
import json
import os
import sys

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif"}
SUPPORTED_VIDEO_EXTENSIONS = {".webm", ".mkv", ".flv", ".avi", ".mov", ".wmv", ".mp4", ".mpeg", ".m4v"}
EXCLUDE_POSTFIX = ["-masklabel", "-condlabel"]


def canonical_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def dpo_pair_key(image_path: str, concept_path: str) -> str:
    try:
        relative = os.path.relpath(image_path, concept_path)
    except ValueError:
        relative = os.path.basename(image_path)
    return os.path.splitext(relative.replace("\\", "/"))[0]


# ------------------------------ STATIC FS CHECK ------------------------------


def list_files(path: str, include_subdirectories: bool):
    dir_list = [os.path.join(path, f) for f in os.listdir(path)]
    files = [p for p in dir_list if os.path.isfile(p)]
    if include_subdirectories:
        for d in dir_list:
            if os.path.isdir(d) and not os.path.basename(d).startswith("."):
                files.extend(list_files(d, True))
    return files


def collect_paths_static(concept: dict, extensions: set):
    path = concept["path"]
    include_sub = concept.get("include_subdirectories", False)
    exts = {e.lower() for e in extensions}
    all_files = sorted(list_files(path, include_sub))
    all_files = [f for f in all_files if os.path.splitext(f)[1].lower() in exts]
    all_files = [f for f in all_files if not any(os.path.splitext(f)[0].endswith(p) for p in EXCLUDE_POSTFIX)]
    return all_files


def static_check(concepts: list[dict], is_validation: bool, extensions: set):
    chosen_type = "DPO_CHOSEN_VAL" if is_validation else "DPO_CHOSEN"
    rejected_type = "DPO_REJECTED_VAL" if is_validation else "DPO_REJECTED"
    enabled = [c for c in concepts if c.get("enabled", True)]
    chosen_concepts = [c for c in enabled if c.get("type") == chosen_type]
    rejected_concepts = [c for c in enabled if c.get("type") == rejected_type]
    if len(chosen_concepts) != len(rejected_concepts):
        print(f"  Mismatched concept counts: {len(chosen_concepts)} chosen vs {len(rejected_concepts)} rejected")
        return 1
    if not chosen_concepts:
        return 0
    concept_pairs = list(zip(chosen_concepts, rejected_concepts))
    concept_lookup = {}
    for pid, (c, r) in enumerate(concept_pairs):
        concept_lookup[canonical_path(c["path"])] = (pid, True)
        concept_lookup[canonical_path(r["path"])] = (pid, False)

    chosen_idx, rejected_idx, chosen_p, rejected_p = {}, {}, {}, {}
    collisions = []
    pi = 0
    for concept in enabled:
        cp = canonical_path(concept["path"])
        files = collect_paths_static(concept, extensions)
        info = concept_lookup.get(cp)
        for f in files:
            idx = pi
            pi += 1
            if info is None:
                continue
            pid, is_chosen = info
            key = (pid, dpo_pair_key(f, cp))
            bi = chosen_idx if is_chosen else rejected_idx
            bp = chosen_p if is_chosen else rejected_p
            if key in bi:
                collisions.append(("chosen" if is_chosen else "rejected", key, bp[key], f))
            bi[key] = idx
            bp[key] = f

    print(f"  chosen unique keys: {len(chosen_idx)}, rejected unique keys: {len(rejected_idx)}")
    mr = sorted(set(chosen_idx) - set(rejected_idx))
    mc = sorted(set(rejected_idx) - set(chosen_idx))
    if collisions:
        print(f"  !! {len(collisions)} within-side key collisions:")
        for side, k, prior, now in collisions[:20]:
            print(f"     {side} key={k!r}\n       prior: {prior}\n       now  : {now}")
    if not mr and not mc:
        print("  (static) No mismatch.")
        return 0
    if mr:
        print(f"  {len(mr)} chosen paths with NO rejected match:")
        for k in mr[:50]:
            print(f"    key={k!r}  path={chosen_p.get(k)}")
    if mc:
        print(f"  {len(mc)} rejected paths with NO chosen match:")
        for k in mc[:50]:
            print(f"    key={k!r}  path={rejected_p.get(k)}")
    return len(mr) + len(mc)


# ------------------------------ RUNTIME MGDS CHECK ------------------------------


def runtime_check(concepts: list[dict], is_validation: bool, extensions: set):
    """Construct a real MGDS pipeline with CollectPaths + a patched PairByFilename
    that prints offending paths before raising. No model/cache required."""
    sys.path.insert(0, r"E:\AI\Data\Packages\OneTrainer")
    sys.path.insert(0, r"E:\AI\Data\Packages\OneTrainer\venv\src\mgds\src")

    import torch  # noqa
    from mgds.MGDS import MGDS  # noqa
    from mgds.PipelineModule import PipelineState
    from mgds.pipelineModules.CollectPaths import CollectPaths

    # Import and patch PairByFilename
    from modules.dataLoader.dpo import PairByFilename as PBF_mod

    OriginalPBF = PBF_mod.PairByFilename

    class LoudPBF(OriginalPBF):
        def _PairByFilename__build_pair_indices(self):
            # mirror the original but print offenders
            chosen_indices = {}
            rejected_indices = {}
            chosen_paths = {}
            rejected_paths = {}
            for index in range(self._get_previous_length("image_path")):
                concept_path = os.path.normcase(os.path.abspath(self._get_previous_item(0, "concept.path", index)))
                pair_info = self.concept_lookup.get(concept_path)
                if pair_info is None:
                    continue
                pid, is_chosen = pair_info
                image_path = self._get_previous_item(0, "image_path", index)
                key = (pid, dpo_pair_key(image_path, concept_path))
                if is_chosen:
                    if key in chosen_indices:
                        print(f"[PBF] chosen collision {key} -> {chosen_paths[key]} vs {image_path}")
                    chosen_indices[key] = index
                    chosen_paths[key] = image_path
                else:
                    if key in rejected_indices:
                        print(f"[PBF] rejected collision {key} -> {rejected_paths[key]} vs {image_path}")
                    rejected_indices[key] = index
                    rejected_paths[key] = image_path
            mr = sorted(set(chosen_indices) - set(rejected_indices))
            mc = sorted(set(rejected_indices) - set(chosen_indices))
            print(f"[PBF] chosen keys {len(chosen_indices)}, rejected keys {len(rejected_indices)}")
            if mr:
                print(f"[PBF] {len(mr)} CHOSEN orphans:")
                for k in mr:
                    print(f"    key={k!r}  path={chosen_paths.get(k)}")
            if mc:
                print(f"[PBF] {len(mc)} REJECTED orphans:")
                for k in mc:
                    print(f"    key={k!r}  path={rejected_paths.get(k)}")
            if mr or mc:
                raise RuntimeError("mismatch reported (see above)")
            self._pair_indices = []
            for k, ci in chosen_indices.items():
                ri = rejected_indices.get(k)
                if ri is not None:
                    self._pair_indices.append((ci, ri))
            self._pair_indices.sort(key=lambda x: x[0])

    PBF_mod.PairByFilename = LoudPBF

    # Set up ConceptConfig-style dicts that MGDS understands: pass list of dicts matching
    # ConceptConfig.to_dict() — but actually MGDS's ConceptPipelineModule expects raw dicts.
    # The concepts.json entries already match that format.
    mgds_concepts = concepts

    chosen_type = "DPO_CHOSEN_VAL" if is_validation else "DPO_CHOSEN"
    rejected_type = "DPO_REJECTED_VAL" if is_validation else "DPO_REJECTED"
    enabled = [c for c in concepts if c.get("enabled", True)]
    chosen_cs = [c for c in enabled if c.get("type") == chosen_type]
    rejected_cs = [c for c in enabled if c.get("type") == rejected_type]
    if not chosen_cs:
        print(f"  (runtime) No {chosen_type} concepts; skipping")
        return 0
    concept_pairs = [(c["path"], r["path"]) for c, r in zip(chosen_cs, rejected_cs)]

    collect_paths = CollectPaths(
        concept_in_name="concept",
        path_in_name="path",
        include_subdirectories_in_name="concept.include_subdirectories",
        enabled_in_name="enabled",
        path_out_name="image_path",
        concept_out_name="concept",
        extensions=list(extensions),
        include_postfix=None,
        exclude_postfix=["-masklabel", "-condlabel"],
    )
    pair_by_filename = LoudPBF(
        concept_pairs=concept_pairs,
        chosen_names=[("image_path", "image_path"), ("concept", "concept")],
        rejected_names=[("image_path", "image_path_rejected")],
    )

    # MGDS expects concepts as list of dicts and settings as a dict.
    settings = {}
    ds = MGDS(
        device=torch.device("cpu"),
        concepts=mgds_concepts,
        settings=settings,
        definition=[collect_paths, pair_by_filename],
        batch_size=1,
        state=PipelineState(),
    )
    try:
        ds.start_next_epoch()
        print("  (runtime) No mismatch.")
        return 0
    except RuntimeError as e:
        print(f"  (runtime) caught: {e}")
        return 2


# ------------------------------ MAIN ------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", default=r"F:\Datasets\RLHF\concepts.json")
    p.add_argument("--allow-videos", action="store_true")
    p.add_argument("--mgds", action="store_true", help="Run the MGDS runtime check in addition to static")
    p.add_argument("--mode", choices=["train", "val", "both"], default="both")
    args = p.parse_args()

    with open(args.concepts, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    extensions = set(SUPPORTED_IMAGE_EXTENSIONS)
    if args.allow_videos:
        extensions |= SUPPORTED_VIDEO_EXTENSIONS

    modes = []
    if args.mode in ("train", "both"):
        modes.append((False, "TRAIN"))
    if args.mode in ("val", "both"):
        modes.append((True, "VAL"))

    total = 0
    for is_val, label in modes:
        print(f"\n=== {label} — static filesystem check ===")
        total += static_check(concepts, is_val, extensions)
        if args.mgds:
            print(f"\n=== {label} — MGDS runtime check ===")
            total += runtime_check(concepts, is_val, extensions)

    sys.exit(0 if total == 0 else 2)


if __name__ == "__main__":
    main()
