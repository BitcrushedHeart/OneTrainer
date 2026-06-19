"""Throwaway diagnostic: spin up the actual MGDS pipeline (CollectPaths +
LoadImage + CalcAspect + AspectBucketing + PairByFilename) and count how
many pairs the trainer would reject due to mismatching crop_resolution.
No SmartDiskCache, no model, no captions printed.
"""

import argparse
import json
import sys

sys.path.insert(0, r"E:\AI\Data\Packages\OneTrainer")
sys.path.insert(0, r"E:\AI\Data\Packages\OneTrainer\venv\src\mgds\src")

from modules.dataLoader.dpo.PairByFilename import PairByFilename

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.LoadImage import LoadImage

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", default=r"F:\Datasets\RLHF\concepts.json")
    p.add_argument("--resolution", default="768")
    p.add_argument("--quantization", type=int, default=64)
    p.add_argument("--mode", choices=["train", "val"], default="train")
    args = p.parse_args()

    with open(args.concepts, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    enabled = [c for c in concepts if c.get("enabled", True)]

    chosen_type = "DPO_CHOSEN_VAL" if args.mode == "val" else "DPO_CHOSEN"
    rejected_type = "DPO_REJECTED_VAL" if args.mode == "val" else "DPO_REJECTED"
    chosen_cs = [c for c in enabled if c.get("type") == chosen_type]
    rejected_cs = [c for c in enabled if c.get("type") == rejected_type]
    if not chosen_cs:
        print(f"No {chosen_type} concepts; nothing to check.")
        return
    concept_pairs = [(c["path"], r["path"]) for c, r in zip(chosen_cs, rejected_cs)]

    settings = {"target_resolution": args.resolution, "target_frames": "1"}

    collect_paths = CollectPaths(
        concept_in_name="concept",
        path_in_name="path",
        include_subdirectories_in_name="concept.include_subdirectories",
        enabled_in_name="enabled",
        path_out_name="image_path",
        concept_out_name="concept",
        extensions=[".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif"],
        include_postfix=None,
        exclude_postfix=["-masklabel", "-condlabel"],
    )
    load_image = LoadImage(
        path_in_name="image_path",
        image_out_name="image",
        range_min=0.0,
        range_max=1.0,
        channels=3,
        supported_extensions={".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif"},
    )
    calc_aspect = CalcAspect(image_in_name="image", resolution_out_name="original_resolution")
    aspect_bucketing = AspectBucketing(
        quantization=args.quantization,
        resolution_in_name="original_resolution",
        target_resolution_in_name="settings.target_resolution",
        enable_target_resolutions_override_in_name="concept.image.enable_resolution_override",
        target_resolutions_override_in_name="concept.image.resolution_override",
        target_frames_in_name="settings.target_frames",
        frame_dim_enabled=False,
        scale_resolution_out_name="scale_resolution",
        crop_resolution_out_name="crop_resolution",
        possible_resolutions_out_name="possible_resolutions",
    )

    pair_by_filename = PairByFilename(
        concept_pairs=concept_pairs,
        chosen_names=[("crop_resolution", "chosen_crop_resolution")],
        rejected_names=[("crop_resolution", "rejected_crop_resolution")],
    )

    ds = MGDS(
        device=torch.device("cpu"),
        concepts=enabled,
        settings=settings,
        definition=[collect_paths, load_image, calc_aspect, aspect_bucketing, pair_by_filename],
        batch_size=1,
        state=PipelineState(),
    )

    ds.start_next_epoch()

    mismatched = 0
    total = 0
    errors = 0
    n = pair_by_filename.length()
    print(f"pair count: {n}")
    for i in range(n):
        try:
            chosen_idx, rejected_idx = pair_by_filename._PairByFilename__get_pair_indices()[i]
            ch = aspect_bucketing.get_item(0, chosen_idx)["crop_resolution"]
            rh = aspect_bucketing.get_item(0, rejected_idx)["crop_resolution"]
            total += 1
            if tuple(ch) != tuple(rh):
                mismatched += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  error on pair {i}: {e!r}")
    print(f"checked={total} mismatched_crop_resolution={mismatched} errors={errors}")


if __name__ == "__main__":
    main()
