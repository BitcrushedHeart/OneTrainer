"""Direct examination of the cache state for the failing items.

Inspects what's in image cache.json and mask cache.json for a list of
failing source paths, loads the .pt files referenced, prints actual
tensor shapes, and verifies whether variant keys match tensor shapes.
"""

import json
import os
import sys

import torch

IMAGE_CACHE = r"F:\workspace\SoReal!\cache\image"
MASK_CACHE = r"F:\workspace\SoReal!\cache\mask"

FAILING = [
    r"F:\Datasets\SoReal!\Sex\pornpics_23932278_008.jpg",
    r"F:\Concepts\Braless\positive\train\66215583_013_fdae.jpg",
    r"F:\Datasets\SoReal!\Portraits\63101792_005_7a52.jpg",
    r"F:\Datasets\SoReal!\Cunnilingus\47121258_064_70f3.webp",
]


def derive_mask_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + "-masklabel.png"


def load_index(d):
    return json.load(open(os.path.join(d, "cache.json"), "r", encoding="utf-8"))


def shape_of(path):
    if not os.path.isfile(path):
        return f"<missing: {path}>"
    try:
        d = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as e:
        return f"<load error: {e}>"
    if not isinstance(d, dict):
        return "<not dict>"
    return {
        k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in d.items() if not k.startswith("__")
    }


def main():
    img_idx = load_index(IMAGE_CACHE)
    mask_idx = load_index(MASK_CACHE)
    print(f"image cache.json: {len(img_idx['entries'])} entries, bucket_method={img_idx.get('bucket_method')}")
    print(f"mask cache.json:  {len(mask_idx['entries'])} entries, bucket_method={mask_idx.get('bucket_method')}")
    print()

    for img_path in FAILING:
        mask_path = derive_mask_path(img_path)
        print(f"=== {os.path.basename(img_path)} ===")

        # Image entry
        img_entry = img_idx["entries"].get(img_path)
        if img_entry is None:
            print(f"  IMAGE entry MISSING in cache.json (path: {img_path})")
        else:
            print(
                f"  IMAGE hash={img_entry.get('hash', '')[:12]}  variants={list(img_entry.get('variants', {}).keys())}"
            )
            for vk, variant in img_entry.get("variants", {}).items():
                cf = variant["cache_file"]
                pt = os.path.join(IMAGE_CACHE, f"{cf}_1.pt")
                shapes = shape_of(pt)
                print(f"    image variant {vk!r}: cache_file={cf}")
                print(f"      schema_keys={variant.get('schema_keys')}")
                if isinstance(shapes, dict):
                    for k, v in shapes.items():
                        print(f"      pt[{k!r}] = {v}")
                else:
                    print(f"      {shapes}")

        # Mask entry
        mask_entry = mask_idx["entries"].get(mask_path)
        if mask_entry is None:
            print(f"  MASK entry MISSING in cache.json (path: {mask_path})")
        else:
            print(
                f"  MASK  hash={mask_entry.get('hash', '')[:12]}  variants={list(mask_entry.get('variants', {}).keys())}"
            )
            for vk, variant in mask_entry.get("variants", {}).items():
                cf = variant["cache_file"]
                pt = os.path.join(MASK_CACHE, f"{cf}_1.pt")
                shapes = shape_of(pt)
                print(f"    mask variant {vk!r}: cache_file={cf}")
                print(f"      schema_keys={variant.get('schema_keys')}")
                if isinstance(shapes, dict):
                    for k, v in shapes.items():
                        print(f"      pt[{k!r}] = {v}")
                else:
                    print(f"      {shapes}")

        # Verify alignment
        if img_entry and mask_entry:
            img_keys = set(img_entry.get("variants", {}).keys())
            mask_keys = set(mask_entry.get("variants", {}).keys())
            common = img_keys & mask_keys
            img_only = img_keys - mask_keys
            mask_only = mask_keys - img_keys
            print(
                f"  variant alignment: common={sorted(common)} img_only={sorted(img_only)} mask_only={sorted(mask_only)}"
            )

            # Sanity: for each variant key K, check that mask shape and image shape are consistent (mask_HW == image_HW)
            for vk in common:
                img_pt = os.path.join(IMAGE_CACHE, f"{img_entry['variants'][vk]['cache_file']}_1.pt")
                mask_pt = os.path.join(MASK_CACHE, f"{mask_entry['variants'][vk]['cache_file']}_1.pt")
                if os.path.isfile(img_pt) and os.path.isfile(mask_pt):
                    try:
                        img_d = torch.load(img_pt, weights_only=False, map_location="cpu")
                        mask_d = torch.load(mask_pt, weights_only=False, map_location="cpu")
                        ish = tuple(img_d.get("latent_image").shape[-2:]) if "latent_image" in img_d else None
                        msh = tuple(mask_d.get("latent_mask").shape[-2:]) if "latent_mask" in mask_d else None
                        cr = img_d.get("crop_resolution")
                        crop_match = ish == msh
                        print(f"  CHECK {vk}: img_HW={ish} mask_HW={msh} crop_res={cr} aligned={crop_match}")
                    except Exception as e:
                        print(f"  CHECK {vk}: load error {e}")
        print()


if __name__ == "__main__":
    sys.exit(main())
