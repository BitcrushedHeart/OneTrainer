"""Build a small scratch copy of the real image cache, migrate it, then load it through SmartDiskCache to confirm validation passes."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _migrate_split_mask_cache import derive_mask_path, migrate

REAL_IMAGE_CACHE = r"F:\workspace\SoReal!\cache\image"
SAMPLE_N = 5


class StubSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, image_paths, mask_paths):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def length(self):
        return len(self.image_paths)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ["image_path", "mask_path"]

    def get_item(self, variation, index, requested_name=None):
        return {"image_path": self.image_paths[index], "mask_path": self.mask_paths[index]}


def main():
    if not os.path.isdir(REAL_IMAGE_CACHE):
        print(f"missing: {REAL_IMAGE_CACHE}")
        return 1

    real_index_path = os.path.join(REAL_IMAGE_CACHE, "cache.json")
    with open(real_index_path, "r", encoding="utf-8") as f:
        real_index = json.load(f)

    real_entries = real_index.get("entries", {})
    sample_paths = list(real_entries.keys())[:SAMPLE_N]
    print(f"sample paths: {len(sample_paths)}")
    for p in sample_paths:
        print(f"  {p}")

    scratch = tempfile.mkdtemp(prefix="ot_mask_split_smoke_")
    scratch_image = os.path.join(scratch, "image")
    scratch_mask = os.path.join(scratch, "mask")
    os.makedirs(scratch_image, exist_ok=True)

    sample_index = {
        "version": real_index.get("version", 3),
        "entries": {p: real_entries[p] for p in sample_paths},
        "hash_index": {},
        "schema": real_index.get("schema"),
        "schema_method": real_index.get("schema_method", "shape_v1"),
        "last_validated": real_index.get("last_validated", 0),
        "blank_sentinel": real_index.get("blank_sentinel", "blank_sentinel.pt"),
    }
    for p, e in sample_index["entries"].items():
        h = e.get("hash")
        if h:
            sample_index["hash_index"].setdefault(h, []).append(p)

    pt_count = 0
    for p, e in sample_index["entries"].items():
        for variant in e.get("variants", {}).values():
            cf = variant["cache_file"]
            v = 1
            while True:
                src_pt = os.path.join(REAL_IMAGE_CACHE, f"{cf}_{v}.pt")
                if not os.path.exists(src_pt):
                    break
                shutil.copy2(src_pt, os.path.join(scratch_image, f"{cf}_{v}.pt"))
                pt_count += 1
                v += 1

    sentinel_name = sample_index["blank_sentinel"]
    sentinel_src = os.path.join(REAL_IMAGE_CACHE, sentinel_name)
    if os.path.isfile(sentinel_src):
        shutil.copy2(sentinel_src, os.path.join(scratch_image, sentinel_name))

    with open(os.path.join(scratch_image, "cache.json"), "w", encoding="utf-8") as f:
        json.dump(sample_index, f, indent=2)

    print(f"\nscratch copy: {scratch_image}")
    print(f"copied {pt_count} pt files")
    print("running migration --apply on scratch...")

    stats = migrate(scratch_image, scratch_mask, apply=True, sample=None)
    print()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nverifying rewritten image .pt files load via torch...")
    bad = 0
    for pt in Path(scratch_image).glob("*.pt"):
        try:
            data = torch.load(str(pt), weights_only=False, map_location="cpu")
            if not isinstance(data, dict) or "latent_image" not in data:
                print(f"  BAD: {pt.name} missing latent_image")
                bad += 1
            elif "latent_mask" in data and pt.name != sentinel_name:
                print(f"  BAD: {pt.name} still has latent_mask")
                bad += 1
        except Exception as e:
            print(f"  EXC: {pt.name}: {e}")
            bad += 1
    print(f"image .pt files OK: {pt_count - bad}/{pt_count}")

    print("\nverifying mask cache .pt files load via torch...")
    mask_bad = 0
    mask_pts = list(Path(scratch_mask).glob("*.pt"))
    for pt in mask_pts:
        try:
            data = torch.load(str(pt), weights_only=False, map_location="cpu")
            if not isinstance(data, dict) or "latent_mask" not in data:
                print(f"  BAD: {pt.name} missing latent_mask")
                mask_bad += 1
        except Exception as e:
            print(f"  EXC: {pt.name}: {e}")
            mask_bad += 1
    print(f"mask .pt files OK: {len(mask_pts) - mask_bad}/{len(mask_pts)}")

    print("\nrunning SmartDiskCache validation against scratch caches...")
    modeltype = sample_index["entries"][sample_paths[0]].get("modeltype", "")

    image_cache = SmartDiskCache(
        cache_dir=scratch_image,
        split_names=["latent_image", "original_resolution", "crop_offset"],
        aggregate_names=["crop_resolution", "image_path"],
        modeltype=modeltype,
        source_path_in_name="image_path",
    )
    image_cache.cache_index = image_cache._load_cache_index()
    image_cache._existing_pt_files = image_cache._scan_existing_pt_files()
    img_status = []
    for fp in sample_paths:
        entry = image_cache.cache_index["entries"].get(fp)
        if entry is None:
            img_status.append((fp, "no_entry"))
            continue
        for res_key in entry.get("variants", {}).keys():
            try:
                mtime = os.path.getmtime(fp)
            except OSError:
                mtime = None
            status = image_cache._validate_entry(fp, entry, res_key, 1, mtime)
            img_status.append((fp, res_key, status))
    print("  image cache validation:")
    for s in img_status:
        print(f"    {s}")

    mask_cache = SmartDiskCache(
        cache_dir=scratch_mask,
        split_names=["latent_mask"],
        aggregate_names=[],
        modeltype=modeltype,
        source_path_in_name="mask_path",
        tolerate_missing_source=True,
    )
    mask_cache.cache_index = mask_cache._load_cache_index()
    mask_cache._existing_pt_files = mask_cache._scan_existing_pt_files()
    mask_status = []
    for fp in sample_paths:
        mp = derive_mask_path(fp)
        entry = mask_cache.cache_index["entries"].get(mp)
        if entry is None:
            mask_status.append((mp, "no_entry"))
            continue
        for res_key in entry.get("variants", {}).keys():
            try:
                mtime = os.path.getmtime(mp)
            except OSError:
                mtime = None
            status = mask_cache._validate_entry(mp, entry, res_key, 1, mtime)
            mask_status.append((mp, res_key, status))
    print("  mask cache validation:")
    for s in mask_status:
        print(f"    {s}")

    img_valid = sum(1 for s in img_status if len(s) == 3 and s[2] == "valid")
    mask_valid = sum(1 for s in mask_status if len(s) == 3 and s[2] == "valid")
    print(f"\n  image entries 'valid': {img_valid}")
    print(f"  mask entries 'valid':  {mask_valid}")

    print(f"\nscratch dir kept for inspection: {scratch}")
    return 0 if bad == 0 and mask_bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
