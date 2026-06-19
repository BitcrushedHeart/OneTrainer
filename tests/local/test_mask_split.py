import json
import os
import time

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest


class DummySource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def length(self):
        return len(next(iter(self.data.values())))

    def get_inputs(self):
        return []

    def get_outputs(self):
        return list(self.data.keys())

    def get_item(self, variation, index, requested_name=None):
        return {k: v[index] for k, v in self.data.items()}


def _drain(ds):
    ds.start_next_epoch()
    return list(ds)


def _build(modules, output_names):
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "A", "path": "dummy"}],
        settings={},
        definition=[[m] for m in modules] + [[OutputPipelineModule(names=output_names)]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _make_cache(cache_dir, *, split_names, source_path_in_name, tolerate=False):
    return SmartDiskCache(
        cache_dir=cache_dir,
        split_names=split_names,
        aggregate_names=[],
        modeltype="test",
        source_path_in_name=source_path_in_name,
        tolerate_missing_source=tolerate,
    )


def _write(path, content):
    with open(path, "wb") as f:
        f.write(content)


def test_image_and_mask_caches_invalidate_independently(tmp_path):
    img_path = str(tmp_path / "a.png")
    mask_path = str(tmp_path / "a-masklabel.png")
    _write(img_path, b"image_v1")
    _write(mask_path, b"mask_v1")

    src = DummySource(
        {
            "image_path": [img_path],
            "mask_path": [mask_path],
            "latent_image": [torch.tensor([1.0])],
            "latent_mask": [torch.tensor([0.5])],
        }
    )
    image_cache = _make_cache(str(tmp_path / "image"), split_names=["latent_image"], source_path_in_name="image_path")
    mask_cache = _make_cache(
        str(tmp_path / "mask"), split_names=["latent_mask"], source_path_in_name="mask_path", tolerate=True
    )

    _drain(_build([src, image_cache, mask_cache], ["latent_image", "latent_mask", "image_path", "mask_path"]))

    img_pt_mtimes = {
        f: os.path.getmtime(str(tmp_path / "image" / f))
        for f in os.listdir(str(tmp_path / "image"))
        if f.endswith(".pt")
    }
    assert img_pt_mtimes
    time.sleep(0.05)

    src.data["latent_mask"] = [torch.tensor([0.9])]
    _write(mask_path, b"mask_v2_different")

    image_cache2 = _make_cache(str(tmp_path / "image"), split_names=["latent_image"], source_path_in_name="image_path")
    mask_cache2 = _make_cache(
        str(tmp_path / "mask"), split_names=["latent_mask"], source_path_in_name="mask_path", tolerate=True
    )
    out = _drain(_build([src, image_cache2, mask_cache2], ["latent_image", "latent_mask", "image_path", "mask_path"]))

    assert pytest.approx(out[0]["latent_image"].flatten().tolist()) == [1.0]
    assert pytest.approx(out[0]["latent_mask"].flatten().tolist()) == [0.9]

    for f, mt in img_pt_mtimes.items():
        assert os.path.getmtime(str(tmp_path / "image" / f)) == mt, f"image cache rewrote {f}"


def test_tolerate_missing_source_does_not_loop_rebuild(tmp_path):
    mask_path = str(tmp_path / "absent-masklabel.png")

    src = DummySource(
        {
            "mask_path": [mask_path],
            "latent_mask": [torch.tensor([0.123])],
        }
    )
    cache = _make_cache(
        str(tmp_path / "mask"), split_names=["latent_mask"], source_path_in_name="mask_path", tolerate=True
    )
    _drain(_build([src, cache], ["latent_mask", "mask_path"]))
    assert mask_path in cache.cache_index["entries"]

    pt_files = [f for f in os.listdir(str(tmp_path / "mask")) if f.endswith(".pt")]
    pt_mtimes = {f: os.path.getmtime(str(tmp_path / "mask" / f)) for f in pt_files}
    time.sleep(0.05)

    src.data["latent_mask"] = [torch.tensor([0.999])]
    cache2 = _make_cache(
        str(tmp_path / "mask"), split_names=["latent_mask"], source_path_in_name="mask_path", tolerate=True
    )
    out = _drain(_build([src, cache2], ["latent_mask", "mask_path"]))

    assert pytest.approx(out[0]["latent_mask"].flatten().tolist()) == [0.123]
    for f, mt in pt_mtimes.items():
        assert os.path.getmtime(str(tmp_path / "mask" / f)) == mt


def test_mask_cache_follows_upstream_when_variants_misordered(tmp_path):
    img_path = str(tmp_path / "x.png")
    mask_path = str(tmp_path / "x-masklabel.png")
    _write(img_path, b"img_bytes")
    _write(mask_path, b"mask_bytes")

    src = DummySource(
        {
            "image_path": [img_path],
            "mask_path": [mask_path],
            "latent_image": [torch.zeros(2, 8, 8)],
            "latent_mask": [torch.full((1, 8, 8), 0.7)],
            "crop_resolution": [(64, 64)],
        }
    )
    image_cache = SmartDiskCache(
        cache_dir=str(tmp_path / "image"),
        split_names=["latent_image"],
        aggregate_names=["crop_resolution"],
        modeltype="test",
        source_path_in_name="image_path",
    )
    mask_cache = SmartDiskCache(
        cache_dir=str(tmp_path / "mask"),
        split_names=["latent_mask"],
        aggregate_names=[],
        modeltype="test",
        source_path_in_name="mask_path",
        tolerate_missing_source=True,
        resolution_from_upstream=True,
    )
    _drain(
        _build(
            [src, image_cache, mask_cache],
            ["latent_image", "latent_mask", "image_path", "mask_path", "crop_resolution"],
        )
    )

    mask_index_path = tmp_path / "mask" / "cache.json"
    idx = json.loads(mask_index_path.read_text())
    entry = idx["entries"][mask_path]
    real_variant_key = next(iter(entry["variants"].keys()))
    real_cache_file = entry["variants"][real_variant_key]["cache_file"]

    fake_variant_key = "16x16"
    fake_cache_file = real_cache_file.split("_")[0] + "_" + fake_variant_key
    fake_pt_dest = tmp_path / "mask" / f"{fake_cache_file}_1.pt"
    torch.save(
        {"latent_mask": torch.full((1, 2, 2), 0.99), "__cache_version": 3, "__modeltype": "test"}, str(fake_pt_dest)
    )

    entry["variants"] = {
        fake_variant_key: {"cache_file": fake_cache_file, "schema_keys": ["latent_mask"]},
        **entry["variants"],
    }
    mask_index_path.write_text(json.dumps(idx))

    image_cache2 = SmartDiskCache(
        cache_dir=str(tmp_path / "image"),
        split_names=["latent_image"],
        aggregate_names=["crop_resolution"],
        modeltype="test",
        source_path_in_name="image_path",
    )
    mask_cache2 = SmartDiskCache(
        cache_dir=str(tmp_path / "mask"),
        split_names=["latent_mask"],
        aggregate_names=[],
        modeltype="test",
        source_path_in_name="mask_path",
        tolerate_missing_source=True,
        resolution_from_upstream=True,
    )
    out = _drain(_build([src, image_cache2, mask_cache2], ["latent_image", "latent_mask", "image_path", "mask_path"]))
    assert out[0]["latent_mask"].shape[-2:] == (8, 8), (
        f"mask shape {out[0]['latent_mask'].shape} should match upstream-chosen variant"
    )
    assert pytest.approx(out[0]["latent_mask"].flatten().tolist()) == [0.7] * 64


def test_extras_in_pt_are_tolerated_when_split_shrinks(tmp_path):
    img_path = str(tmp_path / "img.png")
    _write(img_path, b"image_bytes")

    src1 = DummySource(
        {
            "image_path": [img_path],
            "latent_image": [torch.tensor([1.0])],
            "latent_mask": [torch.tensor([0.5])],
        }
    )
    cache_v1 = _make_cache(
        str(tmp_path / "image"), split_names=["latent_image", "latent_mask"], source_path_in_name="image_path"
    )
    _drain(_build([src1, cache_v1], ["latent_image", "latent_mask", "image_path"]))

    pt_files = [f for f in os.listdir(str(tmp_path / "image")) if f.endswith(".pt")]
    pt_mtimes = {f: os.path.getmtime(str(tmp_path / "image" / f)) for f in pt_files}
    time.sleep(0.05)

    src2 = DummySource(
        {
            "image_path": [img_path],
            "latent_image": [torch.tensor([1.0])],
        }
    )
    cache_v2 = _make_cache(str(tmp_path / "image"), split_names=["latent_image"], source_path_in_name="image_path")
    out = _drain(_build([src2, cache_v2], ["latent_image", "image_path"]))

    assert pytest.approx(out[0]["latent_image"].flatten().tolist()) == [1.0]
    for f, mt in pt_mtimes.items():
        assert os.path.getmtime(str(tmp_path / "image" / f)) == mt
