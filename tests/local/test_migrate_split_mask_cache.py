import json
import os
import sys
from pathlib import Path

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "util"))

from _migrate_split_mask_cache import migrate


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


def _drain(modules, output_names):
    pipe = MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "A", "path": "dummy"}],
        settings={},
        definition=[[m] for m in modules] + [[OutputPipelineModule(names=output_names)]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    pipe.start_next_epoch()
    return list(pipe)


def _build_legacy_image_cache(tmp_path, n=3):
    src_dir = tmp_path / "sources"
    src_dir.mkdir()
    image_paths, mask_paths = [], []
    for i in range(n):
        img = src_dir / f"img_{i}.jpg"
        msk = src_dir / f"img_{i}-masklabel.png"
        img.write_bytes(f"image_{i}_bytes".encode())
        msk.write_bytes(f"mask_{i}_bytes".encode())
        image_paths.append(str(img))
        mask_paths.append(str(msk))

    src = DummySource(
        {
            "image_path": image_paths,
            "latent_image": [torch.tensor([float(i), float(i + 1)]) for i in range(n)],
            "latent_mask": [torch.tensor([float(i) * 0.1]) for i in range(n)],
        }
    )
    image_cache_dir = tmp_path / "image"
    cache = SmartDiskCache(
        cache_dir=str(image_cache_dir),
        split_names=["latent_image", "latent_mask"],
        aggregate_names=[],
        modeltype="testmodel",
        source_path_in_name="image_path",
    )
    _drain([src, cache], ["latent_image", "latent_mask", "image_path"])
    return image_paths, mask_paths, str(image_cache_dir)


def test_migrate_dry_run_does_not_modify(tmp_path):
    _, _, image_cache_dir = _build_legacy_image_cache(tmp_path, n=2)
    cache_json_before = (Path(image_cache_dir) / "cache.json").read_text()
    pt_before = {p.name: p.read_bytes() for p in Path(image_cache_dir).glob("*.pt")}

    stats = migrate(image_cache_dir, str(tmp_path / "mask"), apply=False, sample=None)

    assert stats["entries"] == 2
    assert stats["variants_migrated"] >= 2
    assert (Path(image_cache_dir) / "cache.json").read_text() == cache_json_before
    for name, content in pt_before.items():
        assert (Path(image_cache_dir) / name).read_bytes() == content
    assert not (tmp_path / "mask").exists()


def test_migrate_apply_produces_loadable_split_caches(tmp_path):
    image_paths, mask_paths, image_cache_dir = _build_legacy_image_cache(tmp_path, n=3)
    mask_cache_dir = str(tmp_path / "mask")

    stats = migrate(image_cache_dir, mask_cache_dir, apply=True, sample=None)
    assert stats["entries"] == 3
    assert stats["variants_migrated"] == 3
    assert stats["image_pts_rewritten"] == 3
    assert stats["bytes_saved"] > 0

    mask_index = json.loads((Path(mask_cache_dir) / "cache.json").read_text())
    assert mask_index["version"] == 3
    for mp in mask_paths:
        assert mp in mask_index["entries"]
        entry = mask_index["entries"][mp]
        assert entry["modeltype"] == "testmodel"
        for variant in entry["variants"].values():
            assert variant["schema_keys"] == ["latent_mask"]
            cf = variant["cache_file"]
            assert (Path(mask_cache_dir) / f"{cf}_1.pt").is_file()

    for pt in Path(image_cache_dir).glob("*.pt"):
        data = torch.load(str(pt), weights_only=False, map_location="cpu")
        assert "latent_image" in data
        assert "latent_mask" not in data

    src = DummySource(
        {
            "image_path": image_paths,
            "mask_path": mask_paths,
            "latent_image": [torch.tensor([float(i), float(i + 1)]) for i in range(3)],
            "latent_mask": [torch.tensor([99.0]) for _ in range(3)],
        }
    )
    image_cache = SmartDiskCache(
        cache_dir=image_cache_dir,
        split_names=["latent_image"],
        aggregate_names=[],
        modeltype="testmodel",
        source_path_in_name="image_path",
    )
    mask_cache = SmartDiskCache(
        cache_dir=mask_cache_dir,
        split_names=["latent_mask"],
        aggregate_names=[],
        modeltype="testmodel",
        source_path_in_name="mask_path",
        tolerate_missing_source=True,
    )
    out = _drain([src, image_cache, mask_cache], ["latent_image", "latent_mask", "image_path", "mask_path"])

    assert len(out) == 3
    for item in out:
        latent = item["latent_image"].flatten().tolist()
        mask = item["latent_mask"].flatten().tolist()
        idx = int(latent[0])
        assert pytest.approx(latent) == [float(idx), float(idx + 1)]
        assert pytest.approx(mask) == [float(idx) * 0.1]


def test_migrate_synthetic_when_mask_file_missing(tmp_path):
    image_paths, mask_paths, image_cache_dir = _build_legacy_image_cache(tmp_path, n=1)
    os.remove(mask_paths[0])

    stats = migrate(image_cache_dir, str(tmp_path / "mask"), apply=True, sample=None)
    assert stats["synthetic"] == 1
    assert stats["variants_migrated"] == 1

    mask_index = json.loads((Path(tmp_path / "mask") / "cache.json").read_text())
    assert mask_paths[0] in mask_index["entries"]
    assert mask_index["entries"][mask_paths[0]]["mtime"] == 0.0
