"""CPU-only checks for numbered mask variants."""

import os

from modules.dataLoader.SelectMaskPath import SelectMaskPath
from modules.util import path_util

from mgds.MGDS import MGDS, PipelineState
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.GenerateImageLike import GenerateImageLike
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

from PIL import Image


class DummyPaths(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def length(self):
        return len(self.paths)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ["image_path"]

    def get_item(self, variation, index, requested_name=None):
        return {"image_path": self.paths[index]}


def _write(path, value, size=(8, 8)):
    Image.new("L", size, color=value).save(path)


def _build(paths):
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "A", "path": "dummy"}],
        settings={},
        definition=[
            [DummyPaths(paths)],
            [SelectMaskPath(in_name="image_path", out_name="mask_path")],
            [
                LoadImage(
                    path_in_name="image_path",
                    image_out_name="image",
                    range_min=0,
                    range_max=1,
                    supported_extensions=path_util.supported_image_extensions(),
                )
            ],
            [GenerateImageLike(image_in_name="image", image_out_name="mask", color=255, range_min=0, range_max=1)],
            [
                LoadImage(
                    path_in_name="mask_path",
                    image_out_name="mask",
                    range_min=0,
                    range_max=1,
                    channels=1,
                    supported_extensions={".png"},
                )
            ],
            [OutputPipelineModule(names=["mask_path", "mask"])],
        ],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _items(ds, variations):
    # each epoch advances the variation index the pipeline hands to every module
    out = []
    for _ in range(variations):
        ds.start_next_epoch()
        out.append(next(iter(ds)))
    return out


def test_single_mask_is_stable_across_variations(tmp_path):
    img = str(tmp_path / "a.png")
    _write(img, 128)
    _write(str(tmp_path / "a-masklabel.png"), 200)

    ds = _build([img])
    for variation, item in enumerate(_items(ds, 4)):
        assert item["mask_path"].endswith("a-masklabel.png"), variation


def test_variants_cycle_by_variation(tmp_path):
    img = str(tmp_path / "b.png")
    _write(img, 128)
    _write(str(tmp_path / "b-masklabel.png"), 10)
    _write(str(tmp_path / "b-masklabel1.png"), 250)

    ds = _build([img])
    seen = [os.path.basename(item["mask_path"]) for item in _items(ds, 4)]
    assert seen == ["b-masklabel.png", "b-masklabel1.png", "b-masklabel.png", "b-masklabel1.png"], seen


def test_missing_mask_falls_back_to_all_ones(tmp_path):
    img = str(tmp_path / "c.png")
    _write(img, 128)

    ds = _build([img])
    item = _items(ds, 1)[0]
    assert item["mask"].min().item() == 1.0
    assert item["mask"].max().item() == 1.0


def test_is_mask_filename():
    assert path_util.is_mask_filename("x-masklabel.png")
    assert path_util.is_mask_filename("x-masklabel1.png")
    assert path_util.is_mask_filename("x-masklabel9.png")
    assert not path_util.is_mask_filename("x-masklabel10.png")
    assert not path_util.is_mask_filename("x.png")
    assert not path_util.is_mask_filename("x-condlabel.png")
    # the stem is what counts, matching how the dataset enumeration filters
    assert path_util.is_mask_filename("x-masklabel.jpg")


def test_mask_sample_targets_the_requested_variant(tmp_path):
    from modules.module.BaseImageMaskModel import MaskSample

    img = str(tmp_path / "d.png")
    _write(img, 128)

    assert MaskSample(img, torch.device("cpu")).mask_filename.endswith("d-masklabel.png")
    assert MaskSample(img, torch.device("cpu"), 0).mask_filename.endswith("d-masklabel.png")
    assert MaskSample(img, torch.device("cpu"), 3).mask_filename.endswith("d-masklabel3.png")


def test_mask_sample_reads_the_variant_it_writes(tmp_path):
    from modules.module.BaseImageMaskModel import MaskSample

    img = str(tmp_path / "e.png")
    _write(img, 128)
    _write(str(tmp_path / "e-masklabel.png"), 255)

    # variant 2 has no file yet, so 'fill' would not skip it even though the base mask exists
    assert MaskSample(img, torch.device("cpu"), 0).get_mask_tensor() is not None
    assert MaskSample(img, torch.device("cpu"), 2).get_mask_tensor() is None


def test_variant_files_are_not_collected_as_images(tmp_path):
    from modules.module.BaseImageMaskModel import BaseImageMaskModel

    _write(str(tmp_path / "f.png"), 128)
    _write(str(tmp_path / "f-masklabel.png"), 255)
    _write(str(tmp_path / "f-masklabel1.png"), 255)
    _write(str(tmp_path / "f-masklabel9.png"), 255)

    found = BaseImageMaskModel._BaseImageMaskModel__get_sample_filenames(str(tmp_path))
    assert [os.path.basename(p) for p in found] == ["f.png"]


def test_collect_paths_excludes_every_mask_variant(tmp_path):
    from mgds.pipelineModules.CollectPaths import CollectPaths

    for name in ("g.png", "g-masklabel.png", "g-masklabel1.png", "g-masklabel9.png", "g-condlabel.png", "h.jpg"):
        _write(str(tmp_path / name), 128)

    collect = CollectPaths(
        concept_in_name="concept",
        path_in_name="path",
        include_subdirectories_in_name="concept.include_subdirectories",
        enabled_in_name="enabled",
        path_out_name="image_path",
        concept_out_name="concept",
        extensions=path_util.supported_image_extensions(),
        include_postfix=None,
        exclude_postfix=[*path_util.MASK_POSTFIXES, "-condlabel"],
    )
    concept = {"path": str(tmp_path), "enabled": True, "include_subdirectories": False}
    collect._get_previous_length = lambda name: 1
    collect._get_previous_item = lambda variation, name, index: (
        concept if name == "concept" else concept["include_subdirectories"]
    )
    collect.start(0)

    assert sorted(os.path.basename(p) for p in collect.paths) == ["g.png", "h.jpg"]


def test_every_variant_is_exposed_for_cache_watching(tmp_path):
    """SmartDiskCache resolves watched sidecars at variation 0 only, so each variant
    needs its own variation-invariant name or editing a later one would go unnoticed."""
    img = str(tmp_path / "i.png")
    _write(img, 128)
    _write(str(tmp_path / "i-masklabel.png"), 10)
    _write(str(tmp_path / "i-masklabel3.png"), 250)

    names = SelectMaskPath.variant_out_names("mask_path")
    assert len(names) == path_util.MAX_MASK_VARIANTS + 1

    ds = _build([img])
    ds.start_next_epoch()
    module = next(m for m in ds.loading_pipeline.modules if isinstance(m, SelectMaskPath))

    assert set(names) <= set(module.get_outputs())

    item = module.get_item(0, 0)
    assert item["mask_path_variant0"].endswith("i-masklabel.png")
    assert item["mask_path_variant3"].endswith("i-masklabel3.png")
    # slots with no file resolve to empty, which the watcher skips
    assert item["mask_path_variant1"] == ""
    assert item["mask_path_variant9"] == ""
