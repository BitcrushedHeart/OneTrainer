import json
from pathlib import Path

from PIL import Image, PngImagePlugin

from modules.util.distill_metadata_util import (
    DISTILL_MANIFEST_NAME,
    DistillBuildFilters,
    DistillBuildRequest,
    build_distillation_dataset,
    extract_distill_metadata,
)


def test_extracts_swarm_utf16_webp_metadata(tmp_path: Path):
    image = tmp_path / "sample.webp"
    payload = {
        "sui_image_params": {
            "prompt": "a confirmed good image",
            "negativeprompt": "",
            "cfgscale": 1.0,
            "steps": 12,
            "scheduler": "simple",
            "seed": 123,
            "model": "SoReal!_Distilled_V0.93",
            "width": 832,
            "height": 1216,
        }
    }
    image.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + json.dumps(payload).encode("utf-16-le"))

    meta = extract_distill_metadata(str(image))

    assert meta is not None
    assert meta.prompt == "a confirmed good image"
    assert meta.cfg_scale == 1.0
    assert meta.steps == 12
    assert meta.scheduler == "simple"
    assert meta.seed == 123
    assert meta.model_id == "SoReal!_Distilled_V0.93"
    assert meta.width == 832
    assert meta.height == 1216


def test_extracts_png_text_json_metadata(tmp_path: Path):
    image = tmp_path / "sample.png"
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text(
        "sui_image_params",
        json.dumps(
            {
                "prompt": "png prompt",
                "cfg_scale": 2.5,
                "steps": 8,
                "scheduler": "FlowMatchEulerDiscrete",
                "model_id": "teacher",
            }
        ),
    )
    Image.new("RGB", (8, 8), "white").save(image, pnginfo=pnginfo)

    meta = extract_distill_metadata(str(image))

    assert meta is not None
    assert meta.prompt == "png prompt"
    assert meta.cfg_scale == 2.5
    assert meta.steps == 8
    assert meta.scheduler == "FlowMatchEulerDiscrete"
    assert meta.model_id == "teacher"


def test_sidecar_takes_precedence_over_embedded_metadata(tmp_path: Path):
    image = tmp_path / "sample.webp"
    image.write_bytes(
        b"RIFF\x00\x00\x00\x00WEBP"
        + json.dumps({"sui_image_params": {"prompt": "embedded", "steps": 1}}).encode("utf-16-le")
    )
    image.with_suffix(".json").write_text(
        json.dumps({"prompt": "sidecar", "steps": 4, "cfg_scale": 1.0, "scheduler": "simple"}),
        encoding="utf-8",
    )

    meta = extract_distill_metadata(str(image))

    assert meta is not None
    assert meta.prompt == "sidecar"
    assert meta.steps == 4


def test_build_distillation_dataset_copies_images_sidecars_and_manifest(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "distill"
    source.mkdir()

    good = source / "good.webp"
    good.write_bytes(
        b"RIFF\x00\x00\x00\x00WEBP"
        + json.dumps(
            {
                "sui_image_params": {
                    "prompt": "good prompt",
                    "cfgscale": 1.0,
                    "steps": 12,
                    "scheduler": "simple",
                    "model": "SoReal!_Distilled_V0.93",
                }
            }
        ).encode("utf-16-le")
    )
    wrong_steps = source / "wrong.webp"
    wrong_steps.write_bytes(
        b"RIFF\x00\x00\x00\x00WEBP"
        + json.dumps(
            {
                "sui_image_params": {
                    "prompt": "wrong prompt",
                    "cfgscale": 1.0,
                    "steps": 8,
                    "scheduler": "simple",
                    "model": "SoReal!_Distilled_V0.93",
                }
            }
        ).encode("utf-16-le")
    )
    missing = source / "missing.webp"
    missing.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

    result = build_distillation_dataset(
        DistillBuildRequest(
            source_folder=str(source),
            output_folder=str(output),
            val_percentage=0.0,
            filters=DistillBuildFilters(model_contains="V0.93", steps=12, cfg_scale=1.0, scheduler="simple"),
        )
    )

    assert result["ok"] is True
    assert result["scanned"] == 3
    assert result["accepted"] == 1
    assert result["rejected"]["steps_mismatch"] == 1
    assert result["rejected"]["missing_metadata"] == 1
    copied = output / "train" / "good.webp"
    assert copied.is_file()
    assert copied.with_suffix(".json").is_file()
    assert copied.with_suffix(".txt").read_text(encoding="utf-8") == "good prompt"
    manifest = json.loads((output / DISTILL_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["accepted"] == 1
    assert manifest["items"][0]["metadata"]["prompt"] == "good prompt"
