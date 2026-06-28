import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from web.backend.routers.distill import router


def test_distill_build_dataset_endpoint_returns_manifest(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    image_path = source_dir / "pair_0008.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "prompt": "a certified good image",
                "steps": 12,
                "cfg_scale": 1,
                "model_id": "SoReal!_Distilled_V0.93",
                "scheduler": "flowmatch",
                "width": 8,
                "height": 8,
            }
        ),
        encoding="utf-8",
    )

    app = FastAPI()
    app.include_router(router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/distill/build-dataset",
        json={
            "source_folder": str(source_dir),
            "output_folder": str(output_dir),
            "include_subdirectories": False,
            "val_percentage": 0.0,
            "filters": {
                "model_contains": "SoReal!_Distilled_V0.93",
                "steps": 12,
                "cfg_scale": 1.0,
            },
        },
    )

    assert response.status_code == 200
    manifest = response.json()
    assert manifest["ok"] is True
    assert manifest["scanned"] == 1
    assert manifest["accepted"] == 1
    assert (output_dir / "train" / "pair_0008.png").exists()
    assert (output_dir / "distillation_manifest.json").exists()
