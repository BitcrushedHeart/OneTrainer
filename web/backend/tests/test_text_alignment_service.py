"""Text Alignment Dataset service: scan in a thread, build, wildcard export,
with the OCR and detector halves stubbed."""

import json
import os
import time

from web.backend.services.text_alignment_service import TextAlignmentDatasetService

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(path, prompt, color):
    info = PngInfo()
    info.add_text("sui_image_params", json.dumps({"sui_image_params": {"prompt": prompt}}))
    Image.new("RGB", (48, 32), color).save(path, pnginfo=info)
    return str(path)


def _wait(service, phase, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.status()
        if status["phase"] == phase:
            return status
        if status["phase"] == "error":
            raise AssertionError(status)
        time.sleep(0.02)
    raise AssertionError(f"timed out: {service.status()}")


def test_scan_build_and_wildcard(tmp_path):
    src = tmp_path / "swarm"
    src.mkdir()
    wrong = _png(src / "w.png", 'Neon text reading "Open Daily"', (1, 2, 3))
    _png(src / "v.png", 'Neon text reading "Welcome"', (4, 5, 6))
    out = tmp_path / "ds"

    service = TextAlignmentDatasetService()
    service.read_text_factory = lambda device: lambda p: "Opem Daily" if p == wrong else "Welcome"
    service.count_boxed_factory = lambda device: lambda p, words: len(words)

    assert service.scan(str(tmp_path / "missing"), str(out))["ok"] is False
    assert service.scan(str(src), str(out))["ok"] is True
    status = _wait(service, "scanned")
    assert status["scanned"] == 2 and status["kept"] == 1
    assert status["counts"]["keep"] == 1 and status["counts"]["verbatim"] == 1

    results = service.results()
    assert results["total"] == 1
    item = results["items"][0]
    assert item["ocr_reading"] == "Opem Daily" and item["boxed_words"] == 2 and item["keep"] is True

    assert service.build()["ok"] is True
    status = _wait(service, "built")
    assert status["build"] == {"written": 1, "skipped": 0, "cancelled": False}
    assert sorted(f for f in os.listdir(out) if f.endswith(".json")) == sorted(
        f for f in os.listdir(out) if f.endswith(".json")
    )
    assert any(f.endswith(".txt") for f in os.listdir(out))

    assert service.build()["ok"] is True  # nothing new to write
    assert _wait(service, "built")["build"]["written"] == 0

    wc = TextAlignmentDatasetService.wildcard(count=12, seed=3, name="ta_test", directory=str(tmp_path / "wc"))
    assert wc["ok"] and wc["count"] == 12 and len(wc["sample"]) == 5
    with open(wc["path"], encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line]
    assert len(lines) == 12 and all('text reading "' in line for line in lines)
