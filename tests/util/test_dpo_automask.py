"""Tests for scripts/util/dpo_automask.py on synthetic pairs.

A textured base image stands in for the rejected image; the chosen image is
that base with an edited patch, a global brightness shift plus noise, nothing,
or a widespread change. Checks the mask shape (dilated well past the edit,
feathered), the flags, the apply / undo round trip in a temp dataset and the
review server's refusal to serve anything but listed pairs.
"""

import importlib.util
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


automask = _load("dpo_automask_script", "scripts/util/dpo_automask.py")

H, W = 384, 512


def _base(seed: int = 0) -> np.ndarray:
    """Smooth colour field plus fine texture, float32 HxWx3 in 0..1."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((6, 8, 3)).astype(np.float32)
    smooth = (
        np.asarray(Image.fromarray((coarse * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), dtype=np.float32)
        / 255
    )
    texture = rng.normal(0, 0.03, (H, W, 1)).astype(np.float32)
    return np.clip(0.25 + 0.5 * smooth + texture, 0, 1)


def _reencode_noise(image: np.ndarray, seed: int, shift: float = 0.015, sigma: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(image + shift + rng.normal(0, sigma, image.shape).astype(np.float32), 0, 1)


def _with_patch(image: np.ndarray, y0=150, x0=220, size=40) -> np.ndarray:
    edited = image.copy()
    rng = np.random.default_rng(99)
    edited[y0 : y0 + size, x0 : x0 + size] = np.clip(
        edited[y0 : y0 + size, x0 : x0 + size] * 0.5 + rng.random((size, size, 3)) * 0.4, 0, 1
    )
    return edited


def _save(path, image: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def _score_and_mask(chosen, rejected, params=None):
    params = params or automask.MaskParams()
    difference = automask.difference_score(chosen, rejected, params)
    soft, raw, final, flags = automask.mask_from_score(difference.score, params, difference.background)
    return soft, raw, final, flags, difference.size_mismatch


def test_edited_patch_found_dilated_and_feathered():
    rejected = _base()
    chosen = _reencode_noise(_with_patch(rejected), seed=1)
    soft, raw, final, flags, mismatch = _score_and_mask(chosen, rejected)
    assert flags == ["ok"] and not mismatch
    patch = np.zeros((H, W), bool)
    patch[150:190, 220:260] = True
    # The edit itself is fully inside the mask core.
    assert soft[patch].min() > 0.99
    # The mask reaches well beyond the edited pixels: 3% of 384 = ~12px of
    # dilation on each side, so the core is more than double the patch area.
    core = soft >= 0.99
    assert core.sum() > 2.0 * patch.sum()
    assert core.sum() > raw * H * W
    # Feather: a smooth ring of intermediate values around the core.
    intermediate = (soft > 0.05) & (soft < 0.95)
    assert intermediate.sum() > 0.3 * core.sum()
    # And nothing lights up far away from the edit.
    far = soft.copy()
    far[100:240, 170:310] = 0
    assert far.max() == 0.0
    assert final < 0.1


def test_global_shift_and_noise_is_not_a_full_mask():
    rejected = _base(seed=3)
    chosen = _reencode_noise(rejected, seed=4, shift=0.02, sigma=0.012)
    soft, raw, final, flags, _ = _score_and_mask(chosen, rejected)
    assert "global" not in flags
    assert final < 0.15


def test_identical_pair_flags_none():
    rejected = _base(seed=5)
    soft, raw, final, flags, _ = _score_and_mask(rejected.copy(), rejected)
    assert soft is None and flags == ["none"] and raw == 0.0 and final == 0.0


def test_widespread_change_flags_global():
    rejected = _base(seed=6)
    chosen = _base(seed=7)  # entirely different content
    soft, raw, final, flags, _ = _score_and_mask(chosen, rejected)
    assert flags == ["global"]
    assert soft is not None and soft.min() == 1.0


def test_size_mismatch_is_flagged_and_resized():
    rejected = _base(seed=8)
    chosen = _reencode_noise(_with_patch(rejected), seed=9)
    small = (
        np.asarray(Image.fromarray((rejected * 255).astype(np.uint8)).resize((W // 2, H // 2)), dtype=np.float32) / 255
    )
    difference = automask.difference_score(chosen, small, automask.MaskParams())
    assert difference.size_mismatch and difference.score.shape == (H, W)


def _make_dataset(root: Path) -> dict:
    """Three pairs: 'a' (patch, png/png), 'b' (patch, png chosen / webp
    rejected like the loader's extension fallback) and 'c' (already masked)."""
    base = _base(seed=10)
    _save(root / "chosen" / "train" / "a.png", _reencode_noise(_with_patch(base), 11))
    _save(root / "rejected" / "train" / "a.png", base)
    _save(root / "chosen" / "train" / "b.png", _reencode_noise(_with_patch(base, 60, 60, 50), 12))
    Image.fromarray((base * 255).astype(np.uint8)).save(root / "rejected" / "train" / "b.webp", lossless=True)
    _save(root / "chosen" / "train" / "c.png", _reencode_noise(_with_patch(base), 13))
    _save(root / "rejected" / "train" / "c.png", base)
    existing = np.zeros((H, W), np.uint8)
    existing[100:200, 100:200] = 255
    Image.fromarray(existing, "L").save(root / "chosen" / "train" / "c-masklabel.png")
    Image.fromarray(existing, "L").save(root / "rejected" / "train" / "c-masklabel.png")
    # Files the tool must never touch.
    (root / "chosen" / "train" / "a.txt").write_text("caption")
    (root / "chosen" / "train" / "a.png.originalbak").write_bytes(b"bak")
    return {"existing_mask": existing}


def test_pairing_mirrors_loader(tmp_path):
    _make_dataset(tmp_path)
    pairs, missing = automask.list_pairs(str(tmp_path), "chosen/{}", "rejected/{}")
    assert missing == 0
    assert [p.key for p in pairs] == ["train/a", "train/b", "train/c"]
    assert pairs[1].rejected.endswith("b.webp")
    assert automask.mask_path_for(pairs[1].rejected).endswith("b-masklabel.png")
    # The masked pair is skipped unless its mask is nearly full.
    selected, extra = automask.select_pairs(str(tmp_path), pairs, None)
    assert [p.key for p in selected] == ["train/a", "train/b"]
    selected, extra = automask.select_pairs(str(tmp_path), pairs, 0.01)
    assert [p.key for p in selected] == ["train/a", "train/b", "train/c"] and extra == {"train/c": ["had_mask"]}


def test_propose_apply_undo_round_trip(tmp_path):
    info = _make_dataset(tmp_path)
    stats = automask.run_propose(str(tmp_path), automask.MaskParams(), workers=1)
    assert stats["proposed"] == 2 and stats["flags"] == {"ok": 2}
    proposals = automask.load_proposals(str(tmp_path))
    assert set(proposals["pairs"]) == {"train/a", "train/b"}
    proposal_a = tmp_path / ".automask" / "proposals" / "train" / "a-masklabel.png"
    assert proposal_a.is_file()
    with Image.open(proposal_a) as im:
        assert im.mode == "L"
        values = np.unique(np.asarray(im))
    assert len(values) > 2  # soft, not binary

    # Nothing outside .automask was written by propose.
    train = tmp_path / "chosen" / "train"
    assert sorted(p.name for p in train.iterdir()) == [
        "a.png",
        "a.png.originalbak",
        "a.txt",
        "b.png",
        "c-masklabel.png",
        "c.png",
    ]

    automask.save_decisions(
        str(tmp_path),
        {"decisions": {"train/a": {"decision": "accept"}, "train/b": {"decision": "reject"}}},
    )
    result = automask.run_apply(str(tmp_path))
    assert result["pairs_written"] == 1
    for side in ("chosen", "rejected"):
        target = tmp_path / side / "train" / "a-masklabel.png"
        assert target.read_bytes() == proposal_a.read_bytes()
    assert not (tmp_path / "chosen" / "train" / "b-masklabel.png").exists()
    log = json.loads((tmp_path / ".automask" / "applied.json").read_text())
    assert len(log["runs"]) == 1 and len(log["runs"][0]["entries"]) == 2

    # A second apply skips the now-existing masks unless --overwrite.
    assert automask.run_apply(str(tmp_path)) == {"skipped_existing": 1, "run_id": None}

    undo = automask.run_undo(str(tmp_path))
    assert undo["removed"] == 2
    assert not (tmp_path / "chosen" / "train" / "a-masklabel.png").exists()
    assert not (tmp_path / "rejected" / "train" / "a-masklabel.png").exists()
    assert automask.run_undo(str(tmp_path))["undone_run"] is None

    # Overwrite path: the existing mask of 'c' is backed up and restored.
    automask.run_propose(str(tmp_path), automask.MaskParams(), include_covered=0.01, workers=1)
    automask.save_decisions(str(tmp_path), {"decisions": {"train/c": {"decision": "accept"}}})
    assert automask.run_apply(str(tmp_path)) == {"skipped_existing": 1, "run_id": None}
    result = automask.run_apply(str(tmp_path), overwrite=True)
    assert result["pairs_written"] == 1 and result["overwritten"] == 2
    new_mask = np.asarray(Image.open(tmp_path / "chosen" / "train" / "c-masklabel.png"))
    assert not np.array_equal(new_mask, info["existing_mask"])
    automask.run_undo(str(tmp_path))
    for side in ("chosen", "rejected"):
        restored = np.asarray(Image.open(tmp_path / side / "train" / "c-masklabel.png"))
        assert np.array_equal(restored, info["existing_mask"])


def test_review_server_serves_only_listed_pairs(tmp_path):
    _make_dataset(tmp_path)
    automask.run_propose(str(tmp_path), automask.MaskParams(), workers=1)
    server = automask.make_server(str(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=30) as response:
                return response.status, response.headers.get("Content-Type"), response.read()
        except urllib.error.HTTPError as e:
            return e.code, None, b""

    try:
        status, ctype, body = get("/")
        assert status == 200 and ctype.startswith("text/html")
        status, ctype, body = get("/api/pairs?filter=all")
        assert status == 200 and len(json.loads(body)) == 2
        assert set(automask.PREVIEW_KINDS) == {
            "chosen",
            "rejected",
            "thumb",
            "overlay",
            "heat",
            "chosen_full",
            "rejected_full",
            "overlay_full",
            "overlay_rejected_full",
            "heat_full",
        }
        for kind in automask.PREVIEW_KINDS:
            status, ctype, body = get(f"/preview/0/{kind}.jpg?threshold=4&dilate=0.05")
            assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8", kind
        # Full kinds are native resolution; the small ones are downscaled.
        with Image.open(io.BytesIO(get("/preview/1/overlay_rejected_full.jpg")[2])) as im:
            assert im.size == (W, H)
        with Image.open(io.BytesIO(get("/preview/1/heat_full.jpg")[2])) as im:
            assert im.size == (W, H)
        with Image.open(io.BytesIO(get("/preview/1/thumb.jpg")[2])) as im:
            assert max(im.size) == automask.THUMB_EDGE < W
        assert not any("full" in name for name in os.listdir(tmp_path / ".automask" / "previews"))
        status, _, body = get("/api/pair/1?threshold=3")
        assert status == 200 and json.loads(body)["live"]["flags"] == ["ok"]

        # Anything that is not a listed pair index or a listed kind is refused.
        for kind in automask.PREVIEW_KINDS:
            assert get(f"/preview/2/{kind}.jpg")[0] == 404, kind
            assert get(f"/preview/-1/{kind}.jpg")[0] == 404, kind
        assert get("/preview/0/full.jpg")[0] == 404
        assert get("/preview/0/original.jpg")[0] == 404
        assert get("/preview/0/..%2F..%2Fchosen%2Ftrain%2Fa.png")[0] == 404
        assert get("/chosen/train/a.png")[0] == 404
        assert get("/.automask/proposals.json")[0] == 404
        assert get("/../chosen/train/a.txt")[0] == 404
        assert get("/api/pair/7")[0] == 404

        # A decision persists immediately.
        request = urllib.request.Request(
            base + "/api/decision",
            data=json.dumps({"index": 0, "decision": "accept"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert json.loads(response.read())["decision"] == "accept"
        decisions = json.loads((tmp_path / ".automask" / "decisions.json").read_text())
        assert decisions["decisions"]["train/a"]["decision"] == "accept"
    finally:
        server.shutdown()
        server.server_close()
