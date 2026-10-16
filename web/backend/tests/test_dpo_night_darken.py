"""Night Darken on synthetic images under tmp_path: caption matching and the
brightness skip, the darkening curve, the in-place commit with backups and
Minor Edits pairs, and the restore. CPU only, no dataset involved."""

import csv
import io
import json
import os
import time

from modules.util import dpo_night_darken_util as nd
from modules.util.dpo_curation_util import compute_file_sha256, load_manifest
from web.backend.services.dpo_night_darken_service import DPONightDarkenService

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

NIGHT_CAPTION = "A woman standing on a pier, photo taken at night, city lights behind her."
DARK_CAPTION = "This image is dark and dimly lit, a man in a dark room."
DAY_CAPTION = "A woman with dark hair and dark jeans on a sunny beach."
PLAIN_CAPTION = "A man in a red jacket on a bright street."


def _gradient(size=(96, 64), lo=40, hi=250, color=(1.0, 0.9, 0.8)) -> np.ndarray:
    w, h = size
    ramp = np.linspace(lo, hi, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    rgb = np.stack([ramp * c for c in color], axis=-1)
    rgb[: h // 4, -w // 6 :] = 255  # a clipped light source in the top-right corner
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _write(path, pixels: np.ndarray, fmt: str | None = None, text: dict | None = None) -> str:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    img = Image.fromarray(pixels)
    kwargs = {}
    if text:
        info = PngInfo()
        for k, v in text.items():
            info.add_text(k, v)
        kwargs["pnginfo"] = info
    img.save(str(path), format=fmt, **kwargs)
    return str(path)


def _concept(tmp_path):
    """RLHF-style concept: chosen/train + rejected/train, four pairs."""
    root = tmp_path / "RLHF_T"
    chosen = root / "chosen" / "train"
    rejected = root / "rejected" / "train"
    specs = {
        "pair_0001": (NIGHT_CAPTION, _gradient(), "png"),
        "pair_0002": (NIGHT_CAPTION, _gradient(size=(80, 60), lo=0, hi=50), "png"),  # already dark
        "pair_0003": (DAY_CAPTION, _gradient(), "png"),  # "dark hair": no match
        "pair_0004": (PLAIN_CAPTION, _gradient(), "png"),
        "pair_0005": (DARK_CAPTION, _gradient(size=(64, 96)), "webp"),
        "pair_0006": (NIGHT_CAPTION, _gradient(size=(70, 50), lo=60, hi=200), "png"),
    }
    manifest = {"pairs": []}
    for i, (stem, (caption, pixels, ext)) in enumerate(specs.items()):
        text = {"parameters": json.dumps({"sui_image_params": {"prompt": "p", "seed": i}})} if ext == "png" else None
        _write(chosen / f"{stem}.{ext}", pixels, fmt=ext.upper(), text=text)
        _write(rejected / f"{stem}.png", np.full((32, 32, 3), 90, np.uint8))
        (chosen / f"{stem}.txt").write_text(caption, encoding="utf-8")
        manifest["pairs"].append(
            {
                "pair_id": i + 1,
                "prompt": caption,
                "chosen_file": f"train/{stem}.{ext}",
                "rejected_file": f"train/{stem}.png",
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(root)


def _minor(tmp_path):
    root = tmp_path / "DPO Minor Edits"
    (root / "chosen" / "train").mkdir(parents=True)
    (root / "rejected" / "train").mkdir(parents=True)
    _write(root / "chosen" / "train" / "pair_0000.png", np.zeros((8, 8, 3), np.uint8))
    _write(root / "rejected" / "train" / "pair_0000.png", np.zeros((8, 8, 3), np.uint8))
    (root / "manifest.json").write_text(
        json.dumps(
            {"pairs": [{"pair_id": 0, "prompt": "x", "chosen_file": "chosen/train/pair_0000.png"}], "retired": []}
        ),
        encoding="utf-8",
    )
    return str(root)


def _wait(service, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = service.status()["job"]
        if job and not job["running"]:
            assert job["error"] is None, job
            return job
        time.sleep(0.02)
    raise AssertionError("timed out")


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #


def test_phrase_matching_is_whole_word_and_lighting_aware():
    assert nd.match_night_phrases(NIGHT_CAPTION) == ["at night", "night"]
    assert "dark" in nd.match_night_phrases(DARK_CAPTION)
    assert "dimly lit" in nd.match_night_phrases(DARK_CAPTION)
    assert nd.match_night_phrases(DAY_CAPTION) == []
    assert nd.match_night_phrases("A knight in shining armour, nightly news.") == []
    assert nd.match_night_phrases("Shot in LOW-LIGHT conditions") == ["low light"]
    assert nd.match_night_phrases("the room is dark") == ["dark"]
    assert nd.match_night_phrases("") == []


# --------------------------------------------------------------------------- #
# Darkening curve
# --------------------------------------------------------------------------- #


def test_darken_identity_monotonic_and_highlight_preserving():
    pixels = _gradient()
    assert np.array_equal(nd.darken_rgb(pixels, 0.0), pixels)
    previous = nd.luma_u8(pixels)
    for strength in (0.1, 0.3, 0.5, 0.8, 1.0):
        out = nd.darken_rgb(pixels, strength)
        current = nd.luma_u8(out)
        assert np.all(current <= previous + 0.5), strength
        assert current.mean() < previous.mean()
        previous = current
    mid = pixels[32, 48].astype(float)
    high = pixels[4, 90].astype(float)
    darkened = nd.darken_rgb(pixels, 0.6)
    mid_out, high_out = darkened[32, 48].astype(float), darkened[4, 90].astype(float)
    assert high_out.mean() / max(mid_out.mean(), 1) > high.mean() / mid.mean()
    # Black stays black, and hue survives (channel ordering unchanged).
    black = np.zeros((2, 2, 3), np.uint8)
    assert np.array_equal(nd.darken_rgb(black, 1.0), black)
    warm = np.full((2, 2, 3), (200, 120, 60), np.uint8)
    w = nd.darken_rgb(warm, 0.7)[0, 0]
    assert w[0] > w[1] > w[2]


def test_save_keeps_format_and_png_text(tmp_path):
    png = _write(tmp_path / "a.png", _gradient(), text={"parameters": "{}", "ot_triage_fix": '{"k": 1}'})
    out = str(tmp_path / "a_out.png")
    result = nd.save_darkened_like(png, 0.5, out)
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.text["parameters"] == "{}"
        assert img.text["ot_triage_fix"] == '{"k": 1}'
    assert result["delta_luma"] > 5
    webp = _write(tmp_path / "b.webp", _gradient(), fmt="WEBP")
    nd.save_darkened_like(webp, 0.5, str(tmp_path / "b_out.webp"))
    with Image.open(tmp_path / "b_out.webp") as img:
        assert img.format == "WEBP"
    jpg = _write(tmp_path / "c.jpg", _gradient(), fmt="JPEG")
    nd.save_darkened_like(jpg, 0.5, str(tmp_path / "c_out.jpg"))
    with Image.open(tmp_path / "c_out.jpg") as img:
        assert img.format == "JPEG"


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #


def test_scan_selects_night_captions_that_are_not_dark(tmp_path):
    concept = _concept(tmp_path)
    result = nd.scan_candidates(concept)
    rels = [c["rel"] for c in result["candidates"]]
    assert rels == ["chosen/train/pair_0001.png", "chosen/train/pair_0005.webp", "chosen/train/pair_0006.png"]
    assert result["already_dark"] == ["chosen/train/pair_0002.png"]
    assert result["counts"] == {"images": 6, "caption_matches": 4, "already_dark": 1, "unreadable": 0, "candidates": 3}
    first = result["candidates"][0]
    assert first["phrases"] == ["at night", "night"]
    assert first["size"] == [96, 64]
    assert first["stats"]["median"] > nd.DEFAULT_DARK_MEDIAN
    # Tighter thresholds admit the dark one too.
    loose = nd.scan_candidates(concept, thresholds=nd.DarkThresholds(median=5, mean=5))
    assert loose["counts"]["candidates"] == 4


def test_minor_edits_and_off_policy_are_refused_as_sources(tmp_path):
    for name in ("DPO Minor Edits", "Off Policy 2"):
        root = tmp_path / name
        (root / "chosen").mkdir(parents=True)
        assert nd.source_exclusion_reason(str(root))
    ws = tmp_path / "ws"
    (ws / "Repairs" / "chosen").mkdir(parents=True)
    (ws / "workspace.json").write_text(json.dumps({"datasets": [{"dir": "Repairs", "type": "minor_edits"}]}))
    assert "minor_edits" in nd.source_exclusion_reason(str(ws / "Repairs"))
    assert nd.source_exclusion_reason(_concept(tmp_path)) is None


# --------------------------------------------------------------------------- #
# Review, commit, restore through the service
# --------------------------------------------------------------------------- #


def test_commit_and_restore_round_trip(tmp_path):
    concept = _concept(tmp_path)
    minor = _minor(tmp_path)
    service = DPONightDarkenService()
    assert service.open(str(tmp_path / "DPO Minor Edits"))["ok"] is False
    opened = service.open(concept)
    assert opened["ok"] and opened["counts"]["candidates"] == 0

    assert service.scan()["ok"]
    _wait(service)
    state = service.state()
    assert state["counts"] == {"candidates": 3, "accepted": 0, "accepted_uncommitted": 0, "skipped": 0, "undecided": 3}
    rels = [c["rel"] for c in service.candidates()["candidates"]]
    originals = {rel: compute_file_sha256(os.path.join(concept, rel)) for rel in rels}

    # Previews: strength 0 is the original at preview size; higher is darker.
    light = Image.open(io.BytesIO(service.preview(rels[0], 0.0)))
    dark = Image.open(io.BytesIO(service.preview(rels[0], 0.8)))
    assert light.size == dark.size == (96, 64)
    assert np.asarray(dark).mean() < np.asarray(light).mean()
    assert service.preview("chosen/train/pair_0003.png", 0.5) is None
    assert service.caption(rels[0])["caption"] == NIGHT_CAPTION

    # Decide: two accepted at different strengths, one skipped, undo one and re-accept.
    assert service.decide(rels[0], "accepted", 0.9)["ok"]
    assert service.decide(rels[1], "skipped")["ok"]
    assert service.decide(rels[2], "accepted", 0.2)["ok"]
    assert service.undo(rels[1])["removed"]
    assert service.decide(rels[1], "accepted", 0.5)["ok"]
    decisions = {c["rel"]: c["decision"] for c in service.candidates()["candidates"]}
    assert decisions[rels[0]]["delta_luma"] > decisions[rels[2]]["delta_luma"]
    assert service.state()["counts"]["accepted_uncommitted"] == 3

    # A commit that would collide with pairs the manifest does not know is refused.
    stray = os.path.join(minor, "chosen", "train", "pair_0001.png")
    _write(stray, np.zeros((4, 4, 3), np.uint8))
    assert "already exist" in service.commit(minor, 0.34)["error"]
    os.remove(stray)

    started = service.commit(minor, 0.34)
    assert started == {"ok": True, "pending": 3, "pairs_planned": 1}
    job = _wait(service)
    assert job["phase"] == "done"
    state = service.state()
    assert state["counts"]["accepted_uncommitted"] == 0
    (commit,) = state["commits"]
    assert commit["items"] == 3 and commit["pairs"] == 1 and not commit["restored"]

    # In place: same names, same formats, darker, PNG text kept, backups are the originals.
    for rel in rels:
        path = os.path.join(concept, rel)
        assert compute_file_sha256(path) != originals[rel]
        backup = os.path.join(nd.backup_dir(concept), "train", os.path.basename(rel))
        assert compute_file_sha256(backup) == originals[rel]
        with Image.open(path) as now, Image.open(backup) as was:
            assert now.format == was.format and now.size == was.size
            assert np.asarray(now.convert("RGB")).mean() < np.asarray(was.convert("RGB")).mean()
            if now.format == "PNG":
                assert now.text["parameters"] == was.text["parameters"]
    with open(os.path.join(nd.backup_dir(concept), nd.BACKUP_MANIFEST), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["chosen"] for r in rows] == rels
    assert sum(1 for r in rows if r["minor_edits_pair_id"]) == 1
    entry = next(e for e in load_manifest(concept)["pairs"] if e["chosen_file"] == "train/pair_0001.png")
    assert entry["history"][-1]["action"] == nd.HISTORY_ACTION and entry["chosen_sha256"]

    # The most changed accepted image (strength 0.9) became the pair, numbered after pair_0000.
    minor_manifest = load_manifest(minor)
    assert [e["pair_id"] for e in minor_manifest["pairs"]] == [0, 1]
    pair = minor_manifest["pairs"][1]
    assert pair["chosen_file"] == "train/pair_0001.png" and pair["rejected_file"] == "train/pair_0001.png"
    assert pair["prompt"] == NIGHT_CAPTION and pair["lineage"]["source_chosen"] == rels[0]
    chosen_path = os.path.join(minor, "chosen", "train", "pair_0001.png")
    rejected_path = os.path.join(minor, "rejected", "train", "pair_0001.png")
    assert compute_file_sha256(rejected_path) == originals[rels[0]]
    assert compute_file_sha256(chosen_path) == compute_file_sha256(os.path.join(concept, rels[0]))
    with open(os.path.join(minor, "chosen", "train", "pair_0001.txt"), encoding="utf-8") as f:
        assert f.read() == NIGHT_CAPTION
    for image in (chosen_path, rejected_path):
        with Image.open(nd.mask_path_for(image)) as mask, Image.open(image) as img:
            assert mask.mode == "L" and mask.size == img.size
            assert np.asarray(mask).min() == 255

    # Decisions are frozen while committed.
    assert service.decide(rels[0], "skipped")["ok"] is False
    assert service.undo(rels[0])["ok"] is False

    # Restore: originals back byte for byte, pair gone, manifests as before.
    result = service.restore(commit["id"])
    assert result == {"ok": True, "restored": 3, "pairs_removed": 1, "missing": []}
    for rel in rels:
        assert compute_file_sha256(os.path.join(concept, rel)) == originals[rel]
    assert not os.path.exists(chosen_path) and not os.path.exists(rejected_path)
    assert not os.path.exists(nd.mask_path_for(chosen_path))
    assert not os.path.exists(os.path.join(minor, "chosen", "train", "pair_0001.txt"))
    assert [e["pair_id"] for e in load_manifest(minor)["pairs"]] == [0]
    entry = next(e for e in load_manifest(concept)["pairs"] if e["chosen_file"] == "train/pair_0001.png")
    assert "history" not in entry and "chosen_sha256" not in entry
    state = service.state()
    assert state["commits"][0]["restored"] and state["counts"]["accepted_uncommitted"] == 3
    assert service.restore(commit["id"])["restored"] == 0

    # Fraction 0 commits without touching the Minor Edits folder at all.
    assert service.commit("", 0.0)["ok"]
    _wait(service)
    assert service.state()["commits"][1]["pairs"] == 0
    assert [e["pair_id"] for e in load_manifest(minor)["pairs"]] == [0]


def test_select_minor_edits_takes_the_rounded_share_most_changed_first():
    items = [{"rel": f"r{i}", "delta_luma": float(i)} for i in range(6)]
    assert [it["rel"] for it in nd.select_minor_edits(items, 0.33)] == ["r5", "r4"]
    assert [it["rel"] for it in nd.select_minor_edits(items, 0.01)] == ["r5"]
    assert nd.select_minor_edits(items, 0.0) == []
    assert nd.minor_edits_count(3, 0.34) == 1 and nd.minor_edits_count(10, 0.33) == 3


if __name__ == "__main__":
    pytest.main([__file__])
