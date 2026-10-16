"""Night Darken on synthetic images under tmp_path: caption matching and the
brightness skip, the darkening / hue-saturation / noise adjustments and the
ΔE ranking, the in-place commit with backups and Night Edits pairs, and the
restore. CPU only, no dataset involved."""

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

NIGHT_CAPTION = "A woman standing on a dark pier, photo taken at night, city lights behind her."
DARK_CAPTION = "This image is dark and dimly lit, a man in a dark room at night."
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
    """RLHF-style concept: chosen/train + rejected/train, six pairs."""
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


def _workspace(tmp_path, concept):
    """An Optimise workspace.json beside the concept registering it plus a Minor Edits dataset."""
    minor = tmp_path / "DPO Minor Edits"
    (minor / "chosen" / "train").mkdir(parents=True)
    (minor / "rejected" / "train").mkdir(parents=True)
    (minor / "manifest.json").write_text(json.dumps({"pairs": [], "retired": []}), encoding="utf-8")
    (tmp_path / "workspace.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"dir": os.path.basename(concept), "type": "normal"},
                    {"dir": "DPO Minor Edits", "type": "minor_edits"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(minor)


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
    assert nd.match_night_phrases(NIGHT_CAPTION) == ["at night", "night", "dark"]
    assert "dark" in nd.match_night_phrases(DARK_CAPTION)
    assert "dimly lit" in nd.match_night_phrases(DARK_CAPTION)
    assert nd.match_night_phrases(DAY_CAPTION) == []
    assert nd.match_night_phrases("A knight in shining armour, nightly news.") == []
    assert nd.match_night_phrases("Shot in LOW-LIGHT conditions") == ["low light"]
    assert nd.match_night_phrases("the room is dark") == ["dark"]
    assert nd.match_night_phrases("") == []


# --------------------------------------------------------------------------- #
# Adjustments
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


def test_hue_and_saturation_neutral_is_identity_and_rotate_scale_chroma():
    pixels = _gradient(color=(1.0, 0.6, 0.3))
    assert np.array_equal(nd.hue_saturation_rgb(pixels, 0.0, 1.0), pixels)
    # Saturation 0 is greyscale with the luma kept.
    grey = nd.hue_saturation_rgb(pixels, 0.0, 0.0)
    assert np.abs(grey[..., 0].astype(int) - grey[..., 1].astype(int)).max() <= 1
    assert np.abs(nd.luma_u8(grey) - nd.luma_u8(pixels)).mean() < 1.0
    # Saturation 1.5 moves chroma away from grey; luma still kept.
    rich = nd.hue_saturation_rgb(pixels, 0.0, 1.5)
    spread = lambda a: (a[..., 0].astype(int) - a[..., 2].astype(int)).mean()  # noqa: E731
    assert spread(rich) > spread(pixels) > spread(grey)
    assert np.abs(nd.luma_u8(rich) - nd.luma_u8(pixels)).mean() < 1.5
    # A 180 degree hue turn flips the chroma vector: red-dominant becomes blue-dominant.
    red = np.full((4, 4, 3), (180, 80, 80), np.uint8)
    flipped = nd.hue_saturation_rgb(red, 180.0, 1.0)[0, 0]
    assert flipped[2] > flipped[0]
    assert abs(nd.luma_u8(flipped[None, None])[0, 0] - nd.luma_u8(red[:1, :1])[0, 0]) < 2.0
    # Hue is cyclic: +90 then +90 equals +180 (up to rounding), -360 is neutral.
    twice = nd.hue_saturation_rgb(nd.hue_saturation_rgb(red, 90.0, 1.0), 90.0, 1.0)[0, 0]
    assert np.abs(twice.astype(int) - flipped.astype(int)).max() <= 2
    assert np.abs(nd.hue_saturation_rgb(red, 360.0, 1.0).astype(int) - red.astype(int)).max() <= 1


def test_noise_neutral_identity_deterministic_and_shadow_weighted():
    pixels = _gradient()
    seed = nd.noise_seed("chosen/train/pair_0001.png")
    assert seed == nd.noise_seed("pair_0001") == nd.noise_seed("F:/x/chosen/train/pair_0001.webp")
    assert seed != nd.noise_seed("pair_0002")
    assert np.array_equal(nd.add_noise_rgb(pixels, 0.0, seed), pixels)
    a = nd.add_noise_rgb(pixels, 0.5, seed)
    b = nd.add_noise_rgb(pixels, 0.5, seed)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, nd.add_noise_rgb(pixels, 0.5, seed + 1))
    assert not np.array_equal(a, pixels)
    # Stronger in the shadows: the dark left strip deviates more than the bright right strip.
    diff = np.abs(a.astype(int) - pixels.astype(int)).astype(float)
    assert diff[:, :16].mean() > diff[:, 64:80].mean() * 1.5
    # More noise, more deviation; alpha untouched.
    assert np.abs(nd.add_noise_rgb(pixels, 1.0, seed).astype(int) - pixels.astype(int)).mean() > diff.mean()
    rgba = np.concatenate([pixels, np.full(pixels.shape[:2] + (1,), 200, np.uint8)], axis=-1)
    assert np.array_equal(nd.add_noise_rgb(rgba, 0.8, seed)[..., 3], rgba[..., 3])


def test_apply_adjustments_order_and_delta_e_ranking():
    pixels = _gradient(color=(1.0, 0.7, 0.5))
    assert np.array_equal(nd.apply_adjustments(pixels, nd.Adjust(), "pair_0001"), pixels)
    assert np.array_equal(nd.apply_adjustments(pixels, None, "pair_0001"), pixels)
    settings = nd.Adjust(strength=0.4, noise=0.3, hue=-20.0, saturation=0.7)
    expected = nd.add_noise_rgb(
        nd.hue_saturation_rgb(nd.darken_rgb(pixels, 0.4), -20.0, 0.7), 0.3, nd.noise_seed("pair_0001")
    )
    assert np.array_equal(nd.apply_adjustments(pixels, settings, "pair_0001"), expected)
    assert np.array_equal(nd.apply_adjustments(pixels, settings.to_dict(), nd.noise_seed("pair_0001")), expected)
    # Every slider contributes to the change metric; neutral is zero.
    assert nd.mean_delta_e(pixels, pixels) == 0.0
    darken_only = nd.mean_delta_e(pixels, nd.apply_adjustments(pixels, nd.Adjust(strength=0.3), "p"))
    with_colour = nd.mean_delta_e(pixels, nd.apply_adjustments(pixels, nd.Adjust(strength=0.3, saturation=0.3), "p"))
    with_noise = nd.mean_delta_e(pixels, nd.apply_adjustments(pixels, nd.Adjust(strength=0.3, noise=0.6), "p"))
    hue_only = nd.mean_delta_e(pixels, nd.apply_adjustments(pixels, nd.Adjust(hue=90.0), "p"))
    assert 0 < darken_only < with_colour
    assert darken_only < with_noise
    assert hue_only > 0
    # from_dict clamps and tolerates junk.
    clamped = nd.Adjust.from_dict({"strength": 3, "noise": -1, "hue": 400, "saturation": "x"})
    assert (clamped.strength, clamped.noise, clamped.hue, clamped.saturation) == (1.0, 0.0, 180.0, 1.0)
    assert nd.Adjust.from_dict({}).is_neutral() and not settings.is_neutral()


def test_save_keeps_format_and_png_text(tmp_path):
    png = _write(tmp_path / "a.png", _gradient(), text={"parameters": "{}", "ot_triage_fix": '{"k": 1}'})
    out = str(tmp_path / "a_out.png")
    result = nd.save_adjusted_like(png, nd.Adjust(strength=0.5, noise=0.2), out)
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.text["parameters"] == "{}"
        assert img.text["ot_triage_fix"] == '{"k": 1}'
        saved = np.asarray(img.convert("RGB"))
    # The file holds exactly what apply_adjustments renders with the stem seed.
    assert np.array_equal(saved, nd.apply_adjustments(_gradient(), nd.Adjust(strength=0.5, noise=0.2), "a"))
    assert result["delta_luma"] > 5 and result["delta_e"] > 5
    webp = _write(tmp_path / "b.webp", _gradient(), fmt="WEBP")
    nd.save_adjusted_like(webp, {"strength": 0.5}, str(tmp_path / "b_out.webp"))
    with Image.open(tmp_path / "b_out.webp") as img:
        assert img.format == "WEBP"
    jpg = _write(tmp_path / "c.jpg", _gradient(), fmt="JPEG")
    nd.save_adjusted_like(jpg, {"strength": 0.5}, str(tmp_path / "c_out.jpg"))
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
    assert result["counts"] == {
        "images": 6,
        "caption_matches": 4,
        "partial_matches": 0,
        "already_dark": 1,
        "unreadable": 0,
        "candidates": 3,
    }
    first = result["candidates"][0]
    assert first["phrases"] == ["at night", "night", "dark"]
    assert first["size"] == [96, 64]
    assert first["stats"]["median"] > nd.DEFAULT_DARK_MEDIAN
    # Tighter thresholds admit the dark one too.
    loose = nd.scan_candidates(concept, thresholds=nd.DarkThresholds(median=5, mean=5))
    assert loose["counts"]["candidates"] == 4


def test_sources_and_destinations_optimise_owns_are_refused(tmp_path):
    concept = _concept(tmp_path)
    minor = _workspace(tmp_path, concept)
    for name in ("DPO Minor Edits", "Off Policy 2"):
        root = tmp_path / "elsewhere" / name
        (root / "chosen").mkdir(parents=True)
        assert nd.source_exclusion_reason(str(root))
        assert nd.destination_exclusion_reason(str(root))
    assert "minor_edits" in nd.source_exclusion_reason(minor)
    assert nd.source_exclusion_reason(concept) is None
    # Any registered dataset is refused as a destination, the normal source too.
    assert "registered" in nd.destination_exclusion_reason(concept)
    assert "registered" in nd.destination_exclusion_reason(minor)
    night = nd.default_night_edits_dir(concept)
    assert night == str(tmp_path / nd.NIGHT_EDITS_DIR_NAME)
    assert nd.destination_exclusion_reason(night) is None
    assert nd.night_edits_preflight(minor, 1)
    assert nd.night_edits_preflight(night, 1) is None


# --------------------------------------------------------------------------- #
# Review, commit, restore through the service
# --------------------------------------------------------------------------- #


def test_commit_and_restore_round_trip(tmp_path):
    concept = _concept(tmp_path)
    minor = _workspace(tmp_path, concept)
    night = nd.default_night_edits_dir(concept)
    service = DPONightDarkenService()
    assert service.open(minor)["ok"] is False
    opened = service.open(concept)
    assert opened["ok"] and opened["counts"]["candidates"] == 0
    assert opened["default_night_edits_dir"] == night

    assert service.scan()["ok"]
    _wait(service)
    state = service.state()
    assert state["counts"] == {"candidates": 3, "accepted": 0, "accepted_uncommitted": 0, "skipped": 0, "undecided": 3}
    rels = [c["rel"] for c in service.candidates()["candidates"]]
    originals = {rel: compute_file_sha256(os.path.join(concept, rel)) for rel in rels}

    # Previews: neutral settings are the original at preview size; adjusted is different and darker.
    light = Image.open(io.BytesIO(service.preview(rels[0], None)))
    dark = Image.open(
        io.BytesIO(service.preview(rels[0], {"strength": 0.8, "noise": 0.3, "hue": -15, "saturation": 0.8}))
    )
    assert light.size == dark.size == (96, 64)
    assert np.asarray(dark).mean() < np.asarray(light).mean()
    assert service.preview("chosen/train/pair_0003.png", None) is None
    assert service.caption(rels[0])["caption"] == NIGHT_CAPTION

    # Decide: the strongest combined edit on rels[0], a mild one on rels[2], one re-decided.
    strong = {"strength": 0.9, "noise": 0.5, "hue": -30, "saturation": 0.6}
    assert service.decide(rels[0], "accepted", strong)["ok"]
    assert service.decide(rels[1], "skipped", None)["ok"]
    assert service.decide(rels[2], "accepted", {"strength": 0.2})["ok"]
    assert service.undo(rels[1])["removed"]
    # rels[1] takes the "Darken more" path twice: none -> Dusk -> Urban, jittered for that image.
    first = service.preset(rels[1], "more", None)
    assert first["ok"] and first["preset"] == "dusk" and first["adjust"]["preset"] == "dusk"
    urban = service.preset(rels[1], "more", first["adjust"])
    assert urban["preset"] == "urban" and urban["adjust"] == nd.jittered_preset("urban", rels[1]).to_dict()
    assert service.preset(rels[1], "moonlit", None)["adjust"]["preset"] == "moonlit"
    assert service.preset(rels[1], "bogus", None)["ok"] is False
    assert service.preset("chosen/train/pair_0003.png", "dusk", None)["ok"] is False
    assert service.decide(rels[1], "accepted", urban["adjust"])["ok"]
    decisions = {c["rel"]: c["decision"] for c in service.candidates()["candidates"]}
    assert decisions[rels[0]]["adjust"] == {
        "strength": 0.9,
        "noise": 0.5,
        "hue": -30.0,
        "saturation": 0.6,
        "preset": "",
    }
    assert decisions[rels[1]]["adjust"] == urban["adjust"]
    assert decisions[rels[2]]["adjust"] == {"strength": 0.2, "noise": 0.0, "hue": 0.0, "saturation": 1.0, "preset": ""}
    assert decisions[rels[0]]["delta_e"] > decisions[rels[1]]["delta_e"] > decisions[rels[2]]["delta_e"]
    assert service.state()["counts"]["accepted_uncommitted"] == 3

    # Optimise-owned destinations and colliding stray files are refused; nothing is touched.
    assert "registered" in service.commit(minor, 0.34)["error"]
    stray = os.path.join(night, "chosen", "train", "pair_0001.png")
    _write(stray, np.zeros((4, 4, 3), np.uint8))
    assert "already exist" in service.commit("", 0.34)["error"]
    os.remove(stray)
    for rel in rels:
        assert compute_file_sha256(os.path.join(concept, rel)) == originals[rel]

    started = service.commit("", 0.34)
    assert started == {"ok": True, "pending": 3, "pairs_planned": 1, "night_edits_dir": night}
    job = _wait(service)
    assert job["phase"] == "done"
    state = service.state()
    assert state["counts"]["accepted_uncommitted"] == 0
    (commit,) = state["commits"]
    assert commit["items"] == 3 and commit["pairs"] == 1 and not commit["restored"]
    assert commit["night_edits_dir"] == night

    # In place: same names, same formats, the exact render of the accepted settings, PNG text kept, backups = originals.
    for rel in rels:
        path = os.path.join(concept, rel)
        assert compute_file_sha256(path) != originals[rel]
        backup = os.path.join(nd.backup_dir(concept), "train", os.path.basename(rel))
        assert compute_file_sha256(backup) == originals[rel]
        with Image.open(path) as now, Image.open(backup) as was:
            assert now.format == was.format and now.size == was.size
            now_px, was_px = np.asarray(now.convert("RGB")), np.asarray(was.convert("RGB"))
            assert now_px.mean() < was_px.mean()
            if now.format == "PNG":
                assert now.text["parameters"] == was.text["parameters"]
                expected = nd.apply_adjustments(was_px, decisions[rel]["adjust"], rel)
                assert np.array_equal(now_px, expected)
    with open(os.path.join(nd.backup_dir(concept), nd.BACKUP_MANIFEST), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["chosen"] for r in rows] == rels
    assert list(rows[0].keys()) == list(nd.BACKUP_CSV_FIELDS)
    assert (rows[0]["strength"], rows[0]["noise"], rows[0]["hue"], rows[0]["saturation"]) == (
        "0.9",
        "0.5",
        "-30.0",
        "0.6",
    )
    assert sum(1 for r in rows if r["night_edits_pair_id"]) == 1
    assert [r["preset"] for r in rows] == ["", "urban", ""]
    assert float(rows[0]["delta_e"]) > float(rows[2]["delta_e"])
    entry = next(e for e in load_manifest(concept)["pairs"] if e["chosen_file"] == "train/pair_0001.png")
    assert entry["history"][-1]["action"] == nd.HISTORY_ACTION and entry["chosen_sha256"]
    assert entry["history"][-1]["adjust"]["noise"] == 0.5

    # The most changed accepted image became pair_0001 of the fresh Night Edits dataset.
    assert os.path.isfile(os.path.join(night, "concepts.json"))
    night_manifest = load_manifest(night)
    assert [e["pair_id"] for e in night_manifest["pairs"]] == [1]
    pair = night_manifest["pairs"][0]
    assert pair["chosen_file"] == "train/pair_0001.png" and pair["rejected_file"] == "train/pair_0001.png"
    assert pair["prompt"] == NIGHT_CAPTION and pair["lineage"]["source_chosen"] == rels[0]
    assert pair["lineage"]["adjust"]["hue"] == -30.0
    chosen_path = os.path.join(night, "chosen", "train", "pair_0001.png")
    rejected_path = os.path.join(night, "rejected", "train", "pair_0001.png")
    assert compute_file_sha256(rejected_path) == originals[rels[0]]
    assert compute_file_sha256(chosen_path) == compute_file_sha256(os.path.join(concept, rels[0]))
    with open(os.path.join(night, "chosen", "train", "pair_0001.txt"), encoding="utf-8") as f:
        assert f.read() == NIGHT_CAPTION
    for image in (chosen_path, rejected_path):
        with Image.open(nd.mask_path_for(image)) as mask, Image.open(image) as img:
            assert mask.mode == "L" and mask.size == img.size
            assert np.asarray(mask).min() == 255
    # Minor Edits was never touched.
    assert load_manifest(minor)["pairs"] == [] and not os.listdir(os.path.join(minor, "chosen", "train"))

    # Decisions are frozen while committed.
    assert service.decide(rels[0], "skipped", None)["ok"] is False
    assert service.undo(rels[0])["ok"] is False

    # Restore: originals back byte for byte, pair gone, manifests as before.
    result = service.restore(commit["id"])
    assert result == {"ok": True, "restored": 3, "pairs_removed": 1, "missing": []}
    for rel in rels:
        assert compute_file_sha256(os.path.join(concept, rel)) == originals[rel]
    assert not os.path.exists(chosen_path) and not os.path.exists(rejected_path)
    assert not os.path.exists(nd.mask_path_for(chosen_path))
    assert not os.path.exists(os.path.join(night, "chosen", "train", "pair_0001.txt"))
    assert load_manifest(night)["pairs"] == []
    entry = next(e for e in load_manifest(concept)["pairs"] if e["chosen_file"] == "train/pair_0001.png")
    assert "history" not in entry and "chosen_sha256" not in entry
    state = service.state()
    assert state["commits"][0]["restored"] and state["counts"]["accepted_uncommitted"] == 3
    assert service.restore(commit["id"])["restored"] == 0

    # A second commit numbers from where the manifest left off (0001 again after the restore),
    # and fraction 0 never touches the destination.
    assert service.commit("", 0.0)["ok"]
    _wait(service)
    assert service.state()["commits"][1]["pairs"] == 0
    assert load_manifest(night)["pairs"] == []


def test_presets_map_onto_the_sliders_and_jitter_deterministically():
    grey = np.full((2, 2, 3), nd.PRESET_REFERENCE_SRGB, np.uint8)
    for preset in nd.NIGHT_PRESETS:
        base = nd.preset_adjust(preset["id"])
        assert base.preset == preset["id"] and base.hue == preset["hue"] and base.saturation == preset["saturation"]
        assert 0 < base.strength < 1 and 0 < base.noise < 1
        # Mid-grey lands on brightness * 118 within ~2 levels, through the real darken curve.
        landed = float(nd.darken_rgb(grey, base.strength)[0, 0, 0])
        assert abs(landed - nd.PRESET_REFERENCE_SRGB * preset["brightness"]) <= 2.0, preset["id"]
        # The noise slider reproduces the suggested sigma at the darkened mid-grey level
        # (per-channel sigma of the generated field, 0..1 scale, up to sampling error).
        level = np.full((256, 256, 3), int(round(nd.PRESET_REFERENCE_SRGB * preset["brightness"])), np.uint8)
        noisy = nd.add_noise_rgb(level, base.noise, 1)
        sigma = float((noisy.astype(np.float32) - level.astype(np.float32)).std() / 255.0)
        assert abs(sigma - preset["noise_sigma"]) < 0.25 * preset["noise_sigma"] + 0.001, preset["id"]
    strengths = [nd.preset_adjust(p["id"]).strength for p in nd.NIGHT_PRESETS]
    assert strengths == sorted(strengths)  # Dusk < Urban < Moonlit
    assert nd.solve_strength_for_brightness(1.0) == 0.0 and nd.solve_strength_for_brightness(0.01) == 1.0
    assert nd.noise_slider_for_sigma(0.0, 0.5) == 0.0
    assert nd.preset_table()[0]["adjust"]["preset"] == "dusk"

    # Darken more: none -> Dusk -> Urban -> Moonlit, staying at Moonlit; by strength when no preset is tagged.
    assert nd.next_preset(None) == "dusk"
    assert nd.next_preset({"preset": "dusk"}) == "urban"
    assert nd.next_preset({"preset": "urban"}) == "moonlit"
    assert nd.next_preset({"preset": "moonlit"}) == "moonlit"
    assert nd.next_preset({"strength": nd.preset_adjust("dusk").strength + 0.01}) == "urban"
    assert nd.next_preset({"strength": 1.0}) == "moonlit"
    assert nd.Adjust.from_dict({"preset": "bogus"}).preset == ""

    # Jitter: deterministic per image + preset, different across images, within +-10% and clamped.
    a = nd.jittered_preset("urban", "chosen/train/pair_0001.png")
    b = nd.jittered_preset("urban", "pair_0001")
    c = nd.jittered_preset("urban", "pair_0002")
    assert a == b and a != c and a.preset == "urban" and a.hue == 0.0
    base = nd.preset_adjust("urban")
    jitter = nd.PRESET_JITTER + 1e-6
    for image in ("pair_0001", "pair_0002", "pair_0077", "pair_1234"):
        j = nd.jittered_preset("urban", image)
        assert abs(j.strength / base.strength - 1) <= jitter
        assert abs(j.saturation / base.saturation - 1) <= jitter
        assert abs(j.noise / base.noise - 1) <= jitter
        assert j != base
    assert nd.jittered_preset("urban", "x", jitter=0.0) == base
    assert nd.jittered_preset("moonlit", "x", jitter=5.0).strength <= 1.0  # clamped to the slider range
    assert nd.jittered_preset("dusk", "x").to_dict()["preset"] == "dusk"
    with pytest.raises(KeyError):
        nd.preset_adjust("bogus")


def test_select_night_edits_takes_the_rounded_share_most_changed_first():
    items = [{"rel": f"r{i}", "delta_e": float(i)} for i in range(6)]
    assert [it["rel"] for it in nd.select_night_edits(items, 0.33)] == ["r5", "r4"]
    assert [it["rel"] for it in nd.select_night_edits(items, 0.01)] == ["r5"]
    assert nd.select_night_edits(items, 0.0) == []
    assert nd.night_edits_count(3, 0.34) == 1 and nd.night_edits_count(10, 0.33) == 3


if __name__ == "__main__":
    pytest.main([__file__])


def test_qualifies_needs_both_night_and_dark():
    found = nd.match_night_phrases
    assert not nd.qualifies(found("A woman with dark brown skin, smiling outdoors."))
    assert not nd.qualifies(found("Photographed at night on a city street."))
    assert not nd.qualifies(found("The room is dark and dimly lit."))
    assert nd.qualifies(found("A dark alley at night, lit by one streetlamp."))
    assert nd.qualifies(found("Dark brown skin, taken late at night in a darkened bar."))
    assert not nd.qualifies(found("Dark brown skin and dark hair, nighttime is not mentioned... daytime."))
