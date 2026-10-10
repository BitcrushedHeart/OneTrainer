"""Pure Triage & Fix helpers: prompt shape, ledger, candidates, the in-place
replace and the fix pair, all against real files under tmp_path."""

import json
import os

from modules.util import dpo_triage_fix_util as tf
from modules.util.dpo_curation_util import UNCONDITIONAL_PROMPT, load_manifest, save_manifest
from modules.util.dpo_regen_util import parse_segment_string
from modules.util.swarm_generate_util import _params_from_image

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

CAPTION = "A woman on a beach in Mallorca."
NEWSPAPER = tf.INSTRUCTIONS["other"].replace("{target}", "newspaper")


def _png(path, color=(10, 20, 30), size=(64, 48), block: dict | None = None) -> str:
    info = PngInfo()
    if block is not None:
        info.add_text("parameters", json.dumps({"sui_image_params": block}))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    Image.new("RGB", size, color).save(str(path), pnginfo=info)
    return str(path)


def _dataset(tmp_path, count=2, split="", caption=CAPTION):
    root = tmp_path / "ds"
    chosen = root / "chosen" / split if split else root / "chosen"
    rejected = root / "rejected" / split if split else root / "rejected"
    manifest = {"pairs": []}
    for i in range(count):
        stem = f"pair_{i:04d}"
        _png(chosen / f"{stem}.png", (10 + i, 20, 30), block={"prompt": f"{caption} <segment:x>", "model": "Z"})
        _png(rejected / f"{stem}.png", (90 + i, 20, 30))
        if caption:
            (chosen / f"{stem}.txt").write_text(caption, encoding="utf-8")
        manifest["pairs"].append(
            {
                "pair_id": i,
                "prompt": caption or UNCONDITIONAL_PROMPT,
                "aspectratio": "4:3",
                "chosen_file": f"{stem}.png",
                "rejected_file": f"{stem}.png",
                "chosen_source": f"src/{stem}.png",
                "rejected_source": f"src/r{stem}.png",
            }
        )
    save_manifest(str(root), manifest)
    return str(root), manifest


# --------------------------------------------------------------------------- #
# Fixes and prompt
# --------------------------------------------------------------------------- #


def test_segment_prompt_matches_the_verified_klein_shape():
    fixes = tf.normalize_fixes(
        [{"id": "faces", "include_caption": True}, {"id": "other", "target": "newspaper"}, {"id": "hands"}]
    )
    prompt = tf.segment_prompt(CAPTION, fixes)
    assert prompt == (
        f"<segment:yolo-face_yolov8m.pt,1,0.25> {tf.INSTRUCTIONS['faces']} Here is the original prompt: {CAPTION} "
        f"<segment:newspaper,1,0.3> {NEWSPAPER} "
        f"<segment:yolo-hand_yolov8s.pt,1,0.25> {tf.INSTRUCTIONS['hands']}"
    )
    assert [c["id"] for c in tf.default_categories() if c["include_caption"]] == ["text", "background"]
    assert tf.segment_prompt(CAPTION, tf.normalize_fixes([{"id": "faces"}])) == (
        f"<segment:yolo-face_yolov8m.pt,1,0.25> {tf.INSTRUCTIONS['faces']}"
    )
    feet = tf.normalize_fixes([{"id": "feet"}])
    assert tf.segment_prompt("", feet) == f"<segment:yolo-adetailerFootYolov8x_v20.pt,1,0.25> {tf.INSTRUCTIONS['feet']}"
    assert [c["key"] for c in tf.default_categories()] == [*(str(i) for i in range(1, 10)), "0"]


def test_normalize_fixes_requires_a_target_for_other_and_something_to_do():
    with pytest.raises(ValueError):
        tf.normalize_fixes([{"id": "other"}])
    with pytest.raises(ValueError):
        tf.normalize_fixes([])
    with pytest.raises(ValueError):
        tf.normalize_fixes([{"id": "nope"}])


def test_category_overrides_flow_into_fixes():
    cats = tf.normalize_categories(
        [{"id": "faces", "target": "yolo-face.pt", "threshold": 0.4, "instruction": "Repair the face."}]
    )
    fixes = tf.normalize_fixes([{"id": "faces"}], cats)
    assert fixes[0].target == "yolo-face.pt"
    assert fixes[0].threshold == 0.4
    assert tf.segment_prompt("", fixes) == "<segment:yolo-face.pt,1,0.4> Repair the face."
    # An explicit per-fix instruction wins and the caption toggle is honoured.
    fixes = tf.normalize_fixes([{"id": "faces", "instruction": "Soften.", "include_caption": False}], cats)
    assert tf.segment_prompt(CAPTION, fixes) == "<segment:yolo-face.pt,1,0.4> Soften."


def test_build_segment_kwargs_exact_keys():
    fixes = tf.normalize_fixes([{"id": "faces"}])
    klein = tf.klein_settings({"model": "Flux/klein", "steps": "8", "cfgscale": "1"})
    kwargs = tf.build_segment_kwargs(
        CAPTION, fixes, (640, 480), klein, {"segmentmaskblur": 12, "bogus": 1}, 77, "data:x"
    )
    assert kwargs["prompt"] == tf.segment_prompt(CAPTION, fixes)
    assert (kwargs["width"], kwargs["height"], kwargs["model"], kwargs["images"], kwargs["seed"]) == (
        640,
        480,
        "Flux/klein",
        1,
        77,
    )
    assert kwargs["steps"] == 8 and kwargs["cfg_scale"] == 1.0
    assert kwargs["extra_params"] == {
        "initimage": "data:x",
        "initimagecreativity": 0,
        "segmentmaskblur": 12,
        "donotsave": True,
    }
    # Sampler and scheduler only when set.
    klein = tf.klein_settings({"model": "Flux/klein", "sampler": "euler", "scheduler": "simple"})
    kwargs = tf.build_segment_kwargs(CAPTION, fixes, (640, 480), klein, {}, 77, "data:x")
    assert kwargs["extra_params"]["sampler"] == "euler" and kwargs["extra_params"]["scheduler"] == "simple"
    with pytest.raises(ValueError):
        tf.build_segment_kwargs(CAPTION, fixes, (640, 480), tf.klein_settings({}), {}, 77, "data:x")


def test_build_edit_kwargs_exact_keys():
    fix = tf.normalize_fixes([{"id": "other", "target": "newspaper"}])[0]
    klein = tf.klein_settings({"model": "Flux/klein"})
    kwargs = tf.build_edit_kwargs(tf.edit_instruction(fix), (640, 480), klein, 78, "data:y")
    assert kwargs["prompt"] == NEWSPAPER
    assert kwargs["extra_params"] == {"promptimages": "data:y", "donotsave": True}
    assert "initimage" not in kwargs["extra_params"]
    fix = tf.normalize_fixes([{"id": "other", "target": "newspaper", "instruction": "Make the headline legible"}])[0]
    assert tf.edit_instruction(fix) == "Make the headline legible"


def test_build_edit_kwargs_below_full_creativity_starts_from_the_image():
    klein = tf.klein_settings({"model": "Flux/klein"})
    kwargs = tf.build_edit_kwargs("Make it a photo.", (640, 480), klein, 5, "data:p", 0.5)
    assert kwargs["extra_params"] == {
        "promptimages": "data:p",
        "initimage": "data:p",
        "initimagecreativity": 0.5,
        "donotsave": True,
    }
    full = tf.build_edit_kwargs("Make it a photo.", (640, 480), klein, 5, "data:p", 1.0)
    assert "initimage" not in full["extra_params"] and "initimagecreativity" not in full["extra_params"]


def test_several_other_fixes_keep_their_own_strengths():
    fixes = tf.normalize_fixes(
        [
            {"id": "other", "target": "newspaper", "creativity": 0.6, "threshold": 0.2},
            {"id": "other", "target": "watch", "instruction": "Fix the dial.", "creativity": 0.9, "threshold": 0.4},
            {"id": "other", "target": "Newspaper"},
        ]
    )
    assert [(f.id, f.target, f.creativity, f.threshold) for f in fixes] == [
        ("other", "newspaper", 0.6, 0.2),
        ("other", "watch", 0.9, 0.4),
    ]
    kwargs = tf.build_segment_kwargs(
        CAPTION, fixes, (640, 480), tf.klein_settings({"model": "Flux/klein"}), {}, 7, "data:x"
    )
    assert "<segment:newspaper,0.6,0.2>" in kwargs["prompt"] and "<segment:watch,0.9,0.4>" in kwargs["prompt"]
    assert tf.edit_instructions(fixes) == f"{NEWSPAPER} Fix the dial."


def test_build_mask_kwargs_exact_keys():
    klein = tf.klein_settings({"model": "Flux/klein"})
    kwargs = tf.build_mask_kwargs("Fix the medal.", (640, 480), klein, {"maskblur": 4}, 9, "data:i", "data:m", 1.0)
    assert kwargs["prompt"] == "Fix the medal." and kwargs["seed"] == 9 and kwargs["steps"] == 4
    assert kwargs["extra_params"] == {
        "initimage": "data:i",
        "maskimage": "data:m",
        "initimagecreativity": 1.0,
        "initimagerecompositemask": True,
        "maskshrinkgrow": 32,
        "maskblur": 4,
        # Never written to SwarmUI's output tree (the curation tool scans it).
        "donotsave": True,
    }
    fix = tf.normalize_fixes([{"id": "mask"}])[0]
    assert fix.target == "mask" and fix.instruction == tf.INSTRUCTIONS["mask"] and not fix.include_caption
    assert tf.segment_sub_prompt(fix, CAPTION) == tf.INSTRUCTIONS["mask"]


def test_save_mask_resizes_thresholds_and_rejects_empty(tmp_path):
    import base64
    import io

    painted = Image.new("RGBA", (32, 24), (0, 0, 0, 0))
    for x in range(8, 16):
        for y in range(4, 12):
            painted.putpixel((x, y), (255, 255, 255, 200))
    buffer = io.BytesIO()
    painted.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    target = str(tmp_path / ".triage_fix" / "masks" / "pair_0000.png")
    tf.save_mask(uri, target, (64, 48))
    with Image.open(target) as mask:
        assert mask.size == (64, 48) and mask.mode == "L"
        assert mask.getpixel((20, 12)) == 255 and mask.getpixel((0, 0)) == 0
        assert set(mask.getdata()) <= {0, 255}
    empty = io.BytesIO()
    Image.new("L", (32, 24), 0).save(empty, format="PNG")
    with pytest.raises(ValueError):
        tf.save_mask(empty.getvalue(), target, (64, 48))


def test_background_category_inverts_the_subject_mask():
    fixes = tf.normalize_fixes([{"id": "background"}])
    assert fixes[0].target == "yolo-person_yolov8m-seg.pt-1" and fixes[0].threshold == -0.25
    prompt = tf.segment_prompt("A street.", fixes)
    assert prompt.startswith(f"<segment:yolo-person_yolov8m-seg.pt-1,1,-0.25> {tf.INSTRUCTIONS['background']}")
    assert "Here is the original prompt: A street." in prompt
    parsed = parse_segment_string("<segment:yolo-x.pt-1,0.9,-0.3>")
    assert parsed[0].threshold == -0.3
    assert tf.categorize(parsed, tf.default_categories())[0].id == "other"
    assert tf.categorize(parse_segment_string("<segment:background>"), tf.default_categories())[0].id == "background"
    # Magnitude is clamped but the sign survives.
    assert tf.TriageFix.from_dict({"id": "background", "target": "x", "threshold": -3}).threshold == -1.0


def test_detection_ladder_rungs():
    fixes = tf.normalize_fixes([{"id": "faces"}, {"id": "text"}])
    assert tf.ladder_rungs(fixes) == [tf.RUNG_TAGGED, tf.RUNG_LOW, tf.RUNG_CLIPSEG]
    low = [f.at_rung(tf.RUNG_LOW) for f in fixes]
    assert low[0].target.startswith("yolo-") and low[0].threshold == 0.1
    assert low[1].target == "text" and low[1].threshold == 0.3
    clip = [f.at_rung(tf.RUNG_CLIPSEG) for f in fixes]
    assert clip[0].target == "face" and clip[0].threshold == 0.3 and clip[0].instruction == tf.INSTRUCTIONS["faces"]
    # CLIPSeg-only fixes have no ladder; an inverted mask never climbs.
    assert tf.ladder_rungs(tf.normalize_fixes([{"id": "text"}])) == [tf.RUNG_TAGGED]
    background = tf.normalize_fixes([{"id": "background"}])
    assert tf.ladder_rungs(background) == [tf.RUNG_TAGGED]
    # A category without a fallback word stops at the low rung.
    cats = tf.normalize_categories([{"id": "faces", "fallback_target": ""}])
    assert tf.ladder_rungs(tf.normalize_fixes([{"id": "faces"}], cats)) == [tf.RUNG_TAGGED, tf.RUNG_LOW]
    assert tf.default_categories()[0]["fallback_target"] == "face"


def test_parse_and_categorize_detector_string():
    parsed = parse_segment_string("<segment:face,0.9,0.25> <segment:hand> <segment:newspaper,1,0.2>")
    assert [(f.target, f.creativity, f.threshold) for f in parsed] == [
        ("face", 0.9, 0.25),
        ("hand", 0.95, 0.3),
        ("newspaper", 1.0, 0.2),
    ]
    assert parse_segment_string("no tags here <lora:x:1>") == []
    fixes = tf.categorize(parsed, tf.default_categories())
    assert [(f.id, f.target) for f in fixes] == [("faces", "face"), ("hands", "hand"), ("other", "newspaper")]
    assert tf.categorize(parse_segment_string("<segment:feet>"), tf.default_categories())[0].id == "feet"
    people = tf.categorize(parse_segment_string("<segment:person,1,0.3>"), tf.default_categories())
    assert people[0].id == "background_people" and people[0].target == "person"
    people_cat = next(c for c in tf.default_categories() if c["id"] == "background_people")
    assert people_cat["target"] == "|".join(f"yolo-person_yolov8m-seg.pt-{i}" for i in range(2, 9))
    assert fixes[0].creativity == 0.9
    assert fixes[2].instruction == NEWSPAPER


def test_image_data_uri_png_and_jpeg(tmp_path):
    png = _png(tmp_path / "a.png")
    assert tf.image_data_uri(png).startswith("data:image/png;base64,")
    jpg = tmp_path / "b.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(str(jpg))
    assert tf.image_data_uri(str(jpg)).startswith("data:image/jpeg;base64,")


# --------------------------------------------------------------------------- #
# Ledger and items
# --------------------------------------------------------------------------- #


def test_item_seed_is_stable_and_31_bit():
    a = tf.item_seed(4242, "pair_0007")
    assert a == tf.item_seed(4242, "pair_0007")
    assert a != tf.item_seed(4243, "pair_0007")
    assert 0 <= a < 2**31


def test_ledger_roundtrip_and_corrupt_moved_aside(tmp_path):
    root = str(tmp_path)
    assert tf.load_ledger(root) is None
    ledger = tf.new_ledger(7, {"model": "K"})
    ledger["items"]["pair_0001"] = {"status": tf.STATUS_QUEUED}
    tf.save_ledger(root, ledger)
    loaded = tf.load_ledger(root)
    assert loaded["shuffle_seed"] == 7 and loaded["items"]["pair_0001"]["status"] == "queued"
    with open(tf.ledger_path(root), "w", encoding="utf-8") as f:
        f.write("{not json")
    with pytest.raises(RuntimeError):
        tf.load_ledger(root)
    assert not os.path.exists(tf.ledger_path(root))
    assert any(name.startswith("ledger.json.corrupt-") for name in os.listdir(tf.ledger_dir(root)))


def test_enumerate_items_uses_original_prompt_and_finds_train_split(tmp_path):
    root, manifest = _dataset(tmp_path, count=2, split="train")
    (tmp_path / "ds" / "chosen" / "train" / "pair_0001.txt").write_text("Edited caption", encoding="utf-8")
    os.remove(tmp_path / "ds" / "chosen" / "train" / "pair_0000.txt")
    items = tf.enumerate_items(root, manifest)
    assert [i["stem"] for i in items] == ["pair_0000", "pair_0001"]
    # The prompt Klein sees is the original generation prompt, edited .txt or not.
    assert items[0]["caption"] == CAPTION and items[0]["caption_source"] == "manifest"
    assert items[1]["caption"] == CAPTION and items[1]["caption_source"] == "manifest"
    # The fix pair takes the caption as it is on disk right now.
    assert tf.current_caption(root, manifest["pairs"][1], items[1]["caption"]) == "Edited caption"
    assert tf.current_caption(root, manifest["pairs"][0], items[0]["caption"]) == CAPTION
    # No manifest prompt at all: the .txt is the only source left.
    manifest["pairs"][1]["prompt"] = ""
    assert tf.enumerate_items(root, manifest)[1]["caption_source"] == "pair"
    assert items[0]["chosen_path"].endswith(os.path.join("train", "pair_0000.png"))
    assert (items[0]["width"], items[0]["height"]) == (64, 48)


def test_enumerate_items_unconditional(tmp_path):
    root, manifest = _dataset(tmp_path, count=1, caption="")
    items = tf.enumerate_items(root, manifest)
    assert items[0]["unconditional"] and items[0]["caption"] == ""


def test_ledger_counts_and_ready_order():
    ledger = tf.new_ledger(1)
    ledger["items"] = {
        "a": {"status": "ready", "ready_at": 5.0},
        "b": {"status": "ready", "ready_at": 2.0},
        "c": {"status": "tagged"},
        "d": {"status": "fixed"},
    }
    counts = tf.ledger_counts(ledger, ["a", "b", "c", "d", "e"])
    assert counts["undecided"] == 2 and counts["ready"] == 2 and counts["fixed"] == 1 and counts["total"] == 5
    assert tf.ready_stems(ledger) == ["b", "a"]


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


def test_candidate_change_identical_noise_small_patch_and_real_change(tmp_path):
    size = (400, 300)
    original = _png(tmp_path / "o.png", (100, 100, 100), size=size, block={"prompt": "p"})
    same = _png(tmp_path / "s.png", (100, 100, 100), size=size)  # no text chunk
    assert tf.candidate_change(original, same) == (0.0, 0.0)
    # A VAE round trip nudges every pixel a little: no change.
    noisy = _png(tmp_path / "n.png", (103, 100, 98), size=size)
    mad, changed = tf.candidate_change(original, noisy)
    assert mad > 0 and changed == 0.0 and tf.is_no_change(changed)
    # A small repaired patch (a 30x30 hand in 400x300) barely moves the mean
    # but is a real change.
    patched = _png(tmp_path / "c.png", (100, 100, 100), size=size)
    with Image.open(patched) as img:
        img = img.convert("RGB")
        for x in range(200, 230):
            for y in range(150, 180):
                img.putpixel((x, y), (140, 90, 80))
        img.save(patched)
    mad, changed = tf.candidate_change(original, patched)
    assert mad < tf.NO_CHANGE_MAD and not tf.is_no_change(changed)
    assert tf.candidate_diff(original, patched) == mad


def test_write_candidate_keeps_parameters_chunk_and_adds_provenance(tmp_path):
    block = {"prompt": "A prompt <segment:x>", "model": "SoReal/V14", "seed": 3}
    original = _png(tmp_path / "o.png", block=block)
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (1, 2, 3)).save(buffer, format="PNG")
    target = str(tmp_path / ".triage_fix" / "candidates" / "o__v0.png")
    tf.write_candidate(buffer.getvalue(), original, target, {"mode": "segment", "seed": 9})
    assert _params_from_image(target) == block
    with Image.open(target) as img:
        assert json.loads(img.info[tf.OT_CHUNK]) == {"mode": "segment", "seed": 9}
        assert img.getpixel((0, 0)) == (1, 2, 3)


# --------------------------------------------------------------------------- #
# Replace and fix pair
# --------------------------------------------------------------------------- #


def test_replace_chosen_backs_up_updates_manifest_and_undoes(tmp_path):
    root, manifest = _dataset(tmp_path, count=1, split="train")
    entry = manifest["pairs"][0]
    candidate = _png(tmp_path / "cand.png", (200, 0, 0))
    old_sha = entry.get("chosen_sha256")
    record = tf.replace_chosen(root, manifest, entry, candidate, {"fixes": [{"id": "faces"}], "mode": "segment"})
    placed = tf.absolute(root, record["placed"])
    backup = tf.absolute(root, record["backup"])
    assert placed.endswith(os.path.join("chosen", "train", "pair_0000.png"))
    assert os.path.basename(os.path.dirname(backup)) == ".replaced"
    assert os.path.basename(backup).startswith("pair_0000__") and backup.endswith("_chosen.png")
    with Image.open(placed) as img:
        assert img.getpixel((0, 0)) == (200, 0, 0)
    assert (tmp_path / "ds" / "chosen" / "train" / "pair_0000.txt").read_text(encoding="utf-8") == CAPTION
    saved = load_manifest(root)["pairs"][0]
    assert saved["chosen_sha256"] != old_sha and saved["chosen_source"] == "src/pair_0000.png"
    assert saved["history"][-1]["action"] == "triage_fix" and saved["history"][-1]["retired"] == [record["backup"]]

    tf.undo_replace_chosen(root, manifest, entry, record)
    assert not os.path.exists(backup)
    with Image.open(placed) as img:
        assert img.getpixel((0, 0)) == (10, 20, 30)
    restored = load_manifest(root)["pairs"][0]
    assert "history" not in restored and restored.get("chosen_sha256") == old_sha


def test_fix_pair_folder_rules_and_roundtrip(tmp_path):
    root, manifest = _dataset(tmp_path, count=1)
    assert tf.fix_pair_folder_error(os.path.join(root, "inner"), root)
    other = str(tmp_path / "other")
    os.makedirs(other)
    (tmp_path / "other" / "stray.txt").write_text("x", encoding="utf-8")
    assert tf.fix_pair_folder_error(other, root)
    fix_dir = str(tmp_path / "fixes")
    assert tf.fix_pair_folder_error(fix_dir, root) is None

    fix_manifest = tf.load_fix_manifest(fix_dir)
    fixed = _png(tmp_path / "fixed.png", (200, 0, 0))
    original = _png(tmp_path / "orig.png", (10, 20, 30))
    pair_id = tf.write_fix_pair(fix_dir, fix_manifest, fixed, original, CAPTION, "4:3")
    assert pair_id == 0
    assert os.path.isfile(os.path.join(fix_dir, "chosen", "pair_0000.png"))
    assert os.path.isfile(os.path.join(fix_dir, "rejected", "pair_0000.png"))
    with open(os.path.join(fix_dir, "chosen", "pair_0000.txt"), encoding="utf-8") as f:
        assert f.read() == CAPTION
    assert tf.fix_pair_folder_error(fix_dir, root) is None
    tf.undo_fix_pair(fix_dir, fix_manifest, pair_id)
    assert not os.listdir(os.path.join(fix_dir, "chosen"))
    assert load_manifest(fix_dir)["pairs"] == []
    # Unconditional pairs stay unconditional: no sidecar.
    tf.write_fix_pair(fix_dir, fix_manifest, fixed, original, "", "4:3", unconditional=True)
    assert not os.path.exists(os.path.join(fix_dir, "chosen", "pair_0000.txt"))
    assert load_manifest(fix_dir)["pairs"][0]["prompt"] == UNCONDITIONAL_PROMPT


def test_old_default_instructions_migrate_and_typed_ones_stay():
    cats = tf.normalize_categories(
        [
            {"id": "hands", "instruction": tf.LEGACY_INSTRUCTIONS["hands"][0]},
            {"id": "faces", "instruction": tf.LEGACY_INSTRUCTIONS["faces"][1]},
            {"id": "feet", "instruction": "Five toes please."},
        ]
    )
    table = {cat["id"]: cat for cat in cats}
    assert table["hands"]["instruction"] == tf.INSTRUCTIONS["hands"]
    assert table["faces"]["instruction"] == tf.INSTRUCTIONS["faces"]
    assert table["feet"]["instruction"] == "Five toes please."
    assert set(tf.INSTRUCTIONS) == set(tf.LEGACY_INSTRUCTIONS) == {cat.id for cat in tf.TRIAGE_CATEGORIES}
    # The plain wording must not ask for the look that over-processes a soft photo.
    for text in tf.INSTRUCTIONS.values():
        assert not any(word in text.lower() for word in ("sharp", "crisp", "detail", "texture", " hd"))


def test_ledger_migration_touches_pending_fixes_only():
    other_old = tf.LEGACY_INSTRUCTIONS["other"][1].replace("{target}", "watch")
    ledger = tf.new_ledger(1, {"categories": [{"id": "faces", "instruction": tf.LEGACY_INSTRUCTIONS["faces"][0]}]})
    ledger["items"] = {
        "a": {"status": tf.STATUS_QUEUED, "fixes": [{"id": "other", "target": "watch", "instruction": other_old}]},
        "b": {"status": tf.STATUS_QUEUED, "fixes": [{"id": "hands", "instruction": "Give her five fingers."}]},
        "c": {"status": tf.STATUS_FIXED, "fixes": [{"id": "faces", "instruction": tf.LEGACY_INSTRUCTIONS["faces"][0]}]},
    }
    assert tf.migrate_ledger_instructions(ledger) == 1
    assert ledger["items"]["a"]["fixes"][0]["instruction"] == tf.INSTRUCTIONS["other"].replace("{target}", "watch")
    assert ledger["items"]["b"]["fixes"][0]["instruction"] == "Give her five fingers."
    assert ledger["items"]["c"]["fixes"][0]["instruction"] == tf.LEGACY_INSTRUCTIONS["faces"][0]
    assert ledger["settings"]["categories"][0]["instruction"] == tf.INSTRUCTIONS["faces"]


def test_fixer_loras_are_sanitised_and_sent_on_every_request_kind():
    klein = tf.klein_settings(
        {
            "model": "SoReal",
            "steps": 8,
            "loras": [
                {"name": " faces-v2 ", "weight": "0.8"},
                {"name": "FACES-V2", "weight": 1},
                {"name": "", "weight": 1},
                {"name": "style", "weight": 99},
                "junk",
            ],
        }
    )
    assert klein["loras"] == [{"name": "faces-v2", "weight": 0.8}, {"name": "style", "weight": 4.0}]
    fixes = tf.normalize_fixes([{"id": "faces"}])
    segment = tf.build_segment_kwargs(CAPTION, fixes, (64, 48), klein, {}, 1, "data:i")
    edit = tf.build_edit_kwargs("Do it.", (64, 48), klein, 1, "data:i")
    mask = tf.build_mask_kwargs("Do it.", (64, 48), klein, {}, 1, "data:i", "data:m")
    for kwargs in (segment, edit, mask):
        assert kwargs["loras"] == ["faces-v2", "style"] and kwargs["lora_weights"] == [0.8, 4.0]
    plain = tf.build_segment_kwargs(CAPTION, fixes, (64, 48), tf.klein_settings({"model": "k"}), {}, 1, "data:i")
    assert "loras" not in plain and "lora_weights" not in plain


def test_yolo_targets_read_as_a_noun_in_the_instruction():
    assert tf.target_noun("newspaper") == "newspaper"
    assert tf.target_noun("yolo-adetailerFootYolov8x_v20.pt-2") == "foot"
    assert tf.target_noun("yolo-face_yolov8n_v2.pt") == "face"
    assert tf.target_noun("yolo-Eyes.pt") == "eyes"
    assert tf.target_noun("yolo-person_yolov8m-seg.pt") == "person"
    assert tf.target_noun("yolo-yolo26n-seg.pt:dog") == "dog"
    assert tf.target_noun("yolo-yolo26n-seg.pt") == "area"
    fix = tf.normalize_fixes([{"id": "other", "target": "yolo-lips_v1.pt", "threshold": 0.25}])[0]
    assert fix.instruction == tf.INSTRUCTIONS["other"].replace("{target}", "lips")
    assert tf.segment_prompt("", [fix]).startswith("<segment:yolo-lips_v1.pt,1,0.25> Fix the lips so it looks natural.")


def test_color_correction_reaches_segment_and_mask_requests_and_can_be_overridden():
    assert tf.sanitize_color_correction("linear2") == "Linear2"
    assert tf.sanitize_color_correction("bogus") == tf.sanitize_color_correction(None) == "None"
    klein = tf.klein_settings({"model": "Flux/klein", "color_correction": "Linear2"})
    fixes = tf.normalize_fixes([{"id": "faces"}])
    segment = tf.build_segment_kwargs(CAPTION, fixes, (640, 480), klein, {}, 1, "data:x")
    assert segment["extra_params"]["colorcorrectionbehavior"] == "Linear2"
    mask = tf.build_mask_kwargs("p", (640, 480), klein, {}, 1, "data:i", "data:m")
    assert mask["extra_params"]["colorcorrectionbehavior"] == "Linear2"
    # "None" leaves SwarmUI's own default (no key at all).
    off = tf.build_mask_kwargs("p", (640, 480), {**klein, "color_correction": "None"}, {}, 1, "data:i", "data:m")
    assert "colorcorrectionbehavior" not in off["extra_params"]


def test_crop_reference_is_sent_only_below_one():
    assert tf.sanitize_crop_reference("0.5") == 0.5
    assert tf.sanitize_crop_reference(7) == 1.0 and tf.sanitize_crop_reference(-1) == 0.0
    assert tf.sanitize_crop_reference(None) == tf.sanitize_crop_reference("x") == 1.0
    full = tf.klein_settings({"model": "Flux/klein"})
    assert full["crop_reference"] == 1.0
    kwargs = tf.build_mask_kwargs("p", (640, 480), full, {}, 1, "data:i", "data:m")
    assert "cropreferencestrength" not in kwargs["extra_params"]
    half = tf.klein_settings({"model": "Flux/klein", "crop_reference": 0.5})
    fixes = tf.normalize_fixes([{"id": "faces"}])
    segment = tf.build_segment_kwargs(CAPTION, fixes, (640, 480), half, {}, 1, "data:x")
    assert segment["extra_params"]["cropreferencestrength"] == 0.5
    assert tf.sanitize_overrides({"color_correction": "linear", "crop_reference": None}) == {
        "color_correction": "Linear"
    }
