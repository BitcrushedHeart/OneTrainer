import json
import os

from modules.util.text_alignment_dataset_util import (
    REASON_ABSENT,
    REASON_ALREADY_IN_DATASET,
    REASON_DUPLICATE,
    REASON_KEEP,
    REASON_NO_BOXES,
    REASON_NO_METADATA,
    REASON_NO_TEXT_SPAN,
    REASON_UNREADABLE,
    REASON_VERBATIM,
    apply_plan,
    decide,
    load_registry,
    prompt_from_image,
    scan_dataset,
    target_text,
    write_wildcard_file,
)

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(path, prompt=None, color=(120, 90, 60), size=(64, 48)):
    info = PngInfo()
    if prompt is not None:
        info.add_text("sui_image_params", json.dumps({"sui_image_params": {"prompt": prompt, "steps": 8}}))
    Image.new("RGB", size, color).save(path, pnginfo=info)
    return str(path)


def test_prompt_and_target_text_extraction(tmp_path):
    p = _png(tmp_path / "a.png", 'Neon text reading "Open Daily - £4.50"')
    assert prompt_from_image(p) == 'Neon text reading "Open Daily - £4.50"'
    assert prompt_from_image(_png(tmp_path / "b.png")) is None
    assert target_text('Neon text reading "Open Daily - £4.50"') == "Open Daily - £4.50"
    assert target_text('a shop sign that reads "Bakery"') == "Bakery"
    assert target_text("a quiet street at dusk") is None


def test_decide_defaults_keep_only_rendered_but_wrong():
    assert decide(["wrong"], 1) == REASON_KEEP
    assert decide(["wrong"], 0) == REASON_NO_BOXES
    assert decide(["verbatim"], 3) == REASON_VERBATIM
    assert decide(["absent"], 3) == REASON_ABSENT
    assert decide([], 3) == REASON_UNREADABLE
    assert decide(["verbatim", "absent"], 3) == REASON_ABSENT
    assert decide(["verbatim"], 3, keep_verbatim=True) == REASON_KEEP
    assert decide(["absent"], 0, keep_absent=True, min_boxed_words=0) == REASON_KEEP


def test_scan_and_apply_round_trip(tmp_path):
    src = tmp_path / "swarm"
    src.mkdir()
    (src / ".pending").mkdir()
    out = tmp_path / "dataset"
    wrong = _png(src / "wrong.png", 'Sign text reading "Bakery Open"', color=(1, 2, 3))
    _png(src / "wrong_dup.png", 'Sign text reading "Bakery Open"', color=(1, 2, 3))  # identical bytes
    verbatim = _png(src / "verbatim.png", 'Sign text reading "Welcome"', color=(9, 9, 9))
    absent = _png(src / "absent.png", 'Sign text reading "Closed"', color=(5, 5, 5))
    _png(src / "nometa.png")
    _png(src / "nospan.png", "a street at dusk", color=(7, 7, 7))
    unreadable = _png(src / "unreadable.png", 'Sign text reading "Milk"', color=(8, 8, 8))
    _png(src / ".pending" / "hidden.png", 'Sign text reading "Hidden"')

    readings = {wrong: "Bakory Open", verbatim: "Welcome", absent: "nothing here", unreadable: None}

    def read_text(path):
        return readings.get(path)

    boxed_calls = []

    def count_boxed(path, words):
        boxed_calls.append((path, tuple(words)))
        return 2

    seen = []
    plan = scan_dataset(
        str(src),
        str(out),
        read_text=read_text,
        count_boxed=count_boxed,
        progress=lambda d, t, p: seen.append((d, t)),
    )
    by_name = {os.path.basename(c.source_image): c.reason for c in plan.candidates}
    assert by_name == {
        "wrong.png": REASON_KEEP,
        "wrong_dup.png": REASON_DUPLICATE,
        "verbatim.png": REASON_VERBATIM,
        "absent.png": REASON_ABSENT,
        "nometa.png": REASON_NO_METADATA,
        "nospan.png": REASON_NO_TEXT_SPAN,
        "unreadable.png": REASON_UNREADABLE,
    }
    # the detector only ran for the candidate that already looked wrong
    assert boxed_calls == [(wrong, ("Bakery", "Open"))]
    assert plan.counts[REASON_KEEP] == 1 and seen[-1] == (7, 7)
    kept = plan.kept[0]
    assert kept.words == ["Bakery", "Open"] and kept.boxed_words == 2 and kept.ocr_reading == "Bakory Open"
    assert kept.text_score is not None and 0.0 < kept.text_score < 0.5

    result = apply_plan(plan)
    assert result == {"written": 1, "skipped": 0, "cancelled": False}
    files = sorted(os.listdir(out))
    stems = {os.path.splitext(f)[0] for f in files if f not in ("registry.json", "scan_cache.json")}
    assert len(stems) == 1
    stem = next(iter(stems))
    assert {f"{stem}.png", f"{stem}.txt", f"{stem}.json", "registry.json", "scan_cache.json"} == set(files)
    with open(out / f"{stem}.txt", encoding="utf-8") as f:
        assert f.readline().strip() == 'Sign text reading "Bakery Open"'
    with open(out / f"{stem}.json", encoding="utf-8") as f:
        record = json.load(f)
    assert record["text"] == "Bakery Open" and record["words"] == ["Bakery", "Open"]
    assert record["boxed_words"] == 2 and record["source_image"] == wrong and record["sha256"] == kept.sha256
    assert load_registry(str(out)) == {kept.sha256}

    # a rescan of the same folder never re-admits the image
    plan2 = scan_dataset(str(src), str(out), read_text=read_text, count_boxed=count_boxed)
    assert {os.path.basename(c.source_image): c.reason for c in plan2.candidates}[
        "wrong.png"
    ] == REASON_ALREADY_IN_DATASET
    assert apply_plan(plan2) == {"written": 0, "skipped": 0, "cancelled": False}


def test_scan_respects_limit_and_stop(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    for i in range(5):
        _png(src / f"{i}.png", f'Sign text reading "Word{i}"', color=(i, i, i))
    plan = scan_dataset(str(src), str(tmp_path / "o"), read_text=lambda p: "x", count_boxed=lambda p, w: 1, limit=2)
    assert len(plan.candidates) == 2
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    plan = scan_dataset(
        str(src), str(tmp_path / "o"), read_text=lambda p: "x", count_boxed=lambda p, w: 1, should_stop=stop
    )
    assert plan.cancelled and len(plan.candidates) == 2


def test_wildcard_file(tmp_path):
    path = write_wildcard_file(str(tmp_path / "wc"), "text_alignment", ["a", " b ", ""])
    with open(path, encoding="utf-8") as f:
        assert f.read() == "a\nb\n"


def test_scan_cache_skips_ocr_on_rescan_and_records_reader_agreement(tmp_path):
    from modules.util.text_alignment_dataset_util import (
        SCAN_CACHE_FILE,
        load_scan_cache,
        reader_agreement,
        scan_dataset,
    )

    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    img.save(src / "a.png", pnginfo=None)
    # inject a SwarmUI-style prompt through the same hook the scanner uses
    import modules.util.text_alignment_dataset_util as u

    prompt = 'Neon text reading "Open Daily"'
    u_orig = u.prompt_from_image
    u.prompt_from_image = lambda path: prompt
    try:
        calls = {"ocr": 0, "det": 0}

        def read_text(path):
            calls["ocr"] += 1
            return "Opem Daily"

        def count_boxed(path, words):
            calls["det"] += 1
            return 2, ["Opem", "Daily"]

        plan = scan_dataset(str(src), str(out), read_text=read_text, count_boxed=count_boxed)
        assert plan.counts["keep"] == 1
        cand = plan.kept[0]
        assert cand.reward_reading == "Opem Daily"
        assert cand.reader_agreement == 1.0
        assert calls == {"ocr": 1, "det": 1}
        assert (out / SCAN_CACHE_FILE).exists()
        assert len(load_scan_cache(str(out))) == 1

        plan2 = scan_dataset(str(src), str(out), read_text=read_text, count_boxed=count_boxed)
        assert plan2.counts["keep"] == 1 and plan2.kept[0].reader_agreement == 1.0
        assert calls == {"ocr": 1, "det": 1}  # served from the cache
    finally:
        u.prompt_from_image = u_orig

    assert reader_agreement("Open Daily", ["Opem", "Daily"]) < 1.0
    assert reader_agreement(None, ["x"]) is None
    assert reader_agreement("Open", []) is None


def test_reader_disagreement_rejects_only_when_measured_and_enabled():
    from modules.util.text_alignment_dataset_util import REASON_KEEP, REASON_READER_DISAGREE, decide

    wrong = ["wrong"]
    assert decide(wrong, 2, reader_agreement=0.5, min_agreement=0.0) == REASON_KEEP
    assert decide(wrong, 2, reader_agreement=0.5, min_agreement=0.8) == REASON_READER_DISAGREE
    assert decide(wrong, 2, reader_agreement=0.9, min_agreement=0.8) == REASON_KEEP
    assert decide(wrong, 2, reader_agreement=None, min_agreement=0.8) == REASON_KEEP  # not measured
    assert decide(wrong, 0, reader_agreement=0.1, min_agreement=0.8) == "no_boxes"  # boxes checked first
