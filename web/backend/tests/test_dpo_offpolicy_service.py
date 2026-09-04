"""End-to-end run of the off-policy DPO builder over a tiny on-disk fixture:
real photos with multi-line sidecars, PNG generations carrying a ``prompt``
text chunk, a configured DPO concept whose captions must be excluded, and a
stubbed realism scorer so the hardest negative is deterministic."""

import json
import os
import shutil
import threading
import time
from types import SimpleNamespace

from web.backend.services.dpo_offpolicy_service import DPOOffPolicyService
from web.backend.services.dpo_service import DPOService

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(path: str, size: tuple[int, int], prompt: str | None = None, color=(120, 90, 60)) -> str:
    img = Image.new("RGB", size, color)
    info = PngInfo()
    if prompt is not None:
        info.add_text("prompt", prompt)
    img.save(path, pnginfo=info)
    return path


def _jpg(path: str, size: tuple[int, int], color=(30, 140, 200)) -> str:
    Image.new("RGB", size, color).save(path, quality=90)
    return path


def _wait_finished(service: DPOOffPolicyService, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.status()
        if status["worker_finished"]:
            return status
        time.sleep(0.05)
    raise AssertionError(f"scan did not finish: {service.status()}")


@pytest.fixture
def scores(monkeypatch):
    """Deterministic *cached* P(AI), keyed on the file's basename — served
    through the scan cache's peek, the only score source this tool has. The
    model itself must never be called."""
    import modules.util.realism_detect_util as rdu
    from modules.util.dpo_curation_util import DpoScanCache

    table: dict[str, float] = {}

    def never(*_a, **_k):
        raise AssertionError("the off-policy tool must not run the realism model")

    monkeypatch.setattr(rdu, "score_images_realism", never)
    monkeypatch.setattr(rdu, "realism_model_version", lambda: "test")
    monkeypatch.setattr(
        DpoScanCache, "peek_realism", lambda self, path, model_version=None: table.get(os.path.basename(path))
    )
    return table


@pytest.fixture
def service(monkeypatch, tmp_path):
    svc = DPOOffPolicyService.__new__(DPOOffPolicyService)
    svc.__init__()
    monkeypatch.setattr(DPOOffPolicyService, "get_instance", classmethod(lambda cls: svc))
    # A configured DPO concept whose chosen captions are already paired.
    concept_dir = tmp_path / "curated"
    (concept_dir / "chosen").mkdir(parents=True)
    (concept_dir / "rejected").mkdir()
    (concept_dir / "chosen" / "pair_0000.txt").write_text("caption already curated\n", encoding="utf-8")
    concept = SimpleNamespace(
        enabled=True, path=str(concept_dir), dpo_chosen_pattern="chosen/{}", dpo_rejected_pattern="rejected/{}"
    )
    monkeypatch.setattr(DPOService, "_resolve_concepts", lambda self: [concept])
    monkeypatch.setattr(DPOService, "realism_device", lambda self: {"ok": True, "device": "cpu"})
    yield svc
    svc.cancel()


@pytest.fixture
def fixture_dirs(tmp_path):
    real = tmp_path / "real"
    gen = tmp_path / "generated"
    out = tmp_path / "out"
    real.mkdir()
    gen.mkdir()

    # photo A: two independent captions; the second has two generations.
    _jpg(str(real / "a.jpg"), (450, 300))
    (real / "a.txt").write_text("photo a line one\nphoto a line two\n", encoding="utf-8")
    _png(str(gen / "a2_soft.png"), (175, 100), "photo a line two")
    _png(str(gen / "a2_hard.png"), (175, 100), "photo a line two")
    # photo B: its only caption is already curated in the DPO concept.
    _jpg(str(real / "b.jpg"), (450, 300))
    (real / "b.txt").write_text("caption already curated\n", encoding="utf-8")
    _png(str(gen / "b.png"), (150, 100), "caption already curated")
    # photo C: generation exists but in the wrong orientation.
    _jpg(str(real / "c.jpg"), (1500, 1000))
    (real / "c.txt").write_text("photo c\n", encoding="utf-8")
    _png(str(gen / "c_portrait.png"), (100, 175), "photo c")
    # photo D: matching generation, single caption.
    _jpg(str(real / "d.jpg"), (200, 200))
    (real / "d.txt").write_text("photo d\n", encoding="utf-8")
    _png(str(gen / "d.png"), (104, 96), "photo d")
    # photo E: no generation at all -> only ever a suggestion.
    _jpg(str(real / "e.jpg"), (900, 1350))
    (real / "e.txt").write_text("photo e needs generating\n", encoding="utf-8")
    return real, gen, out


def test_full_session(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})

    res = service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")
    assert res["ok"], res
    status = _wait_finished(service)
    assert status["error"] is None
    assert status["phase"] == "ready"
    assert status["photos"] == 5
    assert status["generated"] == 5
    assert status["excluded_captions"] == 1  # from the configured concept
    assert status["candidates"] == 2  # a and d; b excluded, c wrong aspect, e no gen

    # First candidate: photo a, hardest negative = lowest P(AI).
    first = service.next()
    cand = first["candidate"]
    assert os.path.basename(cand["real_path"]) == "a.jpg"
    assert os.path.basename(cand["rejected_path"]) == "a2_hard.png"
    assert cand["rejected_score"] == 0.1
    assert cand["real_score"] is None  # the fake scorer has no entry for a.jpg
    assert cand["scored"] is True
    assert service.current()["candidate"]["real_path"] == cand["real_path"]
    assert cand["caption"] == "photo a line two"
    assert cand["caption_lines"] == ["photo a line one", "photo a line two"]
    assert cand["generated_count"] == 2
    assert cand["aspect_bucket"] == "3:2"
    assert 0.15 < cand["aspect_mismatch"] < 0.18

    confirmed = service.confirm()
    assert confirmed == {"ok": True, "total_pairs": 1}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["pairs"][0]
    assert entry["prompt"] == "photo a line two"
    assert entry["aspectratio"] == "3:2"
    assert entry["chosen_file"] == "pair_0000.jpg"
    assert entry["rejected_file"] == "pair_0000.png"
    assert (out / "chosen" / "pair_0000.txt").read_text(encoding="utf-8") == "photo a line two"
    assert (out / "chosen" / "pair_0000.jpg").is_file()
    assert (out / "rejected" / "pair_0000.png").is_file()
    # The source photo is untouched by a confirm.
    assert (real / "a.jpg").is_file() and (real / "a.txt").is_file()

    # Second candidate: photo d — caption is wrong: the line goes, the photo stays.
    second = service.next()["candidate"]
    assert os.path.basename(second["real_path"]) == "d.jpg"
    fixed = service.remove_caption()
    assert fixed == {"ok": True, "removed_lines": 1, "removed_file": True, "remaining_lines": 0}
    assert (real / "d.jpg").is_file()
    assert not (real / "d.txt").exists()  # it was the only line
    assert (real / ".offpolicy_trash" / "d.txt").is_file()  # backup until the session ends
    assert (gen / "d.png").is_file()  # generation is never touched

    assert service.next() == {"ok": True, "done": True, "total_pairs": 1}
    final = service.status()
    assert (final["confirmed"], final["skipped"], final["captions_removed"]) == (1, 0, 1)

    # Suggestion: e has no generation and c's only generation is the wrong
    # orientation, so both are worth generating for; a is paired, b excluded.
    seen = set()
    for _ in range(30):
        suggestion = service.suggest()["suggestion"]
        assert suggestion["remaining"] == 2
        seen.add(os.path.basename(suggestion["real_path"]))
        if suggestion["real_path"].endswith("e.jpg"):
            assert suggestion["caption"] == "photo e needs generating"
            assert suggestion["aspect_bucket"] == "2:3"
            assert suggestion["generate_aspect"] == "2:3"  # a ratio the generator can emit
            assert suggestion["within_tolerance"] is True
            assert suggestion["real_size"] == [900, 1350]
            w, h = suggestion["generate_size"]
            assert h > w and w % 16 == 0 and h % 16 == 0
    assert seen == {"c.jpg", "e.jpg"}


def test_skip_is_per_session_and_rerun_excludes_confirmed_captions(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "a.jpg"
    assert service.skip() == {"ok": True}
    assert not (out / "offpolicy_state.json").exists()  # nothing persisted
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "d.jpg"
    assert service.confirm()["ok"]
    assert service.next()["done"]
    service.cancel()

    # Second run over the same output dir: a comes back (skips are per session),
    # d is paired and its caption excluded.
    res = service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")
    assert res["ok"] and res["existing_pairs"] == 1
    status = _wait_finished(service)
    assert status["candidates"] == 1
    assert status["excluded_captions"] == 2  # concept caption + "photo d" from the manifest
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "a.jpg"


def test_no_cached_scores_means_random_negative_flagged_unscored(service, monkeypatch, fixture_dirs):
    import random

    import modules.util.realism_detect_util as rdu

    real, gen, out = fixture_dirs
    monkeypatch.setattr(rdu, "score_images_realism", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model")))
    monkeypatch.setattr(rdu, "realism_model_version", lambda: "test")
    service._rng = random.Random(0)

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    status = _wait_finished(service)
    assert status["unscored"] == 2
    cand = service.next()["candidate"]
    assert cand["scored"] is False and cand["rejected_score"] is None
    assert os.path.basename(cand["rejected_path"]) in {"a2_soft.png", "a2_hard.png"}


def test_missing_model_just_means_everything_is_unscored(service, monkeypatch, fixture_dirs):
    import modules.util.realism_detect_util as rdu

    real, gen, out = fixture_dirs
    monkeypatch.setattr(rdu, "realism_model_version", lambda: None)  # no weights installed
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    status = _wait_finished(service)
    assert status["phase"] == "ready" and status["error"] is None
    assert status["unscored"] == status["candidates"] == 2
    assert service.next()["candidate"]["scored"] is False


def test_start_validates_folders(service, tmp_path):
    res = service.start(str(tmp_path / "nope"), str(tmp_path), str(tmp_path / "out"))
    assert not res["ok"]
    res = service.start(str(tmp_path), str(tmp_path), str(tmp_path / "out"), aspect_tolerance=2.0)
    assert not res["ok"]
    assert service.next() == {"ok": False, "error": "No active session"}
    assert service.confirm() == {"ok": False, "error": "No candidate to confirm"}


def test_start_refuses_nested_folders(service, tmp_path):
    real = tmp_path / "real"
    gen = tmp_path / "gen"
    real.mkdir()
    gen.mkdir()
    assert not service.start(str(gen), str(real), str(gen / "out"))["ok"]
    assert not service.start(str(gen), str(real), str(real / "out"))["ok"]
    (real / "gen").mkdir()
    assert not service.start(str(real / "gen"), str(real), str(tmp_path / "out"))["ok"]
    assert not service.start(str(gen), str(gen), str(tmp_path / "out"))["ok"]
    assert service.status()["active"] is False


def test_start_refuses_output_dir_of_an_active_curation_session(service, monkeypatch, tmp_path):
    real = tmp_path / "real"
    gen = tmp_path / "gen"
    out = tmp_path / "out"
    real.mkdir()
    gen.mkdir()
    curation = DPOService.__new__(DPOService)
    curation._session_active = True
    curation._output_dir = str(out)
    monkeypatch.setattr(DPOService, "get_instance", classmethod(lambda cls: curation))
    res = service.start(str(gen), str(real), str(out))
    assert not res["ok"] and "curation" in res["error"]
    curation._session_active = False
    assert service.start(str(gen), str(real), str(out))["ok"]


def test_a_caption_paired_in_this_session_is_not_paired_again(service, scores, tmp_path):
    """Two photos sharing a caption line: once the first pair is exported the
    caption is excluded, so the second photo must not be presented on it."""
    real = tmp_path / "real"
    gen = tmp_path / "gen"
    real.mkdir()
    gen.mkdir()
    for name in ("p1", "p2"):
        _jpg(str(real / f"{name}.jpg"), (150, 100), color=(10, 20, 30) if name == "p1" else (40, 50, 60))
        (real / f"{name}.txt").write_text("shared caption\n", encoding="utf-8")
    _png(str(gen / "g1.png"), (150, 100), "shared caption", color=(1, 2, 3))
    _png(str(gen / "g2.png"), (150, 100), "shared caption", color=(4, 5, 6))
    scores.update({"g1.png": 0.2, "g2.png": 0.3})

    assert service.start(str(gen), str(real), str(tmp_path / "out"), order="path")["ok"]
    assert _wait_finished(service)["candidates"] == 2
    first = service.next()["candidate"]
    assert os.path.basename(first["real_path"]) == "p1.jpg"
    assert service.confirm()["ok"]
    assert service.next()["done"]


def test_byte_identical_real_photo_is_not_paired_twice(service, scores, tmp_path):
    """The same photo under another name (and another caption) is a duplicate
    chosen sample: the fingerprint check drops it at present time."""
    real = tmp_path / "real"
    gen = tmp_path / "gen"
    real.mkdir()
    gen.mkdir()
    _jpg(str(real / "orig.jpg"), (150, 100))
    shutil.copyfile(str(real / "orig.jpg"), str(real / "zz_copy.jpg"))
    (real / "orig.txt").write_text("caption one\n", encoding="utf-8")
    (real / "zz_copy.txt").write_text("caption two\n", encoding="utf-8")
    _png(str(gen / "g1.png"), (150, 100), "caption one", color=(1, 2, 3))
    _png(str(gen / "g2.png"), (150, 100), "caption two", color=(4, 5, 6))
    scores.update({"g1.png": 0.2, "g2.png": 0.3})

    assert service.start(str(gen), str(real), str(tmp_path / "out"), order="path")["ok"]
    assert _wait_finished(service)["candidates"] == 2
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "orig.jpg"
    assert service.confirm()["ok"]
    assert service.next()["done"]
    assert service.status()["total_pairs"] == 1


def test_remove_caption_keeps_the_other_lines_and_the_photo(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    (real / "a.txt").write_text("photo a line one\nphoto a line two\n", encoding="utf-8")
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    cand = service.next()["candidate"]
    assert os.path.basename(cand["real_path"]) == "a.jpg" and cand["caption"] == "photo a line two"
    res = service.remove_caption()
    assert res == {"ok": True, "removed_lines": 1, "removed_file": False, "remaining_lines": 1}
    assert (real / "a.jpg").is_file()
    assert (real / "a.txt").read_text(encoding="utf-8") == "photo a line one\n"
    # The photo is not re-queued this session; the lookup sees the edit.
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "d.jpg"
    (a,) = service.lookup("a.jpg")["matches"]
    assert a["caption_lines"] == ["photo a line one"]


def test_restart_while_scoring_does_not_leak_the_old_worker(service, monkeypatch, fixture_dirs):
    """cancel() then start() while the previous worker is blocked mid-pick
    (here: inside the score cache): the stale worker must not append its
    candidates to the new session or mark it finished."""
    import modules.util.realism_detect_util as rdu

    real, gen, out = fixture_dirs
    reached = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    from modules.util.dpo_curation_util import DpoScanCache

    def blocking_peek(self, path, model_version=None):
        calls["n"] += 1
        if calls["n"] == 1:
            reached.set()
            assert release.wait(10)
        return 0.5

    monkeypatch.setattr(DpoScanCache, "peek_realism", blocking_peek)
    monkeypatch.setattr(rdu, "realism_model_version", lambda: "test")

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    assert reached.wait(10)
    service.cancel()
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    status = _wait_finished(service)
    assert status["candidates"] == 2 and status["ready"] == 2
    release.set()
    time.sleep(0.3)
    status = service.status()
    assert status["ready"] == 2, status  # the stale worker's chunk did not land
    assert status["phase"] == "ready"
    seen = {os.path.basename(service.next()["candidate"]["real_path"]) for _ in range(2)}
    assert seen == {"a.jpg", "d.jpg"}
    assert service.next()["done"]


def test_photo_smaller_than_its_generation_is_excluded_and_counted(service, scores, tmp_path):
    real = tmp_path / "real2"
    gen = tmp_path / "generated2"
    out = tmp_path / "out2"
    real.mkdir()
    gen.mkdir()
    _jpg(str(real / "small.jpg"), (150, 100))
    (real / "small.txt").write_text("small photo\n", encoding="utf-8")
    _png(str(gen / "small.png"), (300, 200), "small photo")  # generation is larger
    _jpg(str(real / "big.jpg"), (600, 400))
    (real / "big.txt").write_text("big photo\n", encoding="utf-8")
    _png(str(gen / "big.png"), (300, 200), "big photo")
    scores.update({"small.png": 0.5, "big.png": 0.5})

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    status = _wait_finished(service)
    assert status["candidates"] == 1
    assert status["too_small"] == 1
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "big.jpg"


def test_images_already_in_another_dpo_concept_are_never_paired(service, scores, monkeypatch, tmp_path):
    """A generation that the curation tool already committed as a rejected
    sample (recorded in that concept's manifest by path and by byte
    fingerprint) must not be picked here, even via a renamed copy."""
    import json
    import shutil
    from types import SimpleNamespace

    real = tmp_path / "real3"
    gen = tmp_path / "generated3"
    out = tmp_path / "out3"
    curated = tmp_path / "curated3"
    real.mkdir()
    gen.mkdir()
    (curated / "chosen").mkdir(parents=True)
    (curated / "rejected").mkdir()
    _jpg(str(real / "p.jpg"), (450, 300))
    (real / "p.txt").write_text("caption p\n", encoding="utf-8")
    used = _png(str(gen / "used.png"), (175, 100), "caption p", color=(1, 2, 3))
    _png(str(gen / "free.png"), (175, 100), "caption p", color=(4, 5, 6))
    # A renamed byte-identical copy of the used generation.
    shutil.copyfile(used, str(gen / "used_copy.png"))
    # The curated concept committed used.png as a rejected sample.
    shutil.copyfile(used, str(curated / "rejected" / "pair_0000.png"))
    (curated / "chosen" / "pair_0000.txt").write_text("some other caption\n", encoding="utf-8")
    manifest = {
        "pairs": [
            {
                "pair_id": 0,
                "prompt": "some other caption",
                "aspectratio": "7:4",
                "chosen_file": "pair_0000.jpg",
                "rejected_file": "pair_0000.png",
                "chosen_source": None,
                "rejected_source": os.path.normcase(os.path.abspath(used)),
            }
        ]
    }
    (curated / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    concept = SimpleNamespace(
        enabled=True, path=str(curated), dpo_chosen_pattern="chosen/{}", dpo_rejected_pattern="rejected/{}"
    )
    monkeypatch.setattr(DPOService, "_resolve_concepts", lambda self: [concept])
    scores.update({"used.png": 0.0, "used_copy.png": 0.0, "free.png": 0.9})  # the used ones would be "hardest"

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    cand = service.next()["candidate"]
    assert os.path.basename(cand["rejected_path"]) == "free.png"
    assert cand["generated_count"] == 1


def test_lookup_by_filename_reports_generation_size(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.lookup("e.jpg") == {"ok": False, "error": "No active session"}

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)

    res = service.lookup("e")
    assert res["ok"] and res["generations_scanned"] is True
    (m,) = res["matches"]
    assert os.path.basename(m["real_path"]) == "e.jpg"
    assert m["real_size"] == [900, 1350]
    assert m["aspect_bucket"] == "2:3"
    w, h = m["generate_size"]
    assert h > w and w % 16 == 0 and h % 16 == 0
    assert m["lines_needing_generation"] == ["photo e needs generating"]
    assert m["has_usable_generation"] is False

    # Photo a has a usable generation; photo b's caption is paired elsewhere.
    (a,) = service.lookup("A.JPG")["matches"]
    assert a["has_usable_generation"] is True
    (b,) = service.lookup("b.jpg")["matches"]
    assert b["lines_needing_generation"] == []

    assert service.lookup("nope.jpg") == {"ok": True, "matches": [], "generations_scanned": True}
    assert service.lookup("  ")["ok"] is False
    # Traversal-shaped input is just a non-matching string.
    assert service.lookup("../../" + os.path.basename(str(real)) + "/e.jpg")["matches"] == []


def test_suggest_by_bucket_and_bucket_listing(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.buckets() == {"ok": False, "error": "No active session"}
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)

    listing = service.buckets()
    # c (1500x1000, 3:2, wrong-orientation generation only) and e (900x1350, 2:3).
    assert {k: v["photos"] for k, v in listing["buckets"].items()} == {"3:2": 1, "2:3": 1}
    assert listing["remaining"] == 2

    res = service.suggest(aspect_bucket="2:3", count=10)
    assert res["ok"] and res["aspect_bucket"] == "2:3"
    assert [os.path.basename(e["real_path"]) for e in res["suggestions"]] == ["e.jpg"]
    assert res["suggestion"]["caption"] == "photo e needs generating"
    assert res["suggestion"]["remaining"] == 2
    assert res["buckets"] == listing["buckets"]

    assert service.suggest(aspect_bucket="1:1", count=5)["suggestions"] == []
    assert service.suggest(aspect_bucket="1:1", count=5)["suggestion"] is None
    both = service.suggest(count=10)
    assert sorted(os.path.basename(e["real_path"]) for e in both["suggestions"]) == ["c.jpg", "e.jpg"]


def test_suggest_and_buckets_accept_a_keyword_filter(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)

    # Only e's caption contains "generating"; c's is "photo c".
    listing = service.buckets(keywords="generating")
    assert {k: v["photos"] for k, v in listing["buckets"].items()} == {"2:3": 1}
    assert listing["remaining"] == 1
    res = service.suggest(count=10, keywords="Photo, GENERATING")
    assert [os.path.basename(e["real_path"]) for e in res["suggestions"]] == ["e.jpg"]
    assert res["keywords"] == "Photo, GENERATING"
    assert service.suggest(count=10, keywords="no such words")["suggestions"] == []
    assert service.suggest(aspect_bucket="3:2", count=10, keywords="generating")["suggestions"] == []
    assert len(service.suggest(count=10, keywords="")["suggestions"]) == 2


def test_second_run_indexes_real_photos_without_opening_any_file(service, scores, fixture_dirs, monkeypatch):
    import modules.util.dpo_offpolicy_util as util

    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    first = _wait_finished(service)
    assert first["photos"] == 5 and first["index_hits"] == 0
    service.cancel()

    # Warm run: neither the sidecars nor the image headers may be read.
    def explode(*_a, **_k):
        raise AssertionError("warm index must not touch files")

    import web.backend.services.dpo_offpolicy_service as svc_module

    real_read = svc_module.read_caption_lines
    real_image_size = util.image_size
    monkeypatch.setattr(svc_module, "read_caption_lines", explode)
    monkeypatch.setattr(util, "image_size", explode)
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    second = _wait_finished(service)
    assert second["error"] is None
    assert second["photos"] == 5 and second["index_hits"] == 5
    assert second["candidates"] == first["candidates"]
    service.cancel()

    # A changed sidecar is re-read (and only that one). Re-patch rather than
    # undo(): undo would also strip the fixtures' patches (concept resolution,
    # the fake scorer) and turn this into a run against the real config.
    monkeypatch.setattr(util, "image_size", real_image_size)
    calls: list[str] = []

    def counting(path):
        calls.append(os.path.basename(path))
        return real_read(path)

    monkeypatch.setattr(svc_module, "read_caption_lines", counting)
    txt = real / "a.txt"
    txt.write_text("photo a line one\nphoto a line two\nphoto a line three\n", encoding="utf-8")
    os.utime(str(txt), (time.time() + 5, time.time() + 5))
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    third = _wait_finished(service)
    assert third["photos"] == 5 and third["index_hits"] == 4
    assert calls == ["a.txt"]
    (a,) = service.lookup("a.jpg")["matches"]
    assert a["caption_lines"] == ["photo a line one", "photo a line two", "photo a line three"]


def test_generated_scan_reuses_the_curation_tools_cache(service, scores, fixture_dirs, monkeypatch, tmp_path):
    """The curation tool's cache for the same generated tree is imported at
    start, so metadata extraction never runs for files it already covers."""
    from types import SimpleNamespace

    import modules.util.image_metadata_util as meta_module
    from modules.util.dpo_curation_util import DpoScanCache, normalize_prompt_for_grouping, resolve_aspect_ratio
    from modules.util.dpo_offpolicy_util import image_size
    from modules.util.image_metadata_util import extract_metadata

    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})

    # A curation concept whose output folder holds a warm cache of the tree.
    curated = tmp_path / "curated_cache"
    (curated / "chosen").mkdir(parents=True)
    (curated / "rejected").mkdir()
    warm = DpoScanCache(str(curated / ".dpo_hash_cache.json"))
    for name in os.listdir(gen):
        path = str(gen / name)
        warm.get_group_key(
            path,
            lambda p: (
                normalize_prompt_for_grouping(extract_metadata(p).get("prompt", "")),
                resolve_aspect_ratio("", p),
            ),
        )
        warm.get_image_size(path, image_size)
    warm.save()
    concept = SimpleNamespace(
        enabled=True, path=str(curated), dpo_chosen_pattern="chosen/{}", dpo_rejected_pattern="rejected/{}"
    )
    monkeypatch.setattr(DPOService, "_resolve_concepts", lambda self: [concept])

    def explode(*_a, **_k):
        raise AssertionError("metadata must come from the imported cache")

    monkeypatch.setattr(meta_module, "extract_metadata", explode)
    res = service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")
    assert res["ok"] and res["cache_imported"] == len(os.listdir(gen))
    status = _wait_finished(service)
    assert status["error"] is None
    assert status["generated"] == 5
    assert status["cache_imported"] == 5
    # This concept fixture has no curated captions, so b joins a and d.
    assert status["candidates"] == 3


def test_undo_reverses_confirm_skip_and_delete(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    assert service.undo() == {"ok": False, "error": "Nothing to undo"}
    assert service.status()["can_undo"] is False

    # confirm -> undo: pair files and manifest entry gone, caption free again.
    first = service.next()["candidate"]
    assert service.confirm()["ok"]
    assert (out / "chosen" / "pair_0000.jpg").is_file()
    assert service.status()["undo_action"] == "confirm"
    res = service.undo()
    assert res["ok"] and res["undone"] == "confirm"
    assert res["candidate"]["real_path"] == first["real_path"]
    assert not (out / "chosen" / "pair_0000.jpg").exists()
    assert not (out / "chosen" / "pair_0000.txt").exists()
    assert not (out / "rejected" / "pair_0000.png").exists()
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["pairs"] == []
    assert service.status()["confirmed"] == 0 and service.status()["total_pairs"] == 0
    # The same candidate is back on screen and can be confirmed again cleanly.
    assert service.confirm()["ok"]
    assert (out / "chosen" / "pair_0000.jpg").is_file()

    # skip -> undo
    second = service.next()["candidate"]
    assert service.skip()["ok"]
    res = service.undo()
    assert res["ok"] and res["undone"] == "skip" and res["candidate"]["real_path"] == second["real_path"]
    assert service.status()["skipped"] == 0

    # remove caption -> undo: the sidecar comes back exactly.
    original = (real / "d.txt").read_bytes()
    assert service.remove_caption()["ok"]
    assert not (real / "d.txt").exists()
    res = service.undo()
    assert res["ok"] and res["undone"] == "caption"
    assert (real / "d.txt").read_bytes() == original
    assert not (real / ".offpolicy_trash" / "d.txt").exists()  # undo cleans up its own backup
    assert res["candidate"]["caption_lines"] == ["photo d"]
    assert service.status()["captions_removed"] == 0 and service.status()["photos"] == 5

    # Undo with a different candidate on screen pushes that one back in the queue.
    assert service.skip()["ok"]  # d skipped
    assert service.next()["done"]
    assert service.undo()["ok"]  # d back on screen
    assert service.status()["has_candidate"] is True

    # Remove for real, then finishing the session purges the backup.
    assert service.remove_caption()["ok"]
    assert (real / ".offpolicy_trash" / "d.txt").is_file()
    assert service.cancel()["purged"] == 1
    assert not (real / ".offpolicy_trash").exists()
    assert (real / "d.jpg").is_file() and not (real / "d.txt").exists()


def test_candidates_are_shuffled_by_default(service, scores, tmp_path):
    real = tmp_path / "real_s"
    gen = tmp_path / "gen_s"
    out = tmp_path / "out_s"
    real.mkdir()
    gen.mkdir()
    names = [f"p{i:02d}" for i in range(12)]
    for name in names:
        _jpg(str(real / f"{name}.jpg"), (300, 200))
        (real / f"{name}.txt").write_text(f"caption {name}\n", encoding="utf-8")
        _png(str(gen / f"{name}.png"), (150, 100), f"caption {name}")

    def order_with(seed, order="random"):
        service._rng = random.Random(seed)
        assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order=order)["ok"]
        _wait_finished(service)
        seen = []
        while True:
            res = service.next()
            if res.get("done"):
                break
            seen.append(os.path.basename(res["candidate"]["real_path"]))
            assert service.skip()["ok"]
        service.cancel()
        return seen

    import random

    a = order_with(1)
    b = order_with(2)
    assert sorted(a) == sorted(b) == [f"{n}.jpg" for n in names]
    assert a != [f"{n}.jpg" for n in names] or b != [f"{n}.jpg" for n in names]
    assert order_with(3, order="path") == [f"{n}.jpg" for n in names]


def test_reroll_shows_another_generation_and_confirm_uses_it(service, scores, fixture_dirs):
    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    assert service.reroll() == {"ok": False, "error": "No candidate on screen"}
    cand = service.next()["candidate"]
    assert os.path.basename(cand["rejected_path"]) == "a2_hard.png" and cand["shown_count"] == 1
    res = service.reroll()
    assert res["ok"]
    assert os.path.basename(res["candidate"]["rejected_path"]) == "a2_soft.png"
    assert res["candidate"]["shown_count"] == 2 and res["candidate"]["rejected_score"] == 0.9
    assert service.confirm()["ok"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert os.path.basename(manifest["pairs"][0]["rejected_source"]) == "a2_soft.png"
    # d has a single generation: re-roll refuses.
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "d.jpg"
    assert service.reroll()["ok"] is False


def test_allowed_upscale_is_a_start_parameter(service, scores, tmp_path):
    real = tmp_path / "real_u"
    gen = tmp_path / "gen_u"
    out = tmp_path / "out_u"
    real.mkdir()
    gen.mkdir()
    _jpg(str(real / "near.jpg"), (280, 187))  # ~7% under a 300x200 generation
    (real / "near.txt").write_text("near photo\n", encoding="utf-8")
    _png(str(gen / "near.png"), (300, 200), "near photo")
    _jpg(str(real / "far.jpg"), (200, 133))  # 50% under
    (real / "far.txt").write_text("far photo\n", encoding="utf-8")
    _png(str(gen / "far.png"), (300, 200), "far photo")
    scores.update({"near.png": 0.5, "far.png": 0.5})

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    status = _wait_finished(service)
    assert status["candidates"] == 1 and status["too_small"] == 1  # default 10%: near passes, far fails
    service.cancel()

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, max_upscale=1.0, order="path")["ok"]
    assert _wait_finished(service)["candidates"] == 0
    service.cancel()

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, max_upscale=2.0, order="path")["ok"]
    assert _wait_finished(service)["candidates"] == 2
    service.cancel()

    assert not service.start(str(gen), str(real), str(out), max_upscale=0.5)["ok"]


def test_bucket_listing_only_offers_generatable_ratios(service, scores, tmp_path):
    from modules.util.dpo_offpolicy_util import generatable_aspect_labels

    real = tmp_path / "real_g"
    gen = tmp_path / "gen_g"
    out = tmp_path / "out_g"
    real.mkdir()
    gen.mkdir()
    # 7:4 and 4:5 are trainer buckets the generator cannot emit.
    for name, size in (("wide", (2100, 1200)), ("tall", (1200, 1500))):
        _jpg(str(real / f"{name}.jpg"), size)
        (real / f"{name}.txt").write_text(f"{name} caption\n", encoding="utf-8")

    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    listing = service.buckets()
    assert set(listing["buckets"]) <= set(generatable_aspect_labels())
    assert {k: v["photos"] for k, v in listing["buckets"].items()} == {"16:9": 1, "3:4": 1}
    res = service.suggest(aspect_bucket="16:9", count=5)
    assert [os.path.basename(e["real_path"]) for e in res["suggestions"]] == ["wide.jpg"]
    (entry,) = res["suggestions"]
    assert entry["generate_size"] == list(listing["buckets"]["16:9"]["generate_size"])


def test_permanent_skip_survives_restarts_and_undo_lifts_it(service, scores, fixture_dirs):
    from modules.util.dpo_offpolicy_util import PERMANENT_SKIP_FILENAME

    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    assert service.skip_forever() == {"ok": False, "error": "No candidate to skip"}

    assert os.path.basename(service.next()["candidate"]["real_path"]) == "a.jpg"
    res = service.skip_forever()
    assert res["ok"] and res["permanently_skipped"] == 1
    assert json.loads((out / PERMANENT_SKIP_FILENAME).read_text(encoding="utf-8"))["skipped"]
    status = service.status()
    assert status["skipped_forever"] == 1 and status["permanently_skipped"] == 1 and status["skipped"] == 0

    # Undo takes it back off the saved list and puts the candidate back.
    undone = service.undo()
    assert undone["ok"] and undone["undone"] == "skip_forever"
    assert os.path.basename(undone["candidate"]["real_path"]) == "a.jpg"
    assert json.loads((out / PERMANENT_SKIP_FILENAME).read_text(encoding="utf-8"))["skipped"] == []
    assert service.status()["skipped_forever"] == 0

    # Do it for real, plus a session-only skip on the other candidate.
    assert service.skip_forever()["ok"]
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "d.jpg"
    assert service.skip()["ok"]
    assert service.next()["done"]
    service.cancel()

    # Next session: the session skip is gone, the permanent one is not.
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    status = _wait_finished(service)
    assert status["permanently_skipped"] == 1 and status["skipped_forever"] == 0
    assert status["candidates"] == 1
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "d.jpg"


def test_next_never_serves_a_photo_that_was_skipped_after_the_queue_was_built(service, scores, fixture_dirs):
    """``_ready`` is filled by the scan before the session's skips exist (and
    ``undo`` can push a candidate back into it), so ``next`` has to re-check
    the skip sets rather than trusting the queue."""
    from modules.util.dpo_curation_util import _normalize_source_path

    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    assert service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")["ok"]
    _wait_finished(service)
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "a.jpg"

    # d.jpg is still queued; mark it the way a skip would while it sits there.
    service._session_skipped.add(_normalize_source_path(str(real / "d.jpg")))
    assert service.skip()["ok"]
    assert service.next()["done"] is True


def test_skip_forever_round_trips_through_the_http_router(service, scores, fixture_dirs):
    """Covers the endpoint wiring end to end — the router is mounted under
    ``/api``, so a stale backend process would 404 here."""
    from web.backend.main import app

    from fastapi.testclient import TestClient

    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    client = TestClient(app)
    res = client.post(
        "/api/dpo/offpolicy/start",
        json={
            "generated_folder": str(gen),
            "real_folder": str(real),
            "output_dir": str(out),
            "aspect_tolerance": 0.2,
            "order": "path",
        },
    )
    assert res.status_code == 200 and res.json()["ok"]
    _wait_finished(service)

    candidate = client.post("/api/dpo/offpolicy/next").json()["candidate"]
    assert os.path.basename(candidate["real_path"]) == "a.jpg"
    res = client.post("/api/dpo/offpolicy/skip-forever")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "permanently_skipped": 1}
    assert client.get("/api/dpo/offpolicy/status").json()["skipped_forever"] == 1

    # A restart of the whole session must not re-offer it.
    client.post("/api/dpo/offpolicy/cancel")
    client.post(
        "/api/dpo/offpolicy/start",
        json={
            "generated_folder": str(gen),
            "real_folder": str(real),
            "output_dir": str(out),
            "aspect_tolerance": 0.2,
            "order": "path",
        },
    )
    _wait_finished(service)
    assert client.get("/api/dpo/offpolicy/status").json()["permanently_skipped"] == 1
    again = client.post("/api/dpo/offpolicy/next").json()["candidate"]
    assert os.path.basename(again["real_path"]) == "d.jpg"


def test_a_pre_session_skip_file_does_not_hide_photos(service, scores, fixture_dirs):
    """The old always-permanent skip file is retired, not honoured: its photos
    are offered again — that is the point of session-only skips."""
    from modules.util.dpo_offpolicy_util import MIGRATED_SUFFIX, OBSOLETE_SKIP_FILENAMES

    real, gen, out = fixture_dirs
    scores.update({"a2_soft.png": 0.9, "a2_hard.png": 0.1, "d.png": 0.5})
    out.mkdir(exist_ok=True)
    obsolete = out / OBSOLETE_SKIP_FILENAMES[0]
    obsolete.write_text(json.dumps({"skipped": [str(real / "a.jpg")]}), encoding="utf-8")

    result = service.start(str(gen), str(real), str(out), aspect_tolerance=0.2, order="path")
    assert result["ok"] and result.get("permanently_skipped", 0) == 0
    status = _wait_finished(service)
    assert status["candidates"] == 2  # a.jpg is back
    assert os.path.basename(service.next()["candidate"]["real_path"]) == "a.jpg"
    assert not obsolete.exists()
    assert (out / (OBSOLETE_SKIP_FILENAMES[0] + MIGRATED_SUFFIX)).exists()
