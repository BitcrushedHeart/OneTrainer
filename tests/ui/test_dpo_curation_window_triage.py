"""Headless tests for the desktop DPOCurationWindow triage logic.

The UI itself needs a display, but the non-UI glue (order-based pair building and
the commit/export/advance step) is plain Python. We construct the window via
``object.__new__`` to skip Tk entirely and stub the one UI-rebuild call.
"""

import pytest

from modules.util.dpo_curation_util import load_manifest

from PIL import Image

# Importing the window pulls in customtkinter; skip cleanly if it can't load.
dcwmod = pytest.importorskip("modules.ui.DPOCurationWindow")
DPOCurationWindow = dcwmod.DPOCurationWindow


def _bare_window():
    win = object.__new__(DPOCurationWindow)  # no __init__ -> no Tk/display
    win._triage_verdicts = {}
    win._triage_pairs = []
    win._triage_good_pool = []
    win._triage_bad_pool = []
    win._build_triage_pairing_ui = lambda: None  # stub the UI rebuild
    return win


def test_triage_build_pairs_order_and_pools():
    win = _bare_window()
    win.current_remaining_images = ["g0", "b0", "g1", "b1", "b2", "s0"]
    win._triage_verdicts = {
        "g0": "good",
        "g1": "good",
        "b0": "bad",
        "b1": "bad",
        "b2": "bad",
        "s0": "skip",
    }

    win._triage_build_pairs()

    # Order-based by default (Auto-align is the opt-in similarity pass).
    assert win._triage_pairs == [("g0", "b0"), ("g1", "b1")]
    assert win._triage_good_pool == []
    assert win._triage_bad_pool == ["b2"]
    assert win._triage_phase == "pairing"


def test_triage_confirm_exports_pairs_and_advances(tmp_path):
    win = _bare_window()
    out = tmp_path / "out"
    out.mkdir()
    win.output_dir = str(out)
    win.manifest = {"pairs": []}
    win._current_group = {"prompt": "a cat", "aspectratio": "1:1"}
    win.pairs_created_in_group = 0
    advanced = {"n": 0}
    win._advance_group = lambda: advanced.__setitem__("n", advanced["n"] + 1)

    srcs = {}
    for name in ("c0", "r0", "c1", "r1"):
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (8, 8), color="white").save(p)
        srcs[name] = str(p)
    win._triage_pairs = [(srcs["c0"], srcs["r0"]), (srcs["c1"], srcs["r1"])]
    win.current_remaining_images = [*srcs.values(), "leftover"]

    win._triage_confirm()

    assert advanced["n"] == 1
    assert win.pairs_created_in_group == 2
    assert win.current_remaining_images == ["leftover"]

    manifest = load_manifest(str(out))
    assert len(manifest["pairs"]) == 2
    assert (out / "chosen" / "pair_0000.png").is_file()
    assert (out / "rejected" / "pair_0000.png").is_file()
    # Caption is written on the chosen side only, from the group prompt.
    assert (out / "chosen" / "pair_0000.txt").read_text(encoding="utf-8") == "a cat"
    assert not (out / "rejected" / "pair_0000.txt").exists()


def test_triage_counts_and_empty_pool_guard():
    win = _bare_window()
    win.current_remaining_images = ["g0", "b0", "s0", "u0"]
    win._triage_verdicts = {"g0": "good", "b0": "bad", "s0": "skip"}
    assert win._triage_counts() == (1, 1, 1, 1)  # good, bad, skip, unscored

    # No good or no bad -> auto-align is a no-op (guarded before any model call).
    win._triage_pairs = []
    win._triage_good_pool = ["g0"]
    win._triage_bad_pool = []
    win._build_triage_pairing_ui = lambda: (_ for _ in ()).throw(AssertionError("should not rebuild"))
    win._triage_auto_align()  # returns early because bad pool is empty
