"""Sizing and batch planning for SwarmUI-generated off-policy negatives."""

from modules.util.dpo_offpolicy_util import GENERATION_SIZE_MULTIPLE, RealPhoto
from modules.util.swarm_client import SwarmClient, SwarmError, normalize_base_url
from modules.util.swarm_generate_util import (
    DEFAULT_QUANTIZATION,
    DEFAULT_TARGET_RESOLUTION,
    bucket_size_for_photo,
    bucket_sizes,
    filter_pool_lines,
    plan_generations,
)

from mgds.pipelineModules.AspectBucketing import AspectBucketing

import pytest

# --- generation size = the photo's own trainer bucket ------------------------


class _TrainerBuckets(AspectBucketing):
    """mgds' ``AspectBucketing`` driven through its real ``start()`` for a
    config whose only target resolution is ``DEFAULT_TARGET_RESOLUTION`` —
    wired like the trainer's loader, not through the tool's own lookup.

    The expected grid comes from here rather than from pinned numbers: the
    tool follows mgds' bucketing by design, so a change to the trainer's grid
    must move these tests with it instead of failing on a stale count."""

    _TARGET_NAME = "settings.target_resolution"

    def __init__(self):
        super().__init__(
            quantization=DEFAULT_QUANTIZATION,
            resolution_in_name="original_resolution",
            target_resolution_in_name=self._TARGET_NAME,
            enable_target_resolutions_override_in_name="concept.image.enable_resolution_override",
            target_resolutions_override_in_name="concept.image.resolution_override",
            target_frames_in_name="settings.target_frames",
            frame_dim_enabled=False,
            scale_resolution_out_name="scale_resolution",
            crop_resolution_out_name="crop_resolution",
            possible_resolutions_out_name="possible_resolutions",
        )
        self.start(0)

    def _get_previous_length(self, name):
        return 1 if name == self._TARGET_NAME else 0

    def _get_previous_item(self, variation, name, index):
        assert name == self._TARGET_NAME
        return str(DEFAULT_TARGET_RESOLUTION)

    def grid(self) -> set[tuple[int, int]]:
        """Every bucket as ``(width, height)``."""
        return {(int(w), int(h)) for h, w in self.bucket_resolutions[DEFAULT_TARGET_RESOLUTION]}

    def size_for(self, width: int, height: int) -> tuple[int, int]:
        """The ``(width, height)`` bucket the trainer puts this photo in."""
        bucket_h, bucket_w = self.bucket_for_aspect(height / width, DEFAULT_TARGET_RESOLUTION)
        return int(bucket_w), int(bucket_h)


@pytest.fixture(scope="module")
def trainer_buckets():
    return _TrainerBuckets()


def test_bucket_sizes_are_the_trainers_own_grid(trainer_buckets):
    sizes = bucket_sizes()
    assert len(sizes) == len(set(sizes))
    assert set(sizes) == trainer_buckets.grid()
    assert sizes == sorted(sizes, key=lambda size: -size[0] / size[1])  # widest first
    assert (1024, 1024) in sizes
    assert (1344, 768) in sizes and (768, 1344) in sizes
    for width, height in sizes:
        assert width % DEFAULT_QUANTIZATION == 0 and height % DEFAULT_QUANTIZATION == 0
        # Every bucket carries roughly the target pixel budget. The floor
        # leaves room for native generator presets, which mgds does not
        # normalize to the budget (1216x768 is 0.89 of 1024^2).
        assert 0.85 <= (width * height) / (DEFAULT_TARGET_RESOLUTION**2) <= 1.15
    # The grid is symmetric: every landscape bucket has a portrait twin.
    assert {(w, h) for w, h in sizes} == {(h, w) for w, h in sizes}


@pytest.mark.parametrize(
    ("photo", "expected"),
    [
        ((2000, 2000), (1024, 1024)),
        ((2400, 1600), (1280, 832)),  # 3:2
        ((1600, 2400), (832, 1280)),  # 2:3
        ((1920, 1080), (1344, 768)),  # 16:9
        ((2100, 1200), (1344, 768)),  # 7:4 -> nearest bucket
        ((4000, 1000), (2048, 512)),  # extreme panorama
    ],
)
def test_bucket_size_for_photo(photo, expected):
    assert bucket_size_for_photo(*photo) == expected


@pytest.mark.parametrize(
    ("photo", "expected"),
    [
        ((1000, 1000), (992, 992)),
        ((1000, 667), (992, 656)),
        ((600, 1500), (592, 1488)),
    ],
)
def test_photo_below_the_budget_generates_at_its_own_size(photo, expected):
    """A photo under ``DEFAULT_TARGET_RESOLUTION``^2 is never upscaled to meet
    a bucket: it generates at its own dimensions, snapped down to a renderable
    multiple, and the trainer's scale-crop handles the pair."""
    assert bucket_size_for_photo(*photo) == expected


def test_bucket_size_falls_back_for_unreadable_dimensions():
    assert bucket_size_for_photo(0, 0) == (1024, 1024)
    assert bucket_size_for_photo(-5, 100) == (1024, 1024)


def test_generation_size_is_the_trainers_bucket_or_the_photos_own_size(trainer_buckets):
    """At or above the pixel budget the size handed to SwarmUI is exactly the
    bucket the trainer puts the photo in — that is what removes cropping from
    the rejected side. Below it, the photo's own size snapped down to a
    renderable multiple, so the photo is never upscaled to meet its pair."""
    grid = trainer_buckets.grid()
    budget = DEFAULT_TARGET_RESOLUTION**2
    step = GENERATION_SIZE_MULTIPLE
    for width in (600, 1000, 1024, 1500, 2400, 4000):
        for height in (600, 1000, 1024, 1500, 2400, 4000):
            gen_w, gen_h = bucket_size_for_photo(width, height)
            if width * height >= budget:
                assert (gen_w, gen_h) == trainer_buckets.size_for(width, height)
                assert (gen_w, gen_h) in grid
            else:
                assert gen_w % step == 0 and gen_h % step == 0
                assert 0 <= width - gen_w < step and 0 <= height - gen_h < step


# --- planning ----------------------------------------------------------------


def _entry(caption, size=(2400, 1600), path="p.jpg"):
    return {"caption": caption, "real_size": list(size), "real_path": path}


def test_plan_carries_bucket_size_and_image_count():
    jobs, stats = plan_generations([_entry("A woman stands in a sunlit garden.")], images_per_caption=3)
    assert stats == {"tag_style": 0, "too_short": 0, "unusable": 0}
    (job,) = jobs
    assert (job["width"], job["height"]) == (1280, 832)
    assert job["images"] == 3
    assert job["real_path"] == "p.jpg"
    assert job["caption"] == "A woman stands in a sunlit garden."


def test_plan_drops_tag_style_captions_only_when_asked():
    entries = [
        _entry("blonde hair, blue eyes, 1girl, outdoors", path="tags.jpg"),
        _entry("A woman stands in a sunlit garden holding a watering can.", path="prose.jpg"),
    ]
    jobs, stats = plan_generations(entries, exclude_tag_style=True)
    assert [job["real_path"] for job in jobs] == ["prose.jpg"]
    assert stats["tag_style"] == 1
    jobs, stats = plan_generations(entries, exclude_tag_style=False)
    assert len(jobs) == 2 and stats["tag_style"] == 0


def test_plan_drops_short_captions_when_a_minimum_is_set():
    entries = [_entry("A woman.", path="short.jpg"), _entry("A woman stands in a sunlit garden.", path="ok.jpg")]
    jobs, stats = plan_generations(entries, min_caption_words=5)
    assert [job["real_path"] for job in jobs] == ["ok.jpg"]
    assert stats["too_short"] == 1


def _photo(*lines: str, path: str = "p.jpg") -> RealPhoto:
    return RealPhoto(path, path + ".txt", list(lines), 2400, 1600)


def test_pool_filter_judges_each_caption_line_not_the_whole_sidecar():
    long_line = "A woman stands in a sunlit garden holding a watering can."
    pool = [
        (_photo("A woman.", long_line, path="mixed.jpg"), ["A woman.", long_line]),
        (_photo("A woman.", "A man.", path="short.jpg"), ["A woman.", "A man."]),
        (_photo("blonde hair, blue eyes, 1girl", path="tags.jpg"), ["blonde hair, blue eyes, 1girl"]),
    ]
    narrowed, stats = filter_pool_lines(pool, exclude_tag_style=True, min_caption_words=5)
    # The mixed sidecar keeps only its long line, so a draw can never land on
    # the short one and drop the photo.
    assert [(p.path, lines) for p, lines in narrowed] == [("mixed.jpg", [long_line])]
    assert stats == {"tag_style": 1, "too_short": 1}


def test_pool_filter_is_a_no_op_when_both_filters_are_off():
    pool = [(_photo("A woman.", "blonde hair, blue eyes, 1girl"), ["A woman.", "blonde hair, blue eyes, 1girl"])]
    narrowed, stats = filter_pool_lines(pool)
    assert narrowed == pool and stats == {"tag_style": 0, "too_short": 0}


def test_pool_filter_reports_a_short_tag_list_as_tag_style():
    pool = [(_photo("1girl, smile"), ["1girl, smile"])]
    _narrowed, stats = filter_pool_lines(pool, exclude_tag_style=True, min_caption_words=5)
    assert stats == {"tag_style": 1, "too_short": 0}


def test_plan_skips_entries_it_cannot_size_or_prompt():
    entries = [
        _entry("", path="nocaption.jpg"),
        _entry("A woman stands in a garden.", size=(0, 0), path="nosize.jpg"),
        {"caption": "A woman stands in a garden.", "real_size": "nonsense", "real_path": "bad.jpg"},
    ]
    jobs, stats = plan_generations(entries)
    assert jobs == []
    assert stats["unusable"] == 3


# --- client ------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeHTTP:
    """Stands in for a requests.Session; records calls and replays scripted
    responses keyed on the API route."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.cookies = type("Jar", (), {"set": lambda self, *a, **k: None})()

    def post(self, url, json=None, timeout=None):
        route = url.rsplit("/", 1)[-1]
        # Copy: the client reuses its payload dict across a session retry,
        # so recording it by reference would rewrite history.
        self.calls.append((route, dict(json or {})))
        payload = self.script[route]
        return _FakeResponse(payload.pop(0) if isinstance(payload, list) else payload)


def _client(script, **kwargs):
    client = SwarmClient("localhost:7801", **kwargs)
    client._http = _FakeHTTP(script)
    return client


def test_normalize_base_url_adds_scheme_and_trims():
    assert normalize_base_url("localhost:7801") == "http://localhost:7801"
    assert normalize_base_url("http://box:7801/") == "http://box:7801"
    assert normalize_base_url("") == "http://localhost:7801"
    assert normalize_base_url("https://swarm.example") == "https://swarm.example"


def test_generate_passes_the_exact_size_and_returns_paths():
    client = _client(
        {
            "GetNewSession": {"session_id": "S1", "version": "0.9"},
            "GenerateText2Image": {"images": ["View/local/raw/2026-08-30/a.png"]},
        }
    )
    paths = client.generate("a caption", 1280, 832, "MyModels/x.safetensors", images=2, steps=8, cfg_scale=1.5)
    assert paths == ["View/local/raw/2026-08-30/a.png"]
    route, payload = client._http.calls[-1]
    assert route == "GenerateText2Image"
    assert payload["width"] == 1280 and payload["height"] == 832
    assert payload["images"] == 2 and payload["steps"] == 8 and payload["cfgscale"] == 1.5
    assert payload["seed"] == -1  # a batch of negatives must vary
    assert payload["session_id"] == "S1"


def test_stale_session_is_reacquired_once():
    client = _client(
        {
            "GetNewSession": [{"session_id": "S1"}, {"session_id": "S2"}],
            "GenerateText2Image": [{"error_id": "invalid_session_id"}, {"images": ["View/x.png"]}],
        }
    )
    assert client.generate("c", 1024, 1024, "m") == ["View/x.png"]
    sessions = [payload.get("session_id") for route, payload in client._http.calls if route == "GenerateText2Image"]
    assert sessions == ["S1", "S2"]


def test_other_errors_surface():
    client = _client({"GetNewSession": {"session_id": "S1"}, "GenerateText2Image": {"error": "no such model"}})
    with pytest.raises(SwarmError, match="no such model"):
        client.generate("c", 1024, 1024, "missing")


def test_list_models_sorts_names():
    client = _client(
        {
            "GetNewSession": {"session_id": "S1"},
            "ListModels": {"files": [{"name": "b.safetensors"}, {"name": "a.safetensors"}, {}]},
        }
    )
    assert client.list_models() == ["a.safetensors", "b.safetensors"]
