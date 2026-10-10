"""Output lookup tests use only temporary placeholders and fake HTTP responses."""

from types import SimpleNamespace

from modules.util.dpo_replace_util import local_path_for_view
from modules.util.swarm_client import SwarmClient, SwarmError

import pytest


@pytest.mark.parametrize(
    "view",
    [
        "View/local/raw/day/test%20image.png?download=1#preview",
        "/View/local/raw/day/test%20image.png",
        "http://localhost:7801/View/local/raw/day/test%20image.png?cache=123",
        "View\\local\\raw\\day\\test%20image.png",
    ],
)
def test_view_urls_are_decoded_for_local_lookup(tmp_path, view):
    folder = tmp_path / "day"
    folder.mkdir()
    image = folder / "test image.png"
    image.write_bytes(b"placeholder; never decoded as an image")
    assert local_path_for_view(view, str(tmp_path)) == str(image)


def test_decoded_filename_fallback_handles_different_output_layout(tmp_path):
    folder = tmp_path / "Output/local/raw/day"
    folder.mkdir(parents=True)
    image = folder / "test image.png"
    image.touch()
    assert local_path_for_view("View/local/raw/day/test%20image.png?cache=1", str(tmp_path)) == str(image)


@pytest.mark.parametrize(
    "view,expected",
    [
        (
            "View/local/raw/test%20image.png?cache=1",
            "http://localhost:7801/swarm/View/local/raw/test%20image.png?cache=1",
        ),
        ("/View/local/raw/test.png", "http://localhost:7801/swarm/View/local/raw/test.png"),
        ("http://localhost:7801/swarm/View/test.png?cache=1", "http://localhost:7801/swarm/View/test.png?cache=1"),
        ("//localhost:7801/swarm/View/test.png", "http://localhost:7801/swarm/View/test.png"),
    ],
)
def test_fetch_view_accepts_relative_and_same_server_absolute_urls(view, expected):
    calls = []
    client = SwarmClient("http://localhost:7801/swarm", token="test-token")

    def get(url, timeout):
        calls.append((url, timeout))
        return SimpleNamespace(status_code=200, content=b"synthetic-response")

    # Keep the real configured session so the authentication cookie is tested,
    # but replace its get method: no network connection is made.
    client._client().get = get
    assert client.fetch_view(view) == b"synthetic-response"
    assert calls == [(expected, 120)]
    assert client._client().cookies.get("swarm_token") == "test-token"


@pytest.mark.parametrize(
    "view", ["https://example.invalid/View/a.png", "//other.invalid/a.png", "http://localhost:7802/a.png"]
)
def test_fetch_view_rejects_other_servers(view):
    with pytest.raises(SwarmError, match="outside the configured server"):
        SwarmClient("http://localhost:7801").fetch_view(view)
