"""``save_to`` runs: results SwarmUI did not save are written as self-describing PNGs."""

import base64
import io
import json
import os

from modules.util.dpo_replace_util import file_is_generation, local_path_for_view
from modules.util.image_metadata_util import extract_metadata
from modules.util.swarm_generate_util import _params_from_image, save_generation_locally
from web.backend.services.swarm_generate_service import SwarmGenerateService

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(with_params: dict | None = None) -> bytes:
    buf = io.BytesIO()
    info = PngInfo()
    if with_params is not None:
        info.add_text("sui_image_params", json.dumps({"sui_image_params": with_params}))
    Image.new("RGB", (8, 8), (90, 120, 200)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def test_save_generation_locally_embeds_request_params(tmp_path):
    request = {"prompt": "a photo of rosie", "model": "SoReal", "loras": ["SoRealPeople!/RosieButler"], "width": 8}
    target = save_generation_locally(_png(), str(tmp_path / "gen"), "stem", request)
    assert os.path.isfile(target) and target.endswith("stem.png")
    assert _params_from_image(target) == request
    assert extract_metadata(target)["prompt"] == "a photo of rosie"
    assert extract_metadata(target)["loras"] == ["sorealpeople!/rosiebutler"]
    assert file_is_generation(target)
    # A second image for the same stem gets a numbered name instead of overwriting.
    second = save_generation_locally(_png(), str(tmp_path / "gen"), "stem", request)
    assert second.endswith("stem-2.png")


def test_save_generation_locally_keeps_swarm_block(tmp_path):
    swarm = {"prompt": "swarm's own", "model": "X"}
    target = save_generation_locally(_png(swarm), str(tmp_path), "s", {"prompt": "ignored"})
    assert _params_from_image(target) == swarm


def test_store_locally_decodes_data_uris_and_downloads_views(tmp_path):
    class Client:
        def fetch_view(self, view):
            assert view == "View/local/raw/x.png"
            return _png()

    data_uri = "data:image/png;base64," + base64.b64encode(_png()).decode()
    paths = SwarmGenerateService._store_locally(
        Client(), [data_uri, "View/local/raw/x.png"], str(tmp_path), {"prompt": "p", "donotsave": True}
    )
    assert len(paths) == 2 and all(os.path.isfile(p) for p in paths)
    assert all("donotsave" not in _params_from_image(p) for p in paths)
    assert all(_params_from_image(p)["prompt"] == "p" for p in paths)
    # The run loop maps results back to disk through local_path_for_view; a local path is returned as is.
    assert local_path_for_view(paths[0], str(tmp_path)) == paths[0]
