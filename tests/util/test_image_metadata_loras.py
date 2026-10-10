"""``extract_metadata(...)["loras"]``: the LoRA names an image was generated
with, read from SwarmUI's parameter block and from ``<lora:...>`` prompt tags."""

import json
import os
import tempfile
import unittest

from modules.util.image_metadata_util import extract_metadata

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(path: str, chunks: dict[str, str]) -> str:
    info = PngInfo()
    for key, value in chunks.items():
        info.add_text(key, value)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, pnginfo=info)
    return path


def _swarm(params: dict, nested: bool = True) -> str:
    """SwarmUI's chunk payload: current versions nest the settings under
    ``sui_image_params``; older ones put them at the top level."""
    return json.dumps({"sui_image_params": params} if nested else params)


class ImageMetadataLorasTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def test_loras_list_in_nested_swarm_params(self):
        path = _png(
            self._path("a.png"),
            {
                "sui_image_params": _swarm(
                    {"prompt": "a photo of a woman", "aspectratio": "1:1", "loras": ["Flux/MyChar", "detail_boost"]}
                )
            },
        )
        meta = extract_metadata(path)
        self.assertEqual(meta["prompt"], "a photo of a woman")
        self.assertEqual(meta["aspectratio"], "1:1")
        self.assertEqual(meta["loras"], ["flux/mychar", "detail_boost"])

    def test_loras_comma_string_in_top_level_swarm_params(self):
        path = _png(
            self._path("b.png"),
            {"sui_image_params": _swarm({"prompt": "a cat", "loras": "CatLora, Sharpen"}, nested=False)},
        )
        self.assertEqual(extract_metadata(path)["loras"], ["catlora", "sharpen"])

    def test_prompt_tags_merge_with_params_and_dedupe(self):
        path = _png(
            self._path("c.png"),
            {
                "sui_image_params": _swarm(
                    {
                        "prompt": "a dog <lora:DogLora:0.8>, <lora:Extra>, sunny <LoRA:doglora:1>",
                        "loras": ["DogLora"],
                    }
                )
            },
        )
        meta = extract_metadata(path)
        # Params first, then tags in prompt order; the duplicate collapses.
        self.assertEqual(meta["loras"], ["doglora", "extra"])
        # The prompt itself is returned raw: tag stripping is the caller's job.
        self.assertIn("<lora:DogLora:0.8>", meta["prompt"])

    def test_prompt_tags_without_params_list(self):
        path = _png(
            self._path("d.png"),
            {"sui_image_params": _swarm({"prompt": "portrait <lora:Style_A:0.6:0.4> at night"})},
        )
        self.assertEqual(extract_metadata(path)["loras"], ["style_a"])

    def test_no_loras_gives_empty_list(self):
        path = _png(self._path("e.png"), {"sui_image_params": _swarm({"prompt": "plain", "loras": []})})
        self.assertEqual(extract_metadata(path)["loras"], [])
        bare = _png(self._path("f.png"), {"prompt": "just a prompt chunk"})
        meta = extract_metadata(bare)
        self.assertEqual(meta["prompt"], "just a prompt chunk")
        self.assertEqual(meta["loras"], [])
        none = _png(self._path("g.png"), {})
        self.assertEqual(extract_metadata(none), {"prompt": "", "aspectratio": "", "loras": []})

    def test_a1111_parameters_prompt_tags(self):
        text = "a knight <lora:ArmorLora:0.7>, dramatic\nNegative prompt: blurry\nSteps: 20, Sampler: Euler"
        path = _png(self._path("h.png"), {"parameters": text})
        meta = extract_metadata(path)
        self.assertEqual(meta["prompt"], "a knight <lora:ArmorLora:0.7>, dramatic")
        self.assertEqual(meta["loras"], ["armorlora"])

    def test_comfyui_lora_loader_nodes(self):
        workflow = {
            "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "Chars/Hero.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hero on a hill"}},
            "3": {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}},
        }
        path = _png(self._path("i.png"), {"prompt": json.dumps(workflow)})
        meta = extract_metadata(path)
        self.assertEqual(meta["prompt"], "hero on a hill")
        self.assertEqual(meta["loras"], ["chars/hero"])

    def test_jpeg_swarm_block_in_raw_bytes(self):
        """The byte scanner finds the SwarmUI block in non-PNG containers and
        reads the ``loras`` list from the same text."""
        path = self._path("j.jpg")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(path, quality=50)
        block = json.dumps({"sui_image_params": {"prompt": "jpeg prompt", "aspectratio": "4:3", "loras": ["Jpg/Lora"]}})
        with open(path, "ab") as f:
            f.write(block.encode("utf-8"))
        meta = extract_metadata(path)
        self.assertEqual(meta["prompt"], "jpeg prompt")
        self.assertEqual(meta["aspectratio"], "4:3")
        self.assertEqual(meta["loras"], ["jpg/lora"])

    def test_jpeg_without_metadata(self):
        path = self._path("k.jpg")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(path, quality=50)
        self.assertEqual(extract_metadata(path), {"prompt": "", "aspectratio": "", "loras": []})


if __name__ == "__main__":
    unittest.main()
