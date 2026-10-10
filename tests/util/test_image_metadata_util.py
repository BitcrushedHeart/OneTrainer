import json
import tempfile
import unittest
from pathlib import Path

from modules.util.image_metadata_util import extract_metadata, strip_angle_bracket_segments

from PIL import Image
from PIL.PngImagePlugin import PngInfo


class ImageMetadataUtilTest(unittest.TestCase):
    def test_strip_angle_bracket_segments_removes_metadata_like_tags_from_prompt(self):
        prompt = "portrait, <wildcard:hair>, cinematic light, <segment:face>, sharp focus"
        self.assertEqual(
            strip_angle_bracket_segments(prompt),
            "portrait, cinematic light, sharp focus",
        )

    def test_strip_angle_bracket_segments_preserves_inline_text_around_removed_tags(self):
        prompt = "foo<wildcard:bar>baz"
        self.assertEqual(strip_angle_bracket_segments(prompt), "foobaz")

    def test_strip_angle_bracket_segments_removes_orphaned_weight_tails(self):
        for prompt in (
            "portrait, 0.65||>, sharp focus",
            "portrait, 0.85||> sharp focus",
            "portrait 0.65>, sharp focus",
            "portrait, <segment:face,0.65||>, 0.65>, sharp focus",
            "portrait, segment:face||>, sharp focus",
            "portrait, segment:face,0.6,0.5> sharp focus",
            "portrait, Segment:face||> 0.2||>, sharp focus",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(strip_angle_bracket_segments(prompt), "portrait, sharp focus")

    def test_strip_angle_bracket_segments_keeps_numbers_without_closing_bracket(self):
        prompt = "3 people, 1.5m tall, v1.4 style, subject: woman, base: marble"
        self.assertEqual(strip_angle_bracket_segments(prompt), prompt)

    def test_extract_metadata_reads_png_json_prompt_and_aspectratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            png_info = PngInfo()
            png_info.add_text(
                "Comment",
                json.dumps(
                    {
                        "prompt": "portrait, <segment:face>, sharp focus",
                        "aspectratio": "16:9",
                    }
                ),
            )

            Image.new("RGB", (8, 8), color="green").save(image_path, pnginfo=png_info)

            self.assertEqual(
                extract_metadata(str(image_path)),
                {
                    "prompt": "portrait, <segment:face>, sharp focus",
                    "aspectratio": "16:9",
                    "loras": [],
                },
            )

    def test_extract_metadata_reads_swarmui_png_nested_under_parameters(self):
        # SwarmUI writes one tEXt chunk keyed "parameters" whose value is JSON
        # with the generation settings nested a level down under
        # "sui_image_params" — there is no top-level "prompt". Reading only the
        # top level loses every PNG the tool produces, while WEBP survives
        # because the raw scanner regexes the nested key incidentally.
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            png_info = PngInfo()
            png_info.add_text(
                "parameters",
                json.dumps(
                    {
                        "sui_image_params": {
                            "prompt": "portrait, <wildcard:hair>, sharp focus",
                            "negativeprompt": "blurry",
                            "aspectratio": "3:2",
                            "width": 1200,
                            "height": 800,
                        },
                        "sui_extra_data": {"date": "2026-08-11"},
                        "sui_models": [{"name": "some-model"}],
                    }
                ),
            )

            Image.new("RGB", (8, 8), color="blue").save(image_path, pnginfo=png_info)

            self.assertEqual(
                extract_metadata(str(image_path)),
                {
                    "prompt": "portrait, <wildcard:hair>, sharp focus",
                    "aspectratio": "3:2",
                    "loras": [],
                },
            )

    def test_extract_metadata_prefers_top_level_prompt_over_nested_params(self):
        # A top-level "prompt" is the explicit shape; nesting is the fallback.
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            png_info = PngInfo()
            png_info.add_text(
                "parameters",
                json.dumps(
                    {
                        "prompt": "top level wins",
                        "aspectratio": "1:1",
                        "sui_image_params": {"prompt": "nested loses", "aspectratio": "16:9"},
                    }
                ),
            )

            Image.new("RGB", (8, 8), color="red").save(image_path, pnginfo=png_info)

            self.assertEqual(
                extract_metadata(str(image_path)),
                {"prompt": "top level wins", "aspectratio": "1:1", "loras": []},
            )

    def test_extract_metadata_ignores_nested_params_with_blank_prompt(self):
        # An empty nested prompt must not shadow a later chunk that has one.
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            png_info = PngInfo()
            png_info.add_text(
                "parameters",
                json.dumps({"sui_image_params": {"prompt": "   ", "aspectratio": "3:2"}}),
            )
            png_info.add_text(
                "Comment",
                json.dumps({"prompt": "the real one", "aspectratio": "4:3"}),
            )

            Image.new("RGB", (8, 8), color="black").save(image_path, pnginfo=png_info)

            self.assertEqual(
                extract_metadata(str(image_path)),
                {"prompt": "the real one", "aspectratio": "4:3", "loras": []},
            )

    def test_extract_metadata_reads_raw_json_and_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.webp"
            image_path.write_bytes(b'prefix {"prompt":"caf\\u00e9, <wildcard:fur>, cat","aspectratio":"4:3"} suffix')

            self.assertEqual(
                extract_metadata(str(image_path)),
                {
                    "prompt": "caf\u00e9, <wildcard:fur>, cat",
                    "aspectratio": "4:3",
                    "loras": [],
                },
            )

    def test_extract_metadata_reads_utf16le_encoded_webp_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.webp"
            json_block = '{"prompt":"a beautiful landscape","aspectratio":"16:9"}'
            # Embed the JSON as UTF-16LE (as some generators do in WebP EXIF)
            utf16_payload = json_block.encode("utf-16-le")
            image_path.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + utf16_payload)

            self.assertEqual(
                extract_metadata(str(image_path)),
                {
                    "prompt": "a beautiful landscape",
                    "aspectratio": "16:9",
                    "loras": [],
                },
            )

    def test_iter_json_objects_resets_state_between_objects(self):
        from modules.util.image_metadata_util import _iter_json_objects

        # Two valid JSON objects separated by garbage
        text = '{"a":1} junk {"b":2}'
        objects = list(_iter_json_objects(text))
        self.assertEqual(len(objects), 2)
        self.assertEqual(json.loads(objects[0]), {"a": 1})
        self.assertEqual(json.loads(objects[1]), {"b": 2})

    def test_iter_json_objects_handles_strings_with_braces(self):
        from modules.util.image_metadata_util import _iter_json_objects

        text = '{"prompt":"a {nested} thing"}'
        objects = list(_iter_json_objects(text))
        self.assertEqual(len(objects), 1)
        self.assertEqual(json.loads(objects[0]), {"prompt": "a {nested} thing"})
