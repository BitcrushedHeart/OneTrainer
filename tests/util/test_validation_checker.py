import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.validation_checker_util import (
    TRAIN_CONCEPT_TYPES,
    VALIDATION_CONCEPT_TYPES,
    build_explorer_command,
    collect_validation_checker_concepts,
    collect_validation_checker_images,
    dhash_variants,
    hamming_distance,
    move_validation_image_to_train,
    remove_validation_image,
    scan_basic_validation_matches,
    scan_clip_similarity_matches,
)


class ValidationCheckerUtilTest(unittest.TestCase):
    def test_exact_duplicate_detection_flags_identical_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            train_image = train_dir / "img_a.png"
            val_image = val_dir / "img_a.png"
            self._create_pattern_image(train_image)
            shutil.copy2(train_image, val_image)

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))

            self.assertEqual(len(result.exact_matches), 1)
            self.assertEqual(result.exact_matches[0].train_image.path, train_image.as_posix())
            self.assertEqual(result.exact_matches[0].val_image.path, val_image.as_posix())
            self.assertEqual(len(result.perceptual_matches), 0)

    def test_exact_duplicate_detection_ignores_different_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            self._create_pattern_image(train_dir / "img_a.png")
            self._create_pattern_image(val_dir / "img_b.png", variant="different")

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))

            self.assertEqual(result.exact_matches, [])

    def test_perceptual_detection_flags_horizontal_flip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            train_image = train_dir / "img_a.png"
            val_image = val_dir / "img_flip.png"
            self._create_pattern_image(train_image)
            with Image.open(train_image) as image:
                image.transpose(Image.FLIP_LEFT_RIGHT).save(val_image)

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))

            self.assertEqual(len(result.perceptual_matches), 1)
            self.assertIn("horizontal flip", result.perceptual_matches[0].detail)

    def test_perceptual_detection_flags_center_crop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            train_image = train_dir / "img_a.png"
            val_image = val_dir / "img_crop.png"
            self._create_pattern_image(train_image)
            with Image.open(train_image) as image:
                cropped = image.crop((6, 6, image.width - 6, image.height - 6))
                cropped.save(val_image)

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))

            self.assertEqual(len(result.perceptual_matches), 1)
            self.assertIn("center crop", result.perceptual_matches[0].detail)

    def test_perceptual_detection_ignores_distinct_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            self._create_pattern_image(train_dir / "img_a.png")
            self._create_pattern_image(val_dir / "img_b.png", variant="different")

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))

            self.assertEqual(result.perceptual_matches, [])

    def test_hamming_threshold_uses_real_image_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "original.png"
            minor = root / "minor.jpg"
            different = root / "different.png"

            self._create_pattern_image(original)
            self._create_pattern_image(different, variant="different")

            with Image.open(original) as image:
                adjusted = Image.new("RGB", image.size)
                adjusted.paste(image)
                draw = ImageDraw.Draw(adjusted)
                draw.rectangle((42, 42, 48, 48), fill=(245, 220, 0))
                adjusted.save(minor, quality=92)

            original_hash = dict(dhash_variants(original.as_posix()))["original"]
            minor_hash = dict(dhash_variants(minor.as_posix()))["original"]
            different_hash = dict(dhash_variants(different.as_posix()))["original"]

            self.assertLessEqual(hamming_distance(original_hash, minor_hash), 5)
            self.assertGreater(hamming_distance(original_hash, different_hash), 5)

    def test_caption_matches_flag_exact_prompt_reuse_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            train_image = train_dir / "img_a.png"
            val_match = val_dir / "img_b.png"
            val_non_match = val_dir / "img_c.png"
            self._create_pattern_image(train_image)
            self._create_pattern_image(val_match, variant="different")
            self._create_pattern_image(val_non_match, variant="different_2")

            train_image.with_suffix(".txt").write_text("a cat sitting on a windowsill", encoding="utf-8")
            val_match.with_suffix(".txt").write_text("a cat sitting on a windowsill", encoding="utf-8")
            val_non_match.with_suffix(".txt").write_text("a cat on a windowsill", encoding="utf-8")

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))

            self.assertEqual(len(result.caption_matches), 1)
            self.assertEqual(result.caption_matches[0].caption, "a cat sitting on a windowsill")
            self.assertEqual(len(result.caption_matches[0].train_images), 1)
            self.assertEqual(len(result.caption_matches[0].val_images), 1)

    def test_collect_validation_checker_concepts_routes_current_master_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            standard_dir = root / "standard"
            prior_dir = root / "prior"
            val_dir = root / "val"
            standard_dir.mkdir()
            prior_dir.mkdir()
            val_dir.mkdir()

            standard = ConceptConfig.default_values()
            standard.path = standard_dir.as_posix()
            standard.type = ConceptType.STANDARD

            prior = ConceptConfig.default_values()
            prior.path = prior_dir.as_posix()
            prior.type = ConceptType.PRIOR_PREDICTION

            validation = ConceptConfig.default_values()
            validation.path = val_dir.as_posix()
            validation.type = ConceptType.VALIDATION

            train_concepts, val_concepts = collect_validation_checker_concepts([standard, prior, validation])

            self.assertEqual({concept.concept_type for concept in train_concepts}, TRAIN_CONCEPT_TYPES)
            self.assertEqual({concept.concept_type for concept in val_concepts}, VALIDATION_CONCEPT_TYPES)

    def test_collect_validation_checker_images_skips_hidden_and_auxiliary_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            hidden_dir = train_dir / ".hidden"
            train_dir.mkdir()
            val_dir.mkdir()
            hidden_dir.mkdir()

            self._create_pattern_image(train_dir / "visible.png")
            self._create_pattern_image(train_dir / "visible-masklabel.png")
            self._create_pattern_image(train_dir / "visible-condlabel.png")
            self._create_pattern_image(hidden_dir / "hidden.png")

            train_concepts, _ = collect_validation_checker_concepts(self._build_concepts(train_dir, val_dir))
            images = collect_validation_checker_images(train_concepts)

            self.assertEqual(len(images), 1)
            self.assertTrue(images[0].path.endswith("visible.png"))

    def test_remove_validation_image_deletes_sidecar_only_from_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            train_image = train_dir / "img_a.png"
            val_image = val_dir / "img_b.png"
            self._create_pattern_image(train_image)
            self._create_pattern_image(val_image)
            train_sidecar = train_image.with_suffix(".txt")
            val_sidecar = val_image.with_suffix(".txt")
            train_sidecar.write_text("train", encoding="utf-8")
            val_sidecar.write_text("val", encoding="utf-8")

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))
            remove_validation_image(result.val_images[0])

            self.assertTrue(train_image.exists())
            self.assertTrue(train_sidecar.exists())
            self.assertFalse(val_image.exists())
            self.assertFalse(val_sidecar.exists())

    def test_move_validation_image_to_train_moves_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            val_image = val_dir / "img_b.png"
            self._create_pattern_image(train_dir / "img_a.png")
            self._create_pattern_image(val_image)
            val_sidecar = val_image.with_suffix(".txt")
            val_sidecar.write_text("val", encoding="utf-8")

            concepts = self._build_concepts(train_dir, val_dir)
            train_concepts, val_concepts = collect_validation_checker_concepts(concepts)
            val_images = collect_validation_checker_images(val_concepts)

            moved_path = move_validation_image_to_train(val_images[0], train_concepts[0])

            self.assertFalse(val_image.exists())
            self.assertFalse(val_sidecar.exists())
            self.assertTrue(Path(moved_path).exists())
            self.assertTrue(Path(moved_path).with_suffix(".txt").exists())

    def test_build_explorer_command_preserves_spaces_and_commas(self):
        path = r"C:\datasets\val set\img,01.png"
        with mock.patch("modules.util.validation_checker_util.os.name", "nt"):
            command = build_explorer_command(path)
        self.assertEqual(command, ["explorer", "/select,", os.path.normpath(path)])

    def test_deep_scan_gracefully_fails_without_clip_support(self):
        with mock.patch("modules.util.validation_checker_util.load_clip_components", return_value=(None, None, None, None)):
            with self.assertRaisesRegex(RuntimeError, "Deep Scan requires"):
                scan_clip_similarity_matches([], [])

    @unittest.skipUnless(os.environ.get("VALIDATION_CHECKER_RUN_CLIP") == "1", "Set VALIDATION_CHECKER_RUN_CLIP=1 to run CLIP integration test.")
    def test_deep_scan_flags_high_similarity_when_clip_is_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            train_image = train_dir / "img_a.png"
            val_image = val_dir / "img_b.png"
            self._create_pattern_image(train_image)
            with Image.open(train_image) as image:
                image.crop((4, 4, image.width - 4, image.height - 4)).save(val_image)

            result = scan_basic_validation_matches(self._build_concepts(train_dir, val_dir))
            matches = scan_clip_similarity_matches(result.train_images, result.val_images, threshold=0.8)

            self.assertTrue(matches)

    def _build_concepts(self, train_dir: Path, val_dir: Path) -> list[ConceptConfig]:
        train = ConceptConfig.default_values()
        train.name = "train"
        train.path = train_dir.as_posix()
        train.type = ConceptType.STANDARD
        train.include_subdirectories = True

        validation = ConceptConfig.default_values()
        validation.name = "val"
        validation.path = val_dir.as_posix()
        validation.type = ConceptType.VALIDATION
        validation.include_subdirectories = True

        return [train, validation]

    def _create_pattern_image(self, path: Path, variant: str = "base"):
        image = Image.new("RGB", (64, 64), (235, 235, 235))
        draw = ImageDraw.Draw(image)

        if variant == "base":
            draw.rectangle((8, 8, 22, 56), fill=(30, 30, 30))
            draw.rectangle((40, 8, 58, 22), fill=(220, 30, 30))
            draw.polygon(((34, 38), (57, 54), (48, 30)), fill=(40, 60, 210))
            draw.line((25, 12, 52, 45), fill=(30, 150, 30), width=4)
        elif variant == "different":
            draw.rectangle((6, 6, 58, 18), fill=(30, 30, 30))
            draw.ellipse((10, 26, 34, 52), fill=(40, 60, 210))
            draw.rectangle((42, 28, 58, 58), fill=(220, 30, 30))
        elif variant == "different_2":
            draw.ellipse((8, 8, 28, 28), fill=(220, 30, 30))
            draw.rectangle((22, 22, 42, 58), fill=(30, 30, 30))
            draw.line((8, 56, 58, 8), fill=(40, 60, 210), width=6)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        image.save(path)

