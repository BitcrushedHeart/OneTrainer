"""
E2E test for DPOCurationWindow: image scaling (_fit_image) and prompt expander.

Runs headless via Tk withdraw() — no visible window appears.
"""
import json
import os
import tempfile
import unittest

from PIL import Image, PngImagePlugin

# Single shared Tk root + DPOCurationWindow for all tests (Tcl can only init once).
_root = None
_win = None


def _get_win():
    global _root, _win
    if _root is None:
        import customtkinter as ctk
        _root = ctk.CTk()
        _root.withdraw()
        from modules.ui.DPOCurationWindow import DPOCurationWindow
        _win = DPOCurationWindow(_root)
        _win.withdraw()
        _win.geometry("1400x900")
        _win.update_idletasks()
    return _win


def _create_test_png(path: str, width: int, height: int, prompt: str, aspectratio: str = "1:1"):
    """Create a PNG with SwarmUI-style metadata embedded."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    info = PngImagePlugin.PngInfo()
    info.add_text("sui_image_params", json.dumps({
        "prompt": prompt,
        "aspectratio": aspectratio,
    }))
    img.save(path, pnginfo=info)


class TestFitImage(unittest.TestCase):
    """Test that _fit_image scales both up and down while preserving aspect ratio."""

    @classmethod
    def setUpClass(cls):
        cls.win = _get_win()

    def test_scale_down_landscape(self):
        """Large landscape image should scale down to fit 400x400 box."""
        img = Image.new("RGB", (1200, 800))
        result = self.win._fit_image(img, 400, 400)
        self.assertEqual(result.width, 400)
        self.assertEqual(result.height, 266)  # 800 * (400/1200)

    def test_scale_down_portrait(self):
        """Large portrait image should scale down to fit 400x400 box."""
        img = Image.new("RGB", (800, 1200))
        result = self.win._fit_image(img, 400, 400)
        self.assertEqual(result.width, 266)
        self.assertEqual(result.height, 400)

    def test_scale_up_small_image(self):
        """Small image should scale UP to fill 400x400 box."""
        img = Image.new("RGB", (100, 50))
        result = self.win._fit_image(img, 400, 400)
        self.assertEqual(result.width, 400)
        self.assertEqual(result.height, 200)

    def test_scale_up_tiny_square(self):
        """Tiny square image should scale up to exactly fill the box."""
        img = Image.new("RGB", (10, 10))
        result = self.win._fit_image(img, 250, 250)
        self.assertEqual(result.width, 250)
        self.assertEqual(result.height, 250)

    def test_exact_fit_no_change(self):
        """Image exactly matching the box should stay the same size."""
        img = Image.new("RGB", (400, 400))
        result = self.win._fit_image(img, 400, 400)
        self.assertEqual(result.width, 400)
        self.assertEqual(result.height, 400)

    def test_aspect_ratio_preserved(self):
        """Aspect ratio must be preserved within rounding tolerance."""
        img = Image.new("RGB", (73, 137))
        result = self.win._fit_image(img, 500, 500)
        original_ar = img.width / img.height
        result_ar = result.width / result.height
        self.assertAlmostEqual(original_ar, result_ar, places=1)
        # Should have scaled up
        self.assertGreater(result.height, img.height)


class TestPromptExpander(unittest.TestCase):
    """Test that the prompt expander widget is created and toggles correctly."""

    @classmethod
    def setUpClass(cls):
        cls.win = _get_win()

    def test_short_prompt_no_ellipsis(self):
        """Short prompts should display in full without ellipsis."""
        frame = self.win._build_prompt_expander(self.win, "a short prompt")
        children = frame.winfo_children()
        self.assertEqual(len(children), 2)
        label = children[1]
        self.assertEqual(label.cget("text"), "a short prompt")
        frame.destroy()

    def test_long_prompt_truncated(self):
        """Long prompts should be truncated to 100 chars with ellipsis."""
        long_prompt = "a " * 200  # 400 chars
        frame = self.win._build_prompt_expander(self.win, long_prompt)
        children = frame.winfo_children()
        label = children[1]
        displayed = label.cget("text")
        self.assertTrue(displayed.endswith("..."))
        self.assertLessEqual(len(displayed), 104)  # 100 + "..."
        frame.destroy()

    def test_toggle_button_exists(self):
        """Toggle button should show 'Prompt [+]' initially."""
        frame = self.win._build_prompt_expander(self.win, "test prompt")
        btn = frame.winfo_children()[0]
        self.assertEqual(btn.cget("text"), "Prompt [+]")
        frame.destroy()

    def test_toggle_expands_and_collapses(self):
        """Clicking toggle should expand to full prompt then collapse back."""
        long_prompt = "word " * 50  # 250 chars
        frame = self.win._build_prompt_expander(self.win, long_prompt)
        btn = frame.winfo_children()[0]
        label = frame.winfo_children()[1]

        # Initially truncated
        self.assertTrue(label.cget("text").endswith("..."))

        # Simulate click to expand
        btn.invoke()
        self.assertEqual(btn.cget("text"), "Prompt [-]")
        self.assertEqual(label.cget("text"), long_prompt)

        # Simulate click to collapse
        btn.invoke()
        self.assertEqual(btn.cget("text"), "Prompt [+]")
        self.assertTrue(label.cget("text").endswith("..."))

        frame.destroy()


class TestScanAndGroup(unittest.TestCase):
    """Test that image scanning groups by (prompt, aspectratio) correctly."""

    @classmethod
    def setUpClass(cls):
        cls.win = _get_win()

    def test_groups_by_prompt_and_ar(self):
        """Images with same prompt+AR should land in the same group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = "a beautiful sunset over the ocean"
            _create_test_png(os.path.join(tmpdir, "img1.png"), 512, 512, prompt, "1:1")
            _create_test_png(os.path.join(tmpdir, "img2.png"), 512, 512, prompt, "1:1")
            _create_test_png(os.path.join(tmpdir, "img3.png"), 768, 512, "different prompt", "3:2")

            self.win._scan_and_group(tmpdir)
            matching = [g for g in self.win.groups if g['prompt'] == prompt]
            self.assertEqual(len(matching), 1)
            self.assertEqual(len(matching[0]['images']), 2)

    def test_single_image_group_excluded(self):
        """Groups with only 1 image should be excluded (need >=2 for pairing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_png(os.path.join(tmpdir, "lonely.png"), 512, 512, "unique prompt alone")

            self.win._scan_and_group(tmpdir)
            matching = [g for g in self.win.groups if g['prompt'] == "unique prompt alone"]
            self.assertEqual(len(matching), 0)


class TestDisplayImageScalesUp(unittest.TestCase):
    """Integration test: _display_image with a small image should produce a scaled-up CTkImage."""

    @classmethod
    def setUpClass(cls):
        cls.win = _get_win()

    def test_display_image_scales_small_image_up(self):
        """A 64x64 image displayed via _display_image should be larger than 64px."""
        import customtkinter as ctk

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tiny.png")
            Image.new("RGB", (64, 64), (255, 0, 0)).save(path)

            frame = ctk.CTkFrame(self.win)
            frame.pack(expand=True, fill="both")
            self.win._display_image(frame, path, 0, 0)

            for child in frame.winfo_children():
                if hasattr(child, 'image') and child.image is not None:
                    size = child.image.cget("size")
                    self.assertGreater(size[0], 64, "Image should have been scaled up")
                    self.assertGreater(size[1], 64, "Image should have been scaled up")
                    break
            else:
                self.fail("No image label found in frame")

            frame.destroy()

    def test_display_thumbnail_scales_small_image_up(self):
        """A 32x32 image in thumbnail view should fill the 250px box."""
        import customtkinter as ctk

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "micro.png")
            Image.new("RGB", (32, 32), (0, 255, 0)).save(path)

            frame = ctk.CTkFrame(self.win)
            frame.pack(expand=True, fill="both")
            self.win._display_thumbnail(frame, path, 0, 0)

            for child in frame.winfo_children():
                if hasattr(child, 'image') and child.image is not None:
                    size = child.image.cget("size")
                    self.assertEqual(size[0], 250, "Thumbnail should scale up to 250px")
                    self.assertEqual(size[1], 250, "Thumbnail should scale up to 250px")
                    break
            else:
                self.fail("No thumbnail label found in frame")

            frame.destroy()


if __name__ == "__main__":
    unittest.main()
