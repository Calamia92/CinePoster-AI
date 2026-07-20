import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from src import gradcam
from src.predict import DEFAULT_MODEL_PATH, GENRES


def tensorflow_available():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def make_poster():
    image = Image.new("RGB", (300, 450), (12, 12, 18))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 240, 220), fill=(200, 170, 140))
    draw.rectangle((0, 300, 300, 450), fill=(160, 30, 20))
    return image


class GradcamOverlayTests(unittest.TestCase):
    def test_returns_none_without_trained_model(self):
        with mock.patch.object(gradcam, "load_trained_model", return_value=None):
            self.assertIsNone(gradcam.gradcam_overlay(make_poster(), GENRES[0]))

    def test_unknown_genre_raises(self):
        with self.assertRaises(ValueError):
            gradcam.gradcam_overlay(make_poster(), "Documentaire")

    def test_returns_none_for_flat_heatmap(self):
        flat_map = np.zeros((7, 7), dtype="float32")
        with mock.patch.object(gradcam, "compute_heatmap", return_value=flat_map):
            self.assertIsNone(gradcam.gradcam_overlay(make_poster(), GENRES[0]))

    @unittest.skipUnless(
        DEFAULT_MODEL_PATH.exists() and tensorflow_available(),
        "requires the trained model and tensorflow",
    )
    def test_overlay_for_some_genre_matches_poster_size(self):
        poster = make_poster()
        overlay = next(
            (
                result
                for result in (
                    gradcam.gradcam_overlay(poster, genre) for genre in GENRES
                )
                if result is not None
            ),
            None,
        )

        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.size, (300, 450))
        self.assertEqual(overlay.mode, "RGB")

    @unittest.skipUnless(
        DEFAULT_MODEL_PATH.exists() and tensorflow_available(),
        "requires the trained model and tensorflow",
    )
    def test_heatmap_values_are_normalized(self):
        heatmap = gradcam.compute_heatmap(make_poster(), 0)

        self.assertIsNotNone(heatmap)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
