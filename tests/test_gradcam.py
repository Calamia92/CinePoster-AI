import unittest
from unittest import mock

from PIL import Image

from src import gradcam
from src.predict import DEFAULT_MODEL_PATH, GENRES


def tensorflow_available():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def make_poster(size=(300, 450)):
    return Image.new("RGB", size, (150, 60, 40))


class GradcamOverlayTests(unittest.TestCase):
    def test_returns_none_without_trained_model(self):
        with mock.patch.object(gradcam, "load_trained_model", return_value=None):
            self.assertIsNone(gradcam.gradcam_overlay(make_poster(), GENRES[0]))

    def test_unknown_genre_raises(self):
        with self.assertRaises(ValueError):
            gradcam.gradcam_overlay(make_poster(), "Documentaire")

    @unittest.skipUnless(
        DEFAULT_MODEL_PATH.exists() and tensorflow_available(),
        "requires the trained model and tensorflow",
    )
    def test_overlay_matches_poster_size(self):
        overlay = gradcam.gradcam_overlay(make_poster(), GENRES[0])

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
