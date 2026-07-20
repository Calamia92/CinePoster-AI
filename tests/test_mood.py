import unittest

from PIL import Image

from src.mood import MOOD_CLASSES, estimate_mood


class MoodPredictionTests(unittest.TestCase):
    def test_supported_classes_are_explicit(self):
        self.assertIn("Sombre / mysterieux", MOOD_CLASSES)
        self.assertIn("Romantique / dramatique", MOOD_CLASSES)
        self.assertIn("Familiale / humoristique", MOOD_CLASSES)
        self.assertIn("Froid / science-fiction", MOOD_CLASSES)
        self.assertIn("Epique / intense", MOOD_CLASSES)
        self.assertIn("Neutre", MOOD_CLASSES)

    def test_dark_poster_returns_mysterious_mood(self):
        prediction = estimate_mood(Image.new("RGB", (64, 64), (18, 18, 20)))

        self.assertEqual(prediction.label, "Sombre / mysterieux")
        self.assertGreaterEqual(prediction.confidence, 0.5)
        self.assertTrue(prediction.explanation)

    def test_warm_poster_returns_romantic_dramatic_mood(self):
        prediction = estimate_mood(Image.new("RGB", (64, 64), (220, 80, 55)))

        self.assertEqual(prediction.label, "Romantique / dramatique")
        self.assertGreaterEqual(prediction.confidence, 0.5)
        self.assertTrue(prediction.explanation)

    def test_bright_saturated_poster_returns_family_mood(self):
        prediction = estimate_mood(Image.new("RGB", (64, 64), (235, 215, 80)))

        self.assertEqual(prediction.label, "Familiale / humoristique")
        self.assertGreaterEqual(prediction.confidence, 0.5)
        self.assertTrue(prediction.explanation)

    def test_blue_poster_returns_science_fiction_mood(self):
        prediction = estimate_mood(Image.new("RGB", (64, 64), (65, 120, 220)))

        self.assertEqual(prediction.label, "Froid / science-fiction")
        self.assertGreaterEqual(prediction.confidence, 0.5)
        self.assertTrue(prediction.explanation)

    def test_neutral_poster_returns_neutral_mood(self):
        prediction = estimate_mood(Image.new("RGB", (64, 64), (135, 135, 135)))

        self.assertEqual(prediction.label, "Neutre")
        self.assertEqual(prediction.confidence, 0.5)
        self.assertTrue(prediction.explanation)


if __name__ == "__main__":
    unittest.main()
