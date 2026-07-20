import unittest

from PIL import Image, ImageDraw

from src.mood import MOOD_CLASSES, estimate_mood


def make_split_poster(left_color, right_color, size=(96, 144)):
    image = Image.new("RGB", size, left_color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((size[0] // 2, 0, size[0], size[1]), fill=right_color)
    return image


def make_center_spot_poster(background, center, size=(96, 144)):
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    margin_x = size[0] // 4
    margin_y = size[1] // 4
    draw.rectangle(
        (margin_x, margin_y, size[0] - margin_x, size[1] - margin_y),
        fill=center,
    )
    return image


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

    def test_dark_poster_with_bright_face_stays_mysterious(self):
        prediction = estimate_mood(
            make_center_spot_poster((10, 12, 18), (190, 180, 160))
        )

        self.assertEqual(prediction.label, "Sombre / mysterieux")
        self.assertGreater(prediction.confidence, 0.6)
        self.assertIn("zones sombres", prediction.explanation)

    def test_high_contrast_saturated_poster_returns_epic_mood(self):
        prediction = estimate_mood(make_split_poster((5, 5, 10), (245, 70, 25)))

        self.assertEqual(prediction.label, "Epique / intense")
        self.assertGreater(prediction.confidence, 0.6)
        self.assertIn("contraste", prediction.explanation)

    def test_explanation_contains_quantified_visual_cues(self):
        prediction = estimate_mood(Image.new("RGB", (64, 64), (65, 120, 220)))

        self.assertIn("luminosite", prediction.explanation)
        self.assertIn("saturation", prediction.explanation)
        self.assertIn("contraste", prediction.explanation)


if __name__ == "__main__":
    unittest.main()
