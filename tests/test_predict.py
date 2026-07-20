import unittest
from unittest import mock

from PIL import Image

from src import predict
from src.predict import GENRES, analyze_genres, predict_genres


def make_poster(size=(64, 96)):
    return Image.new("RGB", size, (120, 90, 60))


class AnalyzeGenresTests(unittest.TestCase):
    def test_placeholders_when_no_trained_model(self):
        with mock.patch.object(predict, "predict_with_trained_model", return_value=None):
            analysis = analyze_genres(make_poster())

        self.assertFalse(analysis.used_trained_model)
        self.assertIsNone(analysis.load_error)
        self.assertEqual(set(analysis.scores), set(GENRES))

    def test_fallback_and_error_when_model_loading_fails(self):
        with mock.patch.object(
            predict,
            "predict_with_trained_model",
            side_effect=OSError("fichier modele corrompu"),
        ):
            analysis = analyze_genres(make_poster())

        self.assertFalse(analysis.used_trained_model)
        self.assertIn("corrompu", analysis.load_error)
        self.assertEqual(set(analysis.scores), set(GENRES))

    def test_trained_model_scores_are_returned(self):
        fake_scores = {genre: 0.5 for genre in GENRES}
        with mock.patch.object(
            predict, "predict_with_trained_model", return_value=fake_scores
        ):
            analysis = analyze_genres(make_poster())

        self.assertTrue(analysis.used_trained_model)
        self.assertIsNone(analysis.load_error)
        self.assertEqual(analysis.scores, fake_scores)


class PredictGenresTests(unittest.TestCase):
    def test_returns_one_probability_per_genre(self):
        with mock.patch.object(predict, "predict_with_trained_model", return_value=None):
            scores = predict_genres(make_poster())

        self.assertEqual(set(scores), set(GENRES))
        for probability in scores.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
