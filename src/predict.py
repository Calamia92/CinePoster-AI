from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


GENRES = [
    "Action",
    "Science-fiction",
    "Thriller",
    "Drame",
    "Comedie",
    "Romance",
    "Horreur",
    "Aventure",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "genre_classifier.keras"
DEFAULT_GENRES_PATH = REPO_ROOT / "models" / "genres.json"
IMAGE_SIZE = (224, 224)


@dataclass(frozen=True)
class GenreAnalysis:
    scores: dict[str, float]
    used_trained_model: bool
    load_error: str | None = None


@lru_cache(maxsize=1)
def load_trained_model():
    if not DEFAULT_MODEL_PATH.exists():
        return None

    import tensorflow as tf

    return tf.keras.models.load_model(DEFAULT_MODEL_PATH)


@lru_cache(maxsize=1)
def load_genres() -> list[str]:
    if DEFAULT_GENRES_PATH.exists():
        with DEFAULT_GENRES_PATH.open(encoding="utf-8") as json_file:
            return json.load(json_file)
    return GENRES


def predict_with_trained_model(image: Image.Image) -> dict[str, float] | None:
    model = load_trained_model()
    if model is None:
        return None

    import numpy as np
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    resized_image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    batch = preprocess_input(np.expand_dims(image_array, axis=0))
    probabilities = model.predict(batch, verbose=0)[0]

    return {
        genre: float(probability)
        for genre, probability in zip(load_genres(), probabilities, strict=True)
    }


def placeholder_genres(image: Image.Image) -> dict[str, float]:
    width, height = image.size
    aspect_signal = min(width, height) / max(width, height)

    return {
        "Action": 0.64,
        "Science-fiction": 0.47,
        "Thriller": 0.41,
        "Drame": 0.35 + (0.08 * aspect_signal),
        "Comedie": 0.24,
        "Romance": 0.22,
        "Horreur": 0.18,
        "Aventure": 0.39,
    }


def analyze_genres(image: Image.Image) -> GenreAnalysis:
    """Predict genres with the trained model, or fall back to MVP placeholders."""
    try:
        scores = predict_with_trained_model(image)
    except Exception as error:
        return GenreAnalysis(
            scores=placeholder_genres(image),
            used_trained_model=False,
            load_error=str(error),
        )

    if scores is None:
        return GenreAnalysis(
            scores=placeholder_genres(image),
            used_trained_model=False,
        )

    return GenreAnalysis(scores=scores, used_trained_model=True)


def predict_genres(image: Image.Image) -> dict[str, float]:
    """Return genre probabilities from the trained model, or MVP placeholders."""
    return analyze_genres(image).scores
