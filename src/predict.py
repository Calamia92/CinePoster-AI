from __future__ import annotations

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


def predict_genres(image: Image.Image) -> dict[str, float]:
    """Return placeholder genre probabilities until the trained model is wired."""
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
