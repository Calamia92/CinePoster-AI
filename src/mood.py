from dataclasses import dataclass

from PIL import Image, ImageStat


MOOD_CLASSES = (
    "Sombre / mysterieux",
    "Romantique / dramatique",
    "Familiale / humoristique",
    "Froid / science-fiction",
    "Epique / intense",
    "Neutre",
)


@dataclass(frozen=True)
class MoodPrediction:
    label: str
    confidence: float
    explanation: str


@dataclass(frozen=True)
class MoodFeatures:
    brightness: float
    saturation: float
    contrast: float
    red: float
    green: float
    blue: float
    dark_ratio: float
    bright_ratio: float
    warm_bias: float
    cool_bias: float


def _clamp_confidence(value: float) -> float:
    return max(0.5, min(0.92, value))


def _score_to_confidence(score: float) -> float:
    return _clamp_confidence(0.5 + score * 0.42)


def _extract_features(image: Image.Image) -> MoodFeatures:
    thumbnail = image.convert("RGB").resize((128, 128))
    grayscale = thumbnail.convert("L")
    grayscale_stat = ImageStat.Stat(grayscale)
    brightness = grayscale_stat.mean[0] / 255
    contrast = grayscale_stat.stddev[0] / 255

    histogram = grayscale.histogram()
    dark_pixels = sum(histogram[:75])
    bright_pixels = sum(histogram[190:])
    dark_ratio = dark_pixels / sum(histogram)
    bright_ratio = bright_pixels / sum(histogram)

    rgb_mean = ImageStat.Stat(thumbnail).mean
    red, green, blue = [channel / 255 for channel in rgb_mean]

    saturation = ImageStat.Stat(thumbnail.convert("HSV")).mean[1] / 255
    warm_bias = red - max(green, blue)
    cool_bias = blue - max(red, green)

    return MoodFeatures(
        brightness=brightness,
        saturation=saturation,
        contrast=contrast,
        red=red,
        green=green,
        blue=blue,
        dark_ratio=dark_ratio,
        bright_ratio=bright_ratio,
        warm_bias=warm_bias,
        cool_bias=cool_bias,
    )


def _positive(value: float) -> float:
    return max(0.0, value)


def _score_moods(features: MoodFeatures) -> dict[str, float]:
    return {
        "Sombre / mysterieux": max(
            _positive(0.42 - features.brightness) / 0.42,
            _positive(features.dark_ratio - 0.36) / 0.64,
        ),
        "Familiale / humoristique": (
            _positive(features.brightness - 0.56) / 0.44 * 0.45
            + _positive(features.saturation - 0.22) / 0.78 * 0.35
            + _positive(0.24 - features.contrast) / 0.24 * 0.20
        ),
        "Froid / science-fiction": (
            _positive(features.cool_bias - 0.05) / 0.55 * 0.70
            + _positive(features.saturation - 0.20) / 0.80 * 0.30
        ),
        "Romantique / dramatique": (
            _positive(features.warm_bias - 0.05) / 0.55 * 0.55
            + _positive(features.saturation - 0.20) / 0.80 * 0.25
            + _positive(0.72 - features.brightness) / 0.72 * 0.10
            - _positive(features.contrast - 0.28) / 0.50 * 0.20
            - _positive(features.dark_ratio - 0.30) * 0.75
        ),
        "Epique / intense": (
            _positive(features.contrast - 0.18) / 0.50 * 0.65
            + _positive(features.saturation - 0.18) / 0.82 * 0.25
            + min(features.dark_ratio, features.bright_ratio) * 0.70
            + features.dark_ratio * features.saturation * 0.70
        ),
    }


def _format_explanation(label: str, features: MoodFeatures) -> str:
    visual_cues = (
        f"luminosite {features.brightness:.0%}, "
        f"saturation {features.saturation:.0%}, "
        f"contraste {features.contrast:.0%}, "
        f"zones sombres {features.dark_ratio:.0%}"
    )

    explanations = {
        "Sombre / mysterieux": (
            "L'affiche contient une forte proportion de zones sombres et une "
            f"luminosite reduite ({visual_cues})."
        ),
        "Familiale / humoristique": (
            "L'image est lumineuse, coloree et peu agressive, ce qui suggere "
            f"une ambiance accessible ({visual_cues})."
        ),
        "Froid / science-fiction": (
            "Les tons froids dominent l'affiche, surtout le bleu, ce qui evoque "
            f"une ambiance technologique ou distante ({visual_cues})."
        ),
        "Romantique / dramatique": (
            "Les tons chauds dominent l'image, ce qui evoque une ambiance "
            f"emotionnelle ou dramatique ({visual_cues})."
        ),
        "Epique / intense": (
            "Le contraste visuel est marque, avec des couleurs assez presentes, "
            f"ce qui donne une impression d'intensite ({visual_cues})."
        ),
    }
    return explanations.get(
        label,
        "Aucun signal de luminosite, contraste ou dominante couleur ne ressort "
        f"nettement ({visual_cues}).",
    )


def estimate_mood(image: Image.Image) -> MoodPrediction:
    """Estimate poster mood from color, brightness, saturation and contrast."""
    features = _extract_features(image)
    scores = _score_moods(features)
    label, score = max(scores.items(), key=lambda item: item[1])

    if score >= 0.22:
        return MoodPrediction(
            label=label,
            confidence=_score_to_confidence(score),
            explanation=_format_explanation(label, features),
        )

    return MoodPrediction(
        label="Neutre",
        confidence=0.5,
        explanation=_format_explanation("Neutre", features),
    )
