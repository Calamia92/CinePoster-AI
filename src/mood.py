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


def _clamp_confidence(value: float) -> float:
    return max(0.5, min(0.92, value))


def _extract_features(image: Image.Image) -> MoodFeatures:
    thumbnail = image.convert("RGB").resize((128, 128))
    grayscale = thumbnail.convert("L")
    grayscale_stat = ImageStat.Stat(grayscale)
    brightness = grayscale_stat.mean[0] / 255
    contrast = grayscale_stat.stddev[0] / 255

    histogram = grayscale.histogram()
    dark_pixels = sum(histogram[:75])
    dark_ratio = dark_pixels / sum(histogram)

    rgb_mean = ImageStat.Stat(thumbnail).mean
    red, green, blue = [channel / 255 for channel in rgb_mean]

    saturation = ImageStat.Stat(thumbnail.convert("HSV")).mean[1] / 255

    return MoodFeatures(
        brightness=brightness,
        saturation=saturation,
        contrast=contrast,
        red=red,
        green=green,
        blue=blue,
        dark_ratio=dark_ratio,
    )


def estimate_mood(image: Image.Image) -> MoodPrediction:
    """Estimate poster mood from color, brightness, saturation and contrast."""
    features = _extract_features(image)

    if features.brightness < 0.28 or features.dark_ratio > 0.58:
        return MoodPrediction(
            label="Sombre / mysterieux",
            confidence=_clamp_confidence(0.62 + (0.28 - features.brightness)),
            explanation=(
                "L'affiche est dominee par des zones sombres, avec une luminosite "
                "moyenne faible."
            ),
        )

    if (
        features.brightness > 0.62
        and features.saturation > 0.34
        and features.contrast < 0.22
    ):
        return MoodPrediction(
            label="Familiale / humoristique",
            confidence=_clamp_confidence(0.55 + features.saturation * 0.25),
            explanation=(
                "L'image est lumineuse, coloree et peu agressive, ce qui suggere "
                "une ambiance accessible."
            ),
        )

    if (
        features.blue > features.red + 0.12
        and features.blue > features.green + 0.08
        and features.saturation > 0.24
    ):
        return MoodPrediction(
            label="Froid / science-fiction",
            confidence=_clamp_confidence(0.56 + (features.blue - features.red) * 0.35),
            explanation=(
                "Les tons froids dominent l'affiche, en particulier le bleu, ce qui "
                "evoque une ambiance technologique ou distante."
            ),
        )

    if (
        features.red > features.blue + 0.12
        and features.red > features.green + 0.06
        and features.saturation > 0.22
    ):
        return MoodPrediction(
            label="Romantique / dramatique",
            confidence=_clamp_confidence(0.55 + (features.red - features.blue) * 0.3),
            explanation=(
                "Les tons chauds dominent l'image, ce qui evoque une ambiance "
                "emotionnelle ou dramatique."
            ),
        )

    if features.contrast > 0.24 and features.saturation > 0.22:
        return MoodPrediction(
            label="Epique / intense",
            confidence=_clamp_confidence(0.54 + features.contrast * 0.55),
            explanation=(
                "Le contraste visuel est marque, avec des couleurs assez presentes, "
                "ce qui donne une impression d'intensite."
            ),
        )

    return MoodPrediction(
        label="Neutre",
        confidence=0.5,
        explanation=(
            "Aucun signal de luminosite, contraste ou dominante couleur ne ressort "
            "nettement."
        ),
    )
