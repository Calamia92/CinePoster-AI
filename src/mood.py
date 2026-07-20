from dataclasses import dataclass

from PIL import Image, ImageStat


@dataclass(frozen=True)
class MoodPrediction:
    label: str
    confidence: float
    explanation: str


def estimate_mood(image: Image.Image) -> MoodPrediction:
    """Estimate poster mood with a small color heuristic for the MVP."""
    thumbnail = image.resize((128, 128))
    grayscale = thumbnail.convert("L")
    brightness = ImageStat.Stat(grayscale).mean[0] / 255

    rgb_mean = ImageStat.Stat(thumbnail).mean
    red, green, blue = [channel / 255 for channel in rgb_mean]
    saturation_hint = max(rgb_mean) - min(rgb_mean)

    if brightness < 0.34:
        return MoodPrediction(
            label="Sombre / mysterieux",
            confidence=0.72,
            explanation="L'affiche est globalement sombre, avec une faible luminosite moyenne.",
        )

    if red > blue + 0.08 and red > green + 0.04:
        return MoodPrediction(
            label="Romantique / dramatique",
            confidence=0.61,
            explanation="Les tons chauds dominent l'image, ce qui evoque une ambiance emotionnelle.",
        )

    if brightness > 0.68 and saturation_hint > 35:
        return MoodPrediction(
            label="Familiale / humoristique",
            confidence=0.58,
            explanation="L'image est lumineuse et contrastee, avec des couleurs relativement vives.",
        )

    return MoodPrediction(
        label="Epique / neutre",
        confidence=0.52,
        explanation="Aucun signal couleur tres marque ne domine l'affiche.",
    )
