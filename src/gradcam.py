from __future__ import annotations

import numpy as np
from PIL import Image

from src.predict import IMAGE_SIZE, load_genres, load_trained_model


HEATMAP_COLORMAP = "inferno"
OVERLAY_ALPHA = 0.45


def compute_heatmap(image: Image.Image, genre_index: int) -> np.ndarray | None:
    """Return a Grad-CAM map in [0, 1] for one genre, or None without a model."""
    model = load_trained_model()
    if model is None:
        return None

    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    resized_image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    batch = tf.constant(preprocess_input(np.expand_dims(image_array, axis=0)))

    base_model = next(
        layer for layer in model.layers if isinstance(layer, tf.keras.Model)
    )
    head_layers = model.layers[model.layers.index(base_model) + 1 :]

    with tf.GradientTape() as tape:
        conv_maps = base_model(batch, training=False)
        tape.watch(conv_maps)
        outputs = conv_maps
        for layer in head_layers:
            outputs = layer(outputs, training=False)
        score = outputs[:, genre_index]

    gradients = tape.gradient(score, conv_maps)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(conv_maps * weights, axis=-1))[0].numpy()

    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def overlay_heatmap(
    image: Image.Image, heatmap: np.ndarray, alpha: float = OVERLAY_ALPHA
) -> Image.Image:
    """Blend the Grad-CAM map over the poster."""
    from matplotlib import colormaps

    poster = image.convert("RGB")
    heat_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        poster.size, Image.Resampling.BILINEAR
    )
    colored = colormaps[HEATMAP_COLORMAP](np.asarray(heat_image) / 255.0)
    heat_rgb = Image.fromarray(np.uint8(colored[..., :3] * 255))
    return Image.blend(poster, heat_rgb, alpha)


def gradcam_overlay(
    image: Image.Image, genre: str, alpha: float = OVERLAY_ALPHA
) -> Image.Image | None:
    """Return the poster with its attention zones for one genre, or None without a model."""
    genres = load_genres()
    if genre not in genres:
        raise ValueError(f"unknown genre: {genre}")

    heatmap = compute_heatmap(image, genres.index(genre))
    if heatmap is None:
        return None
    return overlay_heatmap(image, heatmap, alpha)
