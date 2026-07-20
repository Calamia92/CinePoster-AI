from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.predict import GENRES


DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "processed" / "annotations.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "genre_classifier.keras"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multilabel poster genre classifier."
    )
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS,
        type=Path,
        help="CSV produced by scripts/prepare_dataset.py.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        type=Path,
        help="Where the trained Keras model will be saved.",
    )
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    parser.add_argument("--validation-split", default=0.2, type=float)
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        type=Path,
        help="Directory where training metrics and plots will be saved.",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze the last MobileNetV2 layers for a second short training pass.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_annotations(path: Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing annotations file: {path}. Run scripts/prepare_dataset.py first."
        )

    annotations = pd.read_csv(path)
    required_columns = {"image_path", *GENRES}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

    annotations["image_path"] = annotations["image_path"].map(
        lambda value: str(resolve_path(Path(value)))
    )
    missing_images = [
        image_path
        for image_path in annotations["image_path"]
        if not Path(image_path).exists()
    ]
    if missing_images:
        sample = "\n".join(missing_images[:5])
        raise FileNotFoundError(f"Missing image files, first examples:\n{sample}")

    return annotations


def load_image(path: tf.Tensor, labels: tf.Tensor, image_size: int):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, labels


def build_dataset(
    annotations: pd.DataFrame,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = annotations["image_path"].to_numpy()
    labels = annotations[GENRES].astype("float32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(annotations), seed=42)
    dataset = dataset.map(
        lambda path, labels: load_image(path, labels, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_annotations(
    annotations: pd.DataFrame,
    validation_split: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_split < 0.5:
        raise ValueError("--validation-split must be between 0 and 0.5")

    shuffled = annotations.sample(frac=1, random_state=42).reset_index(drop=True)
    validation_size = max(1, int(len(shuffled) * validation_split))
    if len(shuffled) - validation_size < 1:
        raise ValueError("Dataset needs at least two rows to create train/validation sets")

    return shuffled.iloc[validation_size:], shuffled.iloc[:validation_size]


def build_model(image_size: int, genre_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(genre_count, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )
    return model


def fine_tune_model(model: tf.keras.Model) -> None:
    base_model = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2")
    )
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )


def merge_histories(histories: list[tf.keras.callbacks.History]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    global_epoch = 1

    for phase_index, history in enumerate(histories, start=1):
        phase_name = "fine_tune" if phase_index > 1 else "transfer_learning"
        epoch_count = len(next(iter(history.history.values()), []))
        for epoch_index in range(epoch_count):
            row: dict[str, float | int | str] = {
                "epoch": global_epoch,
                "phase": phase_name,
            }
            for metric_name, values in history.history.items():
                row[metric_name] = float(values[epoch_index])
            rows.append(row)
            global_epoch += 1

    return pd.DataFrame(rows)


def save_training_report(history: pd.DataFrame, report_dir: Path) -> None:
    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "training_history.csv"
    json_path = report_dir / "training_summary.json"
    plot_path = report_dir / "training_curves.png"

    history.to_csv(csv_path, index=False)

    final_metrics = history.iloc[-1].to_dict()
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(final_metrics, json_file, indent=2)
        json_file.write("\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_groups = [
        ("loss", "val_loss", "Binary crossentropy"),
        ("binary_accuracy", "val_binary_accuracy", "Binary accuracy"),
        ("auc", "val_auc", "Multi-label AUC"),
    ]
    available_groups = [
        group
        for group in metric_groups
        if group[0] in history.columns and group[1] in history.columns
    ]
    if not available_groups:
        print("No train/validation metric pairs available for plotting.")
        return

    fig, axes = plt.subplots(
        1,
        len(available_groups),
        figsize=(5 * len(available_groups), 4),
        constrained_layout=True,
    )
    if len(available_groups) == 1:
        axes = [axes]

    for axis, (train_metric, val_metric, title) in zip(axes, available_groups):
        axis.plot(history["epoch"], history[train_metric], marker="o", label="train")
        axis.plot(history["epoch"], history[val_metric], marker="o", label="validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"Saved training history to {csv_path}")
    print(f"Saved training summary to {json_path}")
    print(f"Saved training curves to {plot_path}")


def train(args: argparse.Namespace) -> Path:
    annotations = read_annotations(args.annotations)
    train_rows, validation_rows = split_annotations(
        annotations,
        validation_split=args.validation_split,
    )

    train_dataset = build_dataset(
        train_rows,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_dataset = build_dataset(
        validation_rows,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(args.image_size, len(GENRES))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    histories = []
    histories.append(
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
        )
    )

    if args.fine_tune:
        fine_tune_model(model)
        histories.append(
            model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=max(2, args.epochs // 2),
                callbacks=callbacks,
            )
        )

    history = merge_histories(histories)
    save_training_report(history, args.report_dir)

    model_path = resolve_path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    genres_path = model_path.parent / "genres.json"
    with genres_path.open("w", encoding="utf-8") as json_file:
        json.dump(GENRES, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")

    processed_genres_path = resolve_path(args.annotations).parent / "genres.json"
    if processed_genres_path.exists() and processed_genres_path != genres_path:
        shutil.copyfile(processed_genres_path, genres_path)

    return model_path


def main() -> None:
    args = parse_args()
    model_path = train(args)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
