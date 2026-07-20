from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from prepare_dataset import REPO_ROOT, resolve_path
from src.predict import GENRES


SOURCE_TO_PROJECT_GENRES = {
    "action": "Action",
    "adventure": "Aventure",
    "comedy": "Comedie",
    "drama": "Drame",
    "horror": "Horreur",
    "romance": "Romance",
    "sci-fi": "Science-fiction",
    "sci fi": "Science-fiction",
    "science fiction": "Science-fiction",
    "science-fiction": "Science-fiction",
    "thriller": "Thriller",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Kaggle movie poster dataset into data/raw/posters.csv."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Unzipped Kaggle dataset directory.",
    )
    parser.add_argument(
        "--output",
        default=REPO_ROOT / "data" / "raw" / "posters.csv",
        type=Path,
        help="Output CSV with image_path and genres columns.",
    )
    return parser.parse_args()


def find_metadata_csv(dataset_dir: Path) -> Path:
    csv_paths = sorted(dataset_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no CSV file found under {dataset_dir}")

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or [])
            lower_fieldnames = {name.lower() for name in fieldnames}
            if {"id", "genre"} <= lower_fieldnames or any(
                genre in fieldnames for genre in GENRES
            ):
                return csv_path

    raise ValueError("no metadata CSV with poster ids and genres was found")


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def index_images(dataset_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for image_path in dataset_dir.rglob("*"):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images[normalize_key(image_path.stem)] = image_path
            images[normalize_key(image_path.name)] = image_path
    return images


def get_value(row: dict[str, str], name: str) -> str:
    for key, value in row.items():
        if key.lower() == name.lower():
            return value
    return ""


def parse_genre_text(raw_genres: str) -> list[str]:
    source_genres = re.split(r"[|,;/]+", raw_genres)
    mapped_genres = []
    for source_genre in source_genres:
        normalized = source_genre.strip().lower()
        if normalized in SOURCE_TO_PROJECT_GENRES:
            mapped_genres.append(SOURCE_TO_PROJECT_GENRES[normalized])
    return sorted(set(mapped_genres), key=GENRES.index)


def parse_genre_columns(row: dict[str, str]) -> list[str]:
    mapped_genres = []
    for source_genre, project_genre in SOURCE_TO_PROJECT_GENRES.items():
        for column_name, value in row.items():
            if column_name.strip().lower() == source_genre and value.strip() == "1":
                mapped_genres.append(project_genre)
    return sorted(set(mapped_genres), key=GENRES.index)


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def convert_dataset(dataset_dir: Path, output_path: Path) -> int:
    dataset_dir = resolve_path(dataset_dir)
    output_path = resolve_path(output_path)
    metadata_path = find_metadata_csv(dataset_dir)
    images = index_images(dataset_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared_rows: list[dict[str, str]] = []
    with metadata_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            poster_id = get_value(row, "Id") or get_value(row, "image_path")
            if not poster_id:
                continue

            poster_key = normalize_key(Path(poster_id).stem)
            image_path = images.get(poster_key) or images.get(normalize_key(poster_id))
            if image_path is None:
                continue

            genres = parse_genre_columns(row)
            if not genres:
                genres = parse_genre_text(get_value(row, "Genre") or get_value(row, "genres"))
            if not genres:
                continue

            prepared_rows.append(
                {
                    "image_path": relative_path(image_path),
                    "genres": "|".join(genres),
                }
            )

    if not prepared_rows:
        raise ValueError(
            f"no usable rows found in {metadata_path}; check image ids and genre columns"
        )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "genres"])
        writer.writeheader()
        writer.writerows(prepared_rows)

    return len(prepared_rows)


def main() -> None:
    args = parse_args()
    row_count = convert_dataset(args.dataset_dir, args.output)
    print(f"Wrote {row_count} rows to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
