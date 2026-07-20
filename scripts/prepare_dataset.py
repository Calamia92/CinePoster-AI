from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.predict import GENRES


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_IMAGE_SIZE = 224
GENRE_ALIASES = {
    "sci-fi": "Science-fiction",
    "science fiction": "Science-fiction",
    "science-fiction": "Science-fiction",
    "comedy": "Comedie",
    "comedie": "Comedie",
    "horror": "Horreur",
    "romance": "Romance",
    "adventure": "Aventure",
    "action": "Action",
    "thriller": "Thriller",
    "drama": "Drame",
    "drame": "Drame",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare poster annotations for multilabel genre training."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV with image_path and genres columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where processed annotations will be written.",
    )
    parser.add_argument(
        "--genre-separator",
        default="|",
        help="Separator used inside the genres column.",
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Fail if an image referenced by the CSV does not exist.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Resize RGB poster images into output-dir/images.",
    )
    parser.add_argument(
        "--image-size",
        default=DEFAULT_IMAGE_SIZE,
        type=int,
        help="Square size used when --copy-images is enabled.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_genre(raw_genre: str) -> str:
    key = raw_genre.strip().lower()
    if not key:
        raise ValueError("empty genre")
    return GENRE_ALIASES.get(key, raw_genre.strip())


def parse_genres(raw_genres: str, separator: str) -> list[str]:
    genres = [normalize_genre(value) for value in raw_genres.split(separator)]
    unknown = sorted(set(genres) - set(GENRES))
    if unknown:
        raise ValueError(f"unknown genres: {', '.join(unknown)}")
    return genres


def processed_image_path(source_path: Path, output_dir: Path) -> Path:
    return output_dir / "images" / f"{source_path.stem}.jpg"


def resize_image(source_path: Path, target_path: Path, image_size: int) -> None:
    from PIL import Image

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.convert("RGB").resize((image_size, image_size)).save(
            target_path, format="JPEG", quality=95
        )


def read_source_rows(input_path: Path) -> Iterable[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"image_path", "genres"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"missing columns: {', '.join(sorted(missing_columns))}")
        yield from reader


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    genre_separator: str,
    validate_images: bool,
    copy_images: bool,
    image_size: int,
) -> int:
    input_path = resolve_path(input_path)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = output_dir / "annotations.csv"
    genres_path = output_dir / "genres.json"
    prepared_rows: list[dict[str, str | int]] = []

    for line_number, row in enumerate(read_source_rows(input_path), start=2):
        raw_image_path = (row.get("image_path") or "").strip()
        raw_genres = (row.get("genres") or "").strip()

        if not raw_image_path:
            raise ValueError(f"line {line_number}: image_path is required")
        if not raw_genres:
            raise ValueError(f"line {line_number}: genres is required")

        source_image_path = resolve_path(Path(raw_image_path))
        if validate_images or copy_images:
            if not source_image_path.exists():
                raise FileNotFoundError(f"line {line_number}: missing image {source_image_path}")

        genres = parse_genres(raw_genres, genre_separator)
        output_image_path = source_image_path

        if copy_images:
            output_image_path = processed_image_path(source_image_path, output_dir)
            resize_image(source_image_path, output_image_path, image_size)

        relative_image_path = output_image_path.relative_to(REPO_ROOT).as_posix()
        prepared_row: dict[str, str | int] = {
            "image_path": relative_image_path,
            "genres": genre_separator.join(genres),
        }
        for genre in GENRES:
            prepared_row[genre] = int(genre in genres)
        prepared_rows.append(prepared_row)

    if not prepared_rows:
        raise ValueError("input CSV contains no data rows")

    with annotations_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["image_path", "genres", *GENRES]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prepared_rows)

    with genres_path.open("w", encoding="utf-8") as json_file:
        json.dump(GENRES, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")

    return len(prepared_rows)


def main() -> None:
    args = parse_args()
    row_count = prepare_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        genre_separator=args.genre_separator,
        validate_images=args.validate_images,
        copy_images=args.copy_images,
        image_size=args.image_size,
    )
    print(f"Prepared {row_count} rows in {resolve_path(args.output_dir)}")


if __name__ == "__main__":
    main()
