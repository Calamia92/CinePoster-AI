# Dataset preparation

Issue #2 prepares the poster dataset for multilabel genre training.

## Recommended dataset

Use a poster dataset that provides one image per movie and one or more genres per
movie. Good sources for this project are:

- Kaggle `raman77768/movie-classifier`, published as Movie Posters with
  multi-label genres and a CC0 public domain license.
- TMDB poster exports combined with TMDB genre metadata.

Do not commit downloaded poster images to Git. Keep raw files under
`data/raw/`, which is ignored by `.gitignore`.

## Kaggle workflow

Authenticate Kaggle locally first:

```bash
kaggle auth login
```

Or set the token for the current shell:

```bash
$env:KAGGLE_API_TOKEN = "your-token"
```

Then download and unzip the selected dataset:

```bash
kaggle datasets download -d raman77768/movie-classifier -p data/raw/kaggle/movie-classifier --unzip
```

Convert the Kaggle files to the project raw format:

```bash
python scripts/convert_kaggle_movie_posters.py --dataset-dir data/raw/kaggle/movie-classifier
```

Then generate the multi-label annotations:

```bash
python scripts/prepare_dataset.py --input data/raw/posters.csv --validate-images
```

The Kaggle conversion keeps only the genres supported by the MVP:

`Action`, `Science-fiction`, `Thriller`, `Drame`, `Comedie`, `Romance`,
`Horreur`, `Aventure`.

## Source CSV format

Create a CSV with these columns:

```csv
image_path,genres
data/raw/posters/inception.jpg,Action|Science-fiction|Thriller
data/raw/posters/little_miss_sunshine.jpg,Comedie|Drame
```

Rules:

- `image_path` can be absolute or relative to the repository root.
- `genres` contains one or more genres separated by `|`.
- Empty rows and rows without genres are rejected.
- Genres are normalized to the project genre list in `src/predict.py`.

## Prepare the annotations

Run:

```bash
python scripts/prepare_dataset.py --input data/raw/posters.csv
```

This writes:

- `data/processed/annotations.csv`: image path plus one binary column per genre.
- `data/processed/genres.json`: ordered genre list used by the model output.

To also validate that every image exists:

```bash
python scripts/prepare_dataset.py --input data/raw/posters.csv --validate-images
```

To create resized RGB copies for training:

```bash
python scripts/prepare_dataset.py --input data/raw/posters.csv --copy-images
```

Resized images are written to `data/processed/images/`.
