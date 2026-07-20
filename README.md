# CinePoster Insight

Analyse multimodale d'affiches de film.

## Objectif MVP

- Importer une affiche dans une WebApp Streamlit.
- Predire plusieurs genres avec une sortie multi-label.
- Estimer une ambiance visuelle.
- Afficher les probabilites sous forme de graphique.

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Preparation du dataset

Le format attendu pour l'entrainement multi-label est documente dans
`docs/dataset.md`.

```bash
python scripts/prepare_dataset.py --input data/raw/posters.csv
```

Pour un dataset Kaggle reel:

```bash
kaggle datasets download -d raman77768/movie-classifier -p data/raw/kaggle/movie-classifier --unzip
python scripts/convert_kaggle_movie_posters.py --dataset-dir data/raw/kaggle/movie-classifier
python scripts/prepare_dataset.py --input data/raw/posters.csv --validate-images
```

## Structure

```text
app.py              # Interface Streamlit
src/predict.py      # Prediction des genres
src/mood.py         # Estimation de l'ambiance
docs/issues.md      # Issues de depart
docs/dataset.md     # Format et preparation du dataset
```

## Roadmap

1. Remplacer les predictions placeholder par un modele CNN pre-entraine.
2. Ajouter un vrai dataset poster + labels multi-genres.
3. Ajouter Grad-CAM pour expliquer les predictions.
