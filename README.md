# CinePoster Insight

CinePoster Insight est une WebApp Streamlit qui analyse une affiche de film pour
estimer ses genres probables, son ambiance visuelle et les zones de l'image qui
ont influence une prediction.

## Fonctionnalites

- Import d'une affiche au format JPG, PNG ou WebP.
- Classification multi-label des genres avec MobileNetV2.
- Sortie `sigmoid` avec une probabilite par genre.
- Estimation de l'ambiance par heuristiques visuelles explicables.
- Graphiques des probabilites et des signaux visuels.
- Grad-CAM pour visualiser les zones d'attention du modele.
- Fallback de demonstration si aucun modele entraine n'est disponible.

## Lancement rapide

```bash
pip install -r requirements.txt
streamlit run app.py
```

Si le modele `models/genre_classifier.keras` est present, l'application
l'utilise automatiquement. Sinon, elle affiche des predictions placeholder pour
conserver une demo fonctionnelle.

## Deploiement Streamlit Community Cloud

Le projet est pret pour Streamlit Community Cloud. Depuis `share.streamlit.io`,
creer une nouvelle app avec ces parametres :

```text
Repository: Calamia92/CinePoster-AI
Branch: add-mood-scores-training-reports
Main file path: app.py
Python version: 3.11 ou 3.12
```

Le repo contient les fichiers requis par Community Cloud :

```text
requirements.txt
.streamlit/config.toml
models/genre_classifier.keras
models/genres.json
reports/training_curves.png
```

Il n'y a pas de secret a renseigner pour lancer l'application. Si la branche est
mergee dans `master`, choisir ensuite `master` comme branche de deploiement.

## Architecture

```text
Affiche uploadee
      |
      +-- MobileNetV2 -> Dense sigmoid -> probabilites de genres
      |
      +-- Heuristiques couleur -> ambiance + signaux visuels
      |
      +-- Grad-CAM -> zones d'attention du modele
```

## Structure du projet

```text
app.py                                # Interface Streamlit
src/predict.py                        # Chargement modele + predictions genres
src/mood.py                           # Analyse d'ambiance et signaux visuels
src/gradcam.py                        # Visualisation Grad-CAM
scripts/convert_kaggle_movie_posters.py
scripts/prepare_dataset.py
scripts/train_genre_classifier.py
tests/                                # Tests unitaires
docs/dataset.md                       # Format et preparation du dataset
```

## Dataset

Le projet utilise un dataset d'affiches avec labels multi-genres. Le format
source attendu est :

```csv
image_path,genres
data/raw/posters/inception.jpg,Action|Science-fiction|Thriller
data/raw/posters/the_notebook.jpg,Romance|Drame
```

Les genres du MVP sont :

```text
Action, Science-fiction, Thriller, Drame, Comedie, Romance, Horreur, Aventure
```

Conversion du dataset Kaggle recommande :

```bash
kaggle datasets download -d raman77768/movie-classifier -p data/raw/kaggle/movie-classifier --unzip
python scripts/convert_kaggle_movie_posters.py --dataset-dir data/raw/kaggle/movie-classifier
python scripts/prepare_dataset.py --input data/raw/posters.csv --validate-images
```

Les images brutes et les annotations preparees sont ignorees par Git pour eviter
de surcharger le depot.

## Entrainement

Une fois `data/processed/annotations.csv` genere :

```bash
python scripts/train_genre_classifier.py --epochs 8 --batch-size 16
```

Pour une demo rapide sur CPU :

```bash
python scripts/train_genre_classifier.py --epochs 3 --batch-size 32
```

Le modele est sauvegarde dans :

```text
models/genre_classifier.keras
models/genres.json
```

Le script sauvegarde aussi un rapport d'entrainement :

```text
reports/training_history.csv
reports/training_summary.json
reports/training_curves.png
```

Ces fichiers permettent d'illustrer l'evolution de la loss, de la binary
accuracy et de l'AUC sur train/validation. Sur le prototype 3 epochs utilise
pour la demo, les dernieres metriques sont environ :

```text
val_loss: 0.396
val_binary_accuracy: 82.5 %
val_auc: 76.8 %
```

Le modele actuellement partage est un prototype entraine rapidement pour la
demonstration. Ses predictions sont utiles pour illustrer le pipeline, mais ne
doivent pas etre presentees comme un systeme de production.

## Ambiance

L'ambiance n'est pas apprise par un dataset labelise. Elle est estimee avec des
signaux visuels calcules sur l'affiche :

- luminosite moyenne ;
- saturation ;
- contraste ;
- proportion de zones sombres ;
- proportion de zones claires ;
- dominante chaude ou froide.

Cette approche est volontairement explicable : l'app affiche les signaux qui ont
oriente l'ambiance choisie. L'application affiche aussi un score par ambiance,
ce qui montre les alternatives possibles au lieu de masquer l'incertitude.

## Grad-CAM

Quand le modele entraine est disponible, l'app peut afficher une visualisation
Grad-CAM pour un genre selectionne. L'affiche est grisee et les zones les plus
contributives retrouvent leurs couleurs.

Cette visualisation donne une indication qualitative des regions utilisees par le
modele. Elle ne prouve pas que le modele "comprend" l'affiche ; elle sert a
expliquer approximativement les activations.

## Tests

```bash
python -m unittest discover -s tests
python -m py_compile app.py src/predict.py src/mood.py src/gradcam.py scripts/train_genre_classifier.py
```

Les tests couvrent :

- le fallback quand aucun modele n'est present ;
- le chargement logique des predictions ;
- les classes d'ambiance ;
- des affiches synthetiques pour les heuristiques couleur ;
- les cas principaux Grad-CAM.

## Limites

- Le modele a ete entraine rapidement, donc certaines predictions peuvent etre
  approximatives.
- Le dataset ne couvre pas tous les genres ni toutes les epoques graphiques.
- L'ambiance est heuristique, pas issue d'un entrainement supervise.
- Grad-CAM donne une explication visuelle approximative, pas une preuve causale.

## Ameliorations possibles

- Entrainer plus longtemps avec augmentation de donnees.
- Ajouter des labels d'ambiance reels et entrainer une tete de classification
  dediee.
- Ajouter une branche texte pour analyser le titre, le slogan ou le synopsis.
- Ajouter une recherche d'affiches similaires avec embeddings CNN et distance
  cosinus.
- Evaluer le modele avec precision, rappel, F1-score et matrice par genre.
