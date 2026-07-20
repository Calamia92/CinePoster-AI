# Issues de depart

## 1. Setup project and Streamlit skeleton

**Objectif**: Avoir une base executable pour que toute l'equipe travaille sur le meme socle.

**Taches**
- Ajouter la structure du projet.
- Ajouter `requirements.txt`.
- Creer une page Streamlit avec upload d'image.
- Afficher une affiche importee.

**Definition of done**
- `streamlit run app.py` lance l'application.
- Une image peut etre importee et affichee.

## 2. Prepare poster dataset and multilabel annotations

**Objectif**: Preparer les donnees pour entrainer le modele multi-label.

**Taches**
- Choisir le dataset.
- Creer un CSV avec chemin d'image et genres.
- Encoder les genres en vecteur multi-label.
- Ajouter le preprocessing image.

**Definition of done**
- Un script ou notebook charge les images et les labels.
- Les labels multi-genres sont exploitables pour l'entrainement.

## 3. Train multilabel genre classifier

**Objectif**: Entrainer un premier modele de classification multi-label.

**Taches**
- Utiliser MobileNetV2, EfficientNetB0 ou ResNet50 en transfer learning.
- Ajouter une sortie `sigmoid`.
- Utiliser `binary_crossentropy`.
- Sauvegarder le modele entraine.

**Definition of done**
- Le modele retourne une probabilite par genre.
- Un fichier modele est sauvegarde et documente.

## 4. Add mood prediction module

**Objectif**: Estimer l'ambiance generale de l'affiche.

**Taches**
- Definir les classes d'ambiance.
- Demarrer avec une heuristique couleur si le dataset manque de labels.
- Prevoir l'interface Python `estimate_mood(image)`.

**Definition of done**
- L'app affiche une ambiance et une courte explication.

## 5. Connect model predictions to Streamlit UI

**Objectif**: Brancher les predictions sur l'interface de demonstration.

**Taches**
- Remplacer les placeholders par le modele entraine.
- Afficher les probabilites des genres en graphique.
- Afficher l'ambiance estimee.
- Gerer les erreurs de chargement du modele.

**Definition of done**
- Une affiche importee produit des predictions visibles dans l'app.

## 6. Bonus: Grad-CAM visualization

**Objectif**: Montrer les zones de l'affiche qui influencent la prediction.

**Taches**
- Implementer Grad-CAM sur la derniere couche convolutionnelle.
- Generer une heatmap.
- Superposer la heatmap sur l'affiche.

**Definition of done**
- L'app affiche une carte de chaleur pour au moins un genre predit.
