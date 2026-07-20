import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.gradcam import gradcam_overlay
from src.mood import analyze_mood, extract_features
from src.predict import analyze_genres


DETECTION_THRESHOLD = 0.5
REPORTS_DIR = Path("reports")
TRAINING_CURVES_PATH = REPORTS_DIR / "training_curves.png"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.json"

CUSTOM_CSS = """
<style>
header[data-testid="stHeader"] {display: none;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1150px;}
.title-block {
    padding-bottom: 1.1rem;
    border-bottom: 1px solid #232b3d;
    margin-bottom: 1.5rem;
}
.eyebrow {
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: #f5b301;
    margin: 0 0 0.25rem;
}
.app-title {font-size: 2.1rem; font-weight: 700; color: #f2f4f8; margin: 0;}
.app-title span {color: #f5b301;}
.app-subtitle {color: #98a1b3; margin: 0.35rem 0 0; font-size: 0.95rem;}
div[data-testid="stImage"] img, div[data-testid="stImageContainer"] img {
    border-radius: 10px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.5);
}
.genre-chip {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    margin: 0 0.3rem 0.35rem 0;
    font-size: 0.83rem;
    background: #1c2333;
    border: 1px solid #2a3247;
    color: #c3cad9;
}
.genre-chip.detected {
    background: rgba(245, 179, 1, 0.14);
    border-color: #f5b301;
    color: #f5b301;
    font-weight: 600;
}
.mood-label {
    font-size: 1.35rem;
    font-weight: 600;
    color: #f5b301;
    margin: 0.1rem 0 0.7rem;
}
</style>
"""


st.set_page_config(
    page_title="CinePoster Insight",
    layout="wide",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="title-block">
        <p class="eyebrow">Analyse d'affiches de film</p>
        <h1 class="app-title">CinePoster <span>Insight</span></h1>
        <p class="app-subtitle">Genres probables, ambiance visuelle et zones
        d'attention du modele en un coup d'oeil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Importer une affiche",
    type=["jpg", "jpeg", "png", "webp"],
    help="Formats acceptes : JPG, JPEG, PNG, WebP.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    with st.container(border=True):
        st.markdown("#### Aucune affiche importee")
        st.caption(
            "Glissez une affiche ci-dessus pour lancer l'analyse des genres, "
            "de l'ambiance et des zones d'attention."
        )
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as error:
    st.error(
        "Impossible de lire cette affiche. Importez une image valide au format "
        f"JPG, PNG ou WebP. Detail : {error}"
    )
    st.stop()

poster_col, result_col = st.columns([1, 1.7], gap="large")

with poster_col:
    st.image(image, caption=uploaded_file.name, use_container_width=True)

with result_col:
    with st.spinner("Analyse de l'affiche en cours..."):
        analysis = analyze_genres(image)
        mood = analyze_mood(image)
        features = extract_features(image)

    if analysis.load_error:
        st.warning(
            "Le modele entraine n'a pas pu etre charge, l'app affiche des "
            f"predictions de secours. Detail : {analysis.load_error}"
        )

    ranked_genres = sorted(
        analysis.scores.items(), key=lambda item: item[1], reverse=True
    )
    detected = [
        genre
        for genre, probability in ranked_genres
        if probability >= DETECTION_THRESHOLD
    ]

    with st.container(border=True):
        st.markdown("#### Genres probables")
        if detected:
            st.markdown(
                f"Dominante **{detected[0]}**"
                + (f", avec {', '.join(detected[1:])}" if len(detected) > 1 else "")
                + "."
            )
        else:
            st.markdown(
                f"Aucun genre ne depasse le seuil de detection "
                f"({DETECTION_THRESHOLD:.0%}) - tendance **{ranked_genres[0][0]}**."
            )

        chips = "".join(
            '<span class="genre-chip'
            + (" detected" if probability >= DETECTION_THRESHOLD else "")
            + f'">{genre} - {probability:.0%}</span>'
            for genre, probability in ranked_genres
        )
        st.markdown(chips, unsafe_allow_html=True)

        genre_df = pd.DataFrame(ranked_genres, columns=["genre", "probabilite"])
        genre_chart = (
            alt.Chart(genre_df)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X(
                    "probabilite:Q",
                    axis=alt.Axis(format="%", title=None),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y("genre:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.probabilite >= DETECTION_THRESHOLD,
                    alt.value("#f5b301"),
                    alt.value("#3d4a5c"),
                ),
                tooltip=[
                    alt.Tooltip("genre:N", title="Genre"),
                    alt.Tooltip("probabilite:Q", format=".0%", title="Probabilite"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(genre_chart, use_container_width=True)

        if analysis.used_trained_model:
            st.caption("Source : modele entraine (models/genre_classifier.keras).")
        else:
            st.caption("Source : predictions de demonstration, aucun modele entraine.")

    mood_col, signals_col = st.columns(2, gap="medium")

    with mood_col:
        with st.container(border=True):
            st.markdown("#### Ambiance estimee")
            st.markdown(
                f'<p class="mood-label">{mood.label}</p>', unsafe_allow_html=True
            )
            st.progress(mood.confidence, text=f"Confiance : {mood.confidence:.0%}")
            st.caption(mood.explanation)

            mood_scores = sorted(
                mood.scores.items(), key=lambda item: item[1], reverse=True
            )
            mood_df = pd.DataFrame(mood_scores, columns=["ambiance", "score"])
            mood_chart = (
                alt.Chart(mood_df)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X(
                        "score:Q",
                        axis=alt.Axis(format="%", title=None),
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y("ambiance:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.ambiance == mood.label,
                        alt.value("#f5b301"),
                        alt.value("#3d4a5c"),
                    ),
                    tooltip=[
                        alt.Tooltip("ambiance:N", title="Ambiance"),
                        alt.Tooltip("score:Q", format=".0%", title="Score"),
                    ],
                )
                .properties(height=190)
            )
            st.altair_chart(mood_chart, use_container_width=True)

    with signals_col:
        with st.container(border=True):
            st.markdown("#### Signaux visuels")
            features_df = pd.DataFrame(
                {
                    "signal": ["Luminosite", "Saturation", "Contraste", "Zones sombres"],
                    "valeur": [
                        features.brightness,
                        features.saturation,
                        features.contrast,
                        features.dark_ratio,
                    ],
                }
            )
            features_chart = (
                alt.Chart(features_df)
                .mark_bar(cornerRadius=4, color="#4c7c9b")
                .encode(
                    x=alt.X(
                        "valeur:Q",
                        axis=alt.Axis(format="%", title=None),
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y("signal:N", sort=None, title=None),
                    tooltip=[
                        alt.Tooltip("signal:N", title="Signal"),
                        alt.Tooltip("valeur:Q", format=".0%", title="Valeur"),
                    ],
                )
                .properties(height=170)
            )
            st.altair_chart(features_chart, use_container_width=True)

if analysis.used_trained_model:
    with st.container(border=True):
        st.markdown("#### Zones d'attention (Grad-CAM)")
        st.caption(
            "L'affiche passe en noir et blanc ; les zones qui ont le plus "
            "contribue au genre selectionne retrouvent leurs couleurs."
        )
        selected_genre = st.selectbox(
            "Genre a expliquer",
            [genre for genre, _ in ranked_genres],
        )
        with st.spinner("Calcul de la carte d'attention..."):
            try:
                overlay = gradcam_overlay(image, selected_genre)
            except Exception as error:
                overlay = None
                st.warning(f"Grad-CAM indisponible pour cette image : {error}")

        if overlay is None:
            st.info(
                "Le modele n'a identifie aucune zone qui augmente le score de "
                "ce genre sur cette affiche. Essayez un autre genre."
            )
        else:
            original_col, overlay_col = st.columns(2, gap="medium")
            original_col.image(
                image, caption="Affiche originale", use_container_width=True
            )
            overlay_col.image(
                overlay,
                caption=f"Attention du modele - {selected_genre}",
                use_container_width=True,
            )

if TRAINING_CURVES_PATH.exists():
    with st.expander("Metriques d'entrainement du modele"):
        if TRAINING_SUMMARY_PATH.exists():
            with TRAINING_SUMMARY_PATH.open(encoding="utf-8") as summary_file:
                summary = json.load(summary_file)
            metric_cols = st.columns(3)
            metric_cols[0].metric("Validation loss", f"{summary['val_loss']:.3f}")
            metric_cols[1].metric(
                "Validation accuracy", f"{summary['val_binary_accuracy']:.1%}"
            )
            metric_cols[2].metric("Validation AUC", f"{summary['val_auc']:.1%}")

        st.image(
            str(TRAINING_CURVES_PATH),
            caption="Evolution loss / accuracy / AUC pendant l'entrainement",
            use_container_width=True,
        )
