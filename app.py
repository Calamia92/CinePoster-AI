import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.gradcam import gradcam_overlay
from src.mood import estimate_mood, extract_features
from src.predict import analyze_genres


DETECTION_THRESHOLD = 0.5

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
    page_icon="🎬",
    layout="wide",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="title-block">
        <p class="eyebrow">Analyse d'affiches de film</p>
        <h1 class="app-title">CinePoster <span>Insight</span></h1>
        <p class="app-subtitle">Genres probables et ambiance visuelle d'une affiche,
        en un coup d'œil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Importer une affiche",
    type=["jpg", "jpeg", "png", "webp"],
    help="Formats acceptés : JPG, JPEG, PNG, WebP.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    with st.container(border=True):
        st.markdown("#### Aucune affiche importée")
        st.caption(
            "Glissez une affiche ci-dessus (JPG, PNG ou WebP) pour lancer "
            "l'analyse des genres et de l'ambiance."
        )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

poster_col, result_col = st.columns([1, 1.7], gap="large")

with poster_col:
    st.image(image, caption=uploaded_file.name, use_container_width=True)

with result_col:
    with st.spinner("Analyse de l'affiche en cours..."):
        analysis = analyze_genres(image)
        mood = estimate_mood(image)
        features = extract_features(image)

    if analysis.load_error:
        st.warning(
            "Le modèle entraîné n'a pas pu être chargé, l'app affiche des "
            f"prédictions de secours. Détail : {analysis.load_error}"
        )

    ranked_genres = sorted(
        analysis.scores.items(), key=lambda item: item[1], reverse=True
    )
    detected = [g for g, p in ranked_genres if p >= DETECTION_THRESHOLD]

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
                f"Aucun genre ne dépasse le seuil de détection "
                f"({DETECTION_THRESHOLD:.0%}) — tendance **{ranked_genres[0][0]}**."
            )

        chips = "".join(
            '<span class="genre-chip'
            + (" detected" if probability >= DETECTION_THRESHOLD else "")
            + f'">{genre} · {probability:.0%}</span>'
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
                    alt.Tooltip("probabilite:Q", format=".0%", title="Probabilité"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(genre_chart, use_container_width=True)

        if analysis.used_trained_model:
            st.caption("Source : modèle entraîné (models/genre_classifier.keras).")
        else:
            st.caption("Source : prédictions de démonstration, aucun modèle entraîné.")

    mood_col, signals_col = st.columns(2, gap="medium")

    with mood_col:
        with st.container(border=True):
            st.markdown("#### Ambiance estimée")
            st.markdown(
                f'<p class="mood-label">{mood.label}</p>', unsafe_allow_html=True
            )
            st.progress(mood.confidence, text=f"Confiance : {mood.confidence:.0%}")
            st.caption(mood.explanation)

    with signals_col:
        with st.container(border=True):
            st.markdown("#### Signaux visuels")
            features_df = pd.DataFrame(
                {
                    "signal": ["Luminosité", "Saturation", "Contraste", "Zones sombres"],
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
            "contribué au genre sélectionné retrouvent leurs couleurs."
        )
        selected_genre = st.selectbox(
            "Genre à expliquer",
            [genre for genre, _ in ranked_genres],
        )
        with st.spinner("Calcul de la carte d'attention..."):
            overlay = gradcam_overlay(image, selected_genre)
        if overlay is None:
            st.info(
                "Le modèle n'a identifié aucune zone qui augmente le score de "
                "ce genre sur cette affiche. Essayez un autre genre."
            )
        else:
            original_col, overlay_col = st.columns(2, gap="medium")
            original_col.image(
                image, caption="Affiche originale", use_container_width=True
            )
            overlay_col.image(
                overlay,
                caption=f"Attention du modèle — {selected_genre}",
                use_container_width=True,
            )
