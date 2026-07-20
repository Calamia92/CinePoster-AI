import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.mood import estimate_mood, extract_features
from src.predict import analyze_genres


DETECTION_THRESHOLD = 0.5

st.set_page_config(
    page_title="CinePoster Insight",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 CinePoster Insight")
st.caption("Analyse des genres et de l'ambiance d'une affiche de film.")

uploaded_file = st.file_uploader(
    "Importer une affiche",
    type=["jpg", "jpeg", "png", "webp"],
    help="Formats acceptés : JPG, JPEG, PNG, WebP.",
)

if uploaded_file is None:
    st.info("Ajoutez une affiche pour lancer l'analyse.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

poster_col, result_col = st.columns([1, 1.6], gap="large")

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

    st.subheader("Genres probables")

    ranked_genres = sorted(
        analysis.scores.items(), key=lambda item: item[1], reverse=True
    )
    top_columns = st.columns(3)
    for column, (genre, probability) in zip(top_columns, ranked_genres[:3]):
        column.metric(genre, f"{probability:.0%}")

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
                alt.value("#e4572e"),
                alt.value("#94a9b8"),
            ),
            tooltip=[
                alt.Tooltip("genre:N", title="Genre"),
                alt.Tooltip("probabilite:Q", format=".0%", title="Probabilité"),
            ],
        )
        .properties(height=250)
    )
    st.altair_chart(genre_chart, use_container_width=True)
    st.caption(
        f"Les genres au-dessus de {DETECTION_THRESHOLD:.0%} sont considérés "
        "comme détectés."
    )

    if analysis.used_trained_model:
        st.caption("Source : modèle entraîné (models/genre_classifier.keras).")
    else:
        st.caption("Source : prédictions de démonstration, aucun modèle entraîné.")

    st.subheader("Ambiance estimée")
    st.metric("Ambiance", mood.label)
    st.progress(mood.confidence, text=f"Confiance : {mood.confidence:.0%}")
    st.caption(mood.explanation)

    st.subheader("Signaux visuels")
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
        .properties(height=160)
    )
    st.altair_chart(features_chart, use_container_width=True)
