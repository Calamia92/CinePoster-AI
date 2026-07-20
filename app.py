from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.mood import estimate_mood
from src.predict import analyze_genres


st.set_page_config(
    page_title="CinePoster Insight",
    layout="wide",
)


st.title("CinePoster Insight")
st.caption("Analyse rapide des genres et de l'ambiance d'une affiche de film.")

uploaded_file = st.file_uploader(
    "Importer une affiche",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Ajoutez une affiche pour lancer l'analyse.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

poster_col, result_col = st.columns([1, 1.4])

with poster_col:
    st.image(image, caption=uploaded_file.name, use_container_width=True)

with result_col:
    analysis = analyze_genres(image)
    mood = estimate_mood(image)

    st.subheader("Genres probables")
    if analysis.load_error:
        st.warning(
            "Le modele entraine n'a pas pu etre charge, l'app affiche les "
            f"predictions placeholder du MVP. Detail: {analysis.load_error}"
        )
    elif analysis.used_trained_model:
        st.caption("Predictions du modele entraine (models/genre_classifier.keras).")
    else:
        st.caption("Aucun modele entraine trouve, predictions placeholder du MVP.")

    genre_df = pd.DataFrame(
        {
            "genre": list(analysis.scores.keys()),
            "probabilite": list(analysis.scores.values()),
        }
    ).sort_values("probabilite", ascending=False)

    st.bar_chart(genre_df, x="genre", y="probabilite", height=260)

    st.subheader("Ambiance estimee")
    st.metric("Ambiance", mood.label, f"{mood.confidence:.0%}")

    st.subheader("Elements visuels")
    st.write(mood.explanation)

st.divider()
st.caption(
    "MVP: l'app utilise models/genre_classifier.keras si le modele a ete entraine."
)
