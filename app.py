from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.mood import estimate_mood
from src.predict import predict_genres


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
    genre_scores = predict_genres(image)
    mood = estimate_mood(image)

    st.subheader("Genres probables")
    genre_df = pd.DataFrame(
        {
            "genre": list(genre_scores.keys()),
            "probabilite": list(genre_scores.values()),
        }
    ).sort_values("probabilite", ascending=False)

    st.bar_chart(genre_df, x="genre", y="probabilite", height=260)

    st.subheader("Ambiance estimee")
    st.metric("Ambiance", mood.label, f"{mood.confidence:.0%}")

    st.subheader("Elements visuels")
    st.write(mood.explanation)

st.divider()
st.caption(
    "MVP: les predictions sont des placeholders. Remplacer src/predict.py par le modele entraine."
)
