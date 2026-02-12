"""Dashboard Streamlit pour Thumalien."""

import streamlit as st

st.set_page_config(page_title="Thumalien", page_icon="🔍", layout="wide")

st.title("Thumalien — Détection de Fake News sur Bluesky")

st.sidebar.header("Configuration")
query = st.sidebar.text_input("Recherche", value="actualité")

st.header("Résultats")
st.info("Pipeline en cours de développement. Connectez les modules pour voir les résultats.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Détection fake news")
    st.write("Aucune donnée pour le moment.")
with col2:
    st.subheader("Analyse émotionnelle")
    st.write("Aucune donnée pour le moment.")

st.header("Empreinte carbone")
st.write("Le suivi CodeCarbon sera affiché ici.")
