"""Dashboard Streamlit pour Thumalien."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.collector.jetstream_collector import JetstreamCollector
from src.config import EMOTION_MODEL_DIR, FAKE_NEWS_MODEL_DIR
from src.database.repository import Repository
from src.detection.classifier import FakeNewsClassifier
from src.emotion.analyzer import EmotionAnalyzer
from src.preprocessing.pipeline import PreprocessingPipeline

# --- Page config ---
st.set_page_config(page_title="Thumalien", page_icon="🔍", layout="wide")
st.title("Thumalien — Détection de Fake News sur Bluesky")

# --- Session state ---
if "posts" not in st.session_state:
    st.session_state.posts = []
if "detection_results" not in st.session_state:
    st.session_state.detection_results = []
if "emotion_results" not in st.session_state:
    st.session_state.emotion_results = []


# --- Cached resources ---
@st.cache_resource
def get_preprocessing():
    return PreprocessingPipeline()


@st.cache_resource
def get_classifier():
    return FakeNewsClassifier()


@st.cache_resource
def get_emotion_analyzer():
    return EmotionAnalyzer()


@st.cache_resource
def get_repository():
    repo = Repository()
    repo.create_tables()
    return repo


def load_model_metrics(model_dir: str) -> dict | None:
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


# --- Sidebar ---
st.sidebar.header("Configuration")
n_posts = st.sidebar.slider("Nombre de posts à collecter", 5, 100, 10)
lang_filter = st.sidebar.text_input("Filtre langue", value="fr")

if st.sidebar.button("Collecter les posts"):
    with st.spinner(f"Collecte de {n_posts} posts via Jetstream..."):
        collector = JetstreamCollector(lang_filter=lang_filter)
        posts = collector.collect_posts_sync(n=n_posts)

        pipeline = get_preprocessing()
        for p in posts:
            p["cleaned_text"] = pipeline.clean_text(p["text"])

        try:
            repo = get_repository()
            for p in posts:
                repo.save_post(p)
        except Exception:
            pass

        st.session_state.posts = posts
        st.session_state.detection_results = []
        st.session_state.emotion_results = []

    st.sidebar.success(f"{len(posts)} posts collectés")

# --- Posts table ---
st.header("Posts collectés")
if st.session_state.posts:
    df_posts = pd.DataFrame([
        {
            "DID": p["did"][-15:],
            "Texte": p["text"][:120] + ("..." if len(p["text"]) > 120 else ""),
            "Langue": ", ".join(p.get("langs", [])),
        }
        for p in st.session_state.posts
    ])
    st.dataframe(df_posts, use_container_width=True)
else:
    st.info("Aucun post. Utilisez le bouton dans la barre latérale pour collecter.")

# --- Two columns: Detection + Emotion ---
col_detect, col_emotion = st.columns(2)

# --- Detection ---
with col_detect:
    st.header("Détection Fake News")
    if st.session_state.posts and st.button("Lancer la détection"):
        with st.spinner("Classification fine-tunée en cours..."):
            try:
                classifier = get_classifier()
                texts = [p.get("cleaned_text", p["text"]) for p in st.session_state.posts]
                results = classifier.predict(texts)
                st.session_state.detection_results = results

                try:
                    repo = get_repository()
                    for p, r in zip(st.session_state.posts, results):
                        db_post = repo.save_post(p)
                        repo.save_detection_result(
                            post_id=db_post.id,
                            label=r["label"],
                            score=r["score"],
                            model_name=classifier.model_name,
                        )
                except Exception:
                    pass
            except FileNotFoundError as e:
                st.error(str(e))

    if st.session_state.detection_results:
        label_colors = {"REAL": "green", "FAKE": "red"}
        for post, det in zip(st.session_state.posts, st.session_state.detection_results):
            color = label_colors.get(det["label"], "gray")
            st.markdown(
                f'<span style="color:{color};font-weight:bold">'
                f'[{det["label"]} — {det["score"]:.0%}]</span> '
                f'{post["text"][:100]}',
                unsafe_allow_html=True,
            )

        labels = [r["label"] for r in st.session_state.detection_results]
        fig = px.histogram(
            x=labels,
            title="Distribution des labels",
            color=labels,
            color_discrete_map=label_colors,
        )
        fig.update_layout(xaxis_title="Label", yaxis_title="Nombre")
        st.plotly_chart(fig, use_container_width=True)
    elif st.session_state.posts:
        st.write("Cliquez sur 'Lancer la détection' après avoir collecté des posts.")

# --- Emotion ---
with col_emotion:
    st.header("Analyse émotionnelle")
    if st.session_state.posts and st.button("Analyser les émotions"):
        with st.spinner("Analyse VADER + modèle entraîné..."):
            analyzer = get_emotion_analyzer()
            texts = [p.get("cleaned_text", p["text"]) for p in st.session_state.posts]

            vader_results = [analyzer.analyze_vader(t) for t in texts]
            bert_results = analyzer.analyze_bert(texts)

            results = []
            for vader, bert in zip(vader_results, bert_results):
                combined = dict(vader)
                combined["bert_emotion"] = bert.get("emotion", "unknown")
                combined["bert_score"] = bert.get("score", 0.0)
                combined["emotions_bert"] = bert.get("all_scores", {})
                results.append(combined)

            st.session_state.emotion_results = results

            try:
                repo = get_repository()
                for p, emo in zip(st.session_state.posts, results):
                    db_post = repo.save_post(p)
                    repo.save_emotion_result(
                        post_id=db_post.id,
                        compound=emo["compound"],
                        positive=emo["pos"],
                        negative=emo["neg"],
                        neutral=emo["neu"],
                        emotions_bert=emo.get("emotions_bert"),
                    )
            except Exception:
                pass

    if st.session_state.emotion_results:
        df_emo = pd.DataFrame(st.session_state.emotion_results)

        st.subheader("VADER (baseline)")
        fig_compound = px.bar(
            df_emo,
            y="compound",
            title="Score compound par post",
            color="compound",
            color_continuous_scale=["red", "gray", "green"],
            range_color=[-1, 1],
        )
        st.plotly_chart(fig_compound, use_container_width=True)

        avg_scores = {
            "Positif": df_emo["pos"].mean(),
            "Négatif": df_emo["neg"].mean(),
            "Neutre": df_emo["neu"].mean(),
        }
        fig_avg = px.pie(
            names=list(avg_scores.keys()),
            values=list(avg_scores.values()),
            title="Répartition moyenne des sentiments (VADER)",
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        if df_emo["bert_emotion"].iloc[0] != "unknown":
            st.subheader("Modèle entraîné (6 émotions)")
            emotion_colors = {
                "sadness": "#636EFA",
                "joy": "#00CC96",
                "love": "#FF6692",
                "anger": "#EF553B",
                "fear": "#AB63FA",
                "surprise": "#FFA15A",
            }
            emotions = df_emo["bert_emotion"].tolist()
            fig_emotions = px.histogram(
                x=emotions,
                title="Distribution des émotions détectées",
                color=emotions,
                color_discrete_map=emotion_colors,
            )
            fig_emotions.update_layout(xaxis_title="Émotion", yaxis_title="Nombre")
            st.plotly_chart(fig_emotions, use_container_width=True)

            for i, (post, emo) in enumerate(
                zip(st.session_state.posts, st.session_state.emotion_results)
            ):
                color = emotion_colors.get(emo["bert_emotion"], "gray")
                st.markdown(
                    f'<span style="color:{color};font-weight:bold">'
                    f'[{emo["bert_emotion"]} — {emo["bert_score"]:.0%}]</span> '
                    f'{post["text"][:80]}',
                    unsafe_allow_html=True,
                )
                if i >= 9:
                    st.caption(f"... et {len(st.session_state.posts) - 10} autres")
                    break
    elif st.session_state.posts:
        st.write("Cliquez sur 'Analyser les émotions' après avoir collecté des posts.")

# --- Model Performance ---
st.header("Performance des modèles")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Détection Fake News")
    fn_metrics = load_model_metrics(FAKE_NEWS_MODEL_DIR)
    if fn_metrics:
        st.metric("Accuracy", f"{fn_metrics['accuracy']:.1%}")
        st.metric("F1 Score", f"{fn_metrics['f1']:.1%}")
        st.caption(
            f"Base: {fn_metrics['base_model']} | "
            f"Epochs: {fn_metrics['epochs']} | "
            f"Train: {fn_metrics['train_size']:,} | Val: {fn_metrics['val_size']:,}"
        )
    else:
        st.info("Modèle non entraîné. Lancez : python -m src.training.train_fake_news")

with col_m2:
    st.subheader("Analyse émotionnelle")
    emo_metrics = load_model_metrics(EMOTION_MODEL_DIR)
    if emo_metrics:
        st.metric("Accuracy", f"{emo_metrics['accuracy']:.1%}")
        st.metric("F1 Macro", f"{emo_metrics['f1_macro']:.1%}")
        st.caption(
            f"Base: {emo_metrics['base_model']} | "
            f"Classes: {emo_metrics['num_classes']} | "
            f"Epochs: {emo_metrics['epochs']}"
        )
    else:
        st.info("Modèle non entraîné. Lancez : python -m src.training.train_emotion")

# --- Stats BDD ---
st.header("Statistiques")
try:
    repo = get_repository()
    stats = repo.get_stats()

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total posts", stats["total_posts"])
    col_s2.metric("Analyses détection", stats["total_detections"])
    col_s3.metric("Analyses émotion", stats["total_emotions"])

    if stats["label_distribution"]:
        fig_dist = px.bar(
            x=list(stats["label_distribution"].keys()),
            y=list(stats["label_distribution"].values()),
            title="Distribution globale des labels",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
except Exception as e:
    st.warning(f"Base de données non disponible : {e}")

st.markdown("---")
st.caption("Thumalien — Projet M1 Data & IA")
