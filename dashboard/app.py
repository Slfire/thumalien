"""Dashboard Streamlit pour Thumalien."""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from src.collector.jetstream_collector import JetstreamCollector
from src.config import EMISSIONS_FILE, EMOTION_MODEL_DIR, FAKE_NEWS_MODEL_DIR
from src.database.repository import Repository
from src.detection.classifier import FakeNewsClassifier
from src.emotion.analyzer import EmotionAnalyzer
from src.energy.monitor import EnergyMonitor
from src.preprocessing.pipeline import PreprocessingPipeline

# --- Page config ---
st.set_page_config(page_title="Thumalien", page_icon="🔍", layout="wide")
st.title("Thumalien — Détection de Fake News sur Bluesky")

# --- Session state ---
for key in ("posts", "detection_results", "emotion_results", "energy_results"):
    if key not in st.session_state:
        st.session_state[key] = []


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


# --- Full pipeline ---
def run_full_pipeline(n_posts: int, lang_filter: str, progress_bar, status_text):
    """Execute le pipeline complet : collecte -> detection -> emotion -> DB."""
    energy_results = []

    # 1. Collecte
    status_text.text("Collecte des posts via Jetstream...")
    progress_bar.progress(0.1)
    collector = JetstreamCollector(lang_filter=lang_filter)
    posts = collector.collect_posts_sync(n=n_posts)

    # 2. Preprocessing
    status_text.text("Nettoyage des textes...")
    progress_bar.progress(0.3)
    pipeline = get_preprocessing()
    for p in posts:
        p["cleaned_text"] = pipeline.clean_text(p["text"])

    texts = [p.get("cleaned_text", p["text"]) for p in posts]

    # 3. Detection + energy
    status_text.text("Detection fake news...")
    progress_bar.progress(0.45)
    detection_results = []
    monitor = EnergyMonitor()
    monitor.start()
    try:
        classifier = get_classifier()
        detection_results = classifier.predict(texts)
    except FileNotFoundError:
        pass
    energy = monitor.stop()
    energy["task"] = "detection"
    energy["timestamp"] = datetime.now(timezone.utc).isoformat()
    energy["n_posts"] = len(texts)
    energy_results.append(energy)

    # 4. Emotion + energy
    status_text.text("Analyse emotionnelle...")
    progress_bar.progress(0.65)
    monitor2 = EnergyMonitor()
    monitor2.start()
    analyzer = get_emotion_analyzer()
    vader_results = [analyzer.analyze_vader(t) for t in texts]
    bert_results = analyzer.analyze_bert(texts)
    emotion_results = []
    for vader, bert in zip(vader_results, bert_results):
        combined = dict(vader)
        combined["bert_emotion"] = bert.get("emotion", "unknown")
        combined["bert_score"] = bert.get("score", 0.0)
        combined["emotions_bert"] = bert.get("all_scores", {})
        emotion_results.append(combined)
    energy2 = monitor2.stop()
    energy2["task"] = "emotion"
    energy2["timestamp"] = datetime.now(timezone.utc).isoformat()
    energy2["n_posts"] = len(texts)
    energy_results.append(energy2)

    # 5. Save to DB
    status_text.text("Sauvegarde en base de donnees...")
    progress_bar.progress(0.85)
    try:
        repo = get_repository()
        for i, p in enumerate(posts):
            db_post = repo.save_post(p)
            if i < len(detection_results):
                det = detection_results[i]
                repo.save_detection_result(
                    post_id=db_post.id,
                    label=det["label"],
                    score=det["score"],
                    model_name="distilbert-fake-news-lora",
                )
            if i < len(emotion_results):
                emo = emotion_results[i]
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

    # Save energy
    for e in energy_results:
        EnergyMonitor.save_record(e, EMISSIONS_FILE)

    progress_bar.progress(1.0)
    status_text.success(f"Pipeline termine : {len(posts)} posts analyses")

    return {
        "posts": posts,
        "detection_results": detection_results,
        "emotion_results": emotion_results,
        "energy_results": energy_results,
    }


# --- Sidebar ---
st.sidebar.header("Configuration")
n_posts = st.sidebar.slider("Nombre de posts a collecter", 5, 100, 10)
lang_filter = st.sidebar.text_input("Filtre langue", value="fr")

st.sidebar.markdown("---")

# Combined pipeline button
if st.sidebar.button("Collecter & Analyser", type="primary"):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    result = run_full_pipeline(n_posts, lang_filter, progress_bar, status_text)

    st.session_state.posts = result["posts"]
    st.session_state.detection_results = result["detection_results"]
    st.session_state.emotion_results = result["emotion_results"]
    st.session_state.energy_results = result["energy_results"]

st.sidebar.markdown("---")
st.sidebar.caption("Ou etape par etape :")

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

    st.sidebar.success(f"{len(posts)} posts collectes")

# --- Summary metrics ---
if st.session_state.posts:
    col1, col2, col3 = st.columns(3)
    col1.metric("Posts collectes", len(st.session_state.posts))
    if st.session_state.detection_results:
        fake_count = sum(
            1 for r in st.session_state.detection_results if r["label"] == "FAKE"
        )
        col2.metric(
            "Posts FAKE",
            f"{fake_count}/{len(st.session_state.detection_results)}",
        )
    if st.session_state.emotion_results:
        emotions = [e["bert_emotion"] for e in st.session_state.emotion_results]
        dominant = Counter(emotions).most_common(1)[0][0] if emotions else "-"
        col3.metric("Emotion dominante", dominant)

st.divider()

# --- Posts table ---
st.header("Posts collectes")
if st.session_state.posts:
    df_posts = pd.DataFrame(
        [
            {
                "DID": p["did"][-15:],
                "Texte": p["text"][:120] + ("..." if len(p["text"]) > 120 else ""),
                "Langue": ", ".join(p.get("langs", [])),
            }
            for p in st.session_state.posts
        ]
    )
    st.dataframe(df_posts, use_container_width=True)

    # Word Cloud
    all_text = " ".join(
        p.get("cleaned_text", p["text"]) for p in st.session_state.posts
    )
    if len(all_text.split()) > 10:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
        ).generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title("Nuage de mots des posts collectes")
        st.pyplot(fig_wc)
        plt.close(fig_wc)

    # Time series
    timestamps = [p.get("created_at") for p in st.session_state.posts]
    valid_ts = [t for t in timestamps if t]
    if len(valid_ts) > 1:
        df_time = pd.DataFrame({"timestamp": pd.to_datetime(valid_ts)})
        df_time["count"] = 1
        df_time = df_time.set_index("timestamp").resample("1min").sum().reset_index()
        if len(df_time) > 1:
            fig_time = px.line(
                df_time,
                x="timestamp",
                y="count",
                title="Volume de posts dans le temps",
            )
            st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("Aucun post. Utilisez le bouton dans la barre laterale pour collecter.")

st.divider()

# --- Two columns: Detection + Emotion ---
col_detect, col_emotion = st.columns(2)

# --- Detection ---
with col_detect:
    st.header("Detection Fake News")
    if st.session_state.posts and st.button("Lancer la detection"):
        with st.spinner("Classification en cours..."):
            try:
                classifier = get_classifier()
                texts = [
                    p.get("cleaned_text", p["text"]) for p in st.session_state.posts
                ]

                monitor = EnergyMonitor()
                monitor.start()
                results = classifier.predict(texts)
                energy = monitor.stop()
                energy["task"] = "detection"
                energy["timestamp"] = datetime.now(timezone.utc).isoformat()
                energy["n_posts"] = len(texts)
                EnergyMonitor.save_record(energy, EMISSIONS_FILE)
                st.session_state.energy_results.append(energy)

                st.session_state.detection_results = results

                try:
                    repo = get_repository()
                    for p, r in zip(st.session_state.posts, results):
                        db_post = repo.save_post(p)
                        repo.save_detection_result(
                            post_id=db_post.id,
                            label=r["label"],
                            score=r["score"],
                            model_name="distilbert-fake-news-lora",
                        )
                except Exception:
                    pass
            except FileNotFoundError as e:
                st.error(str(e))

    if st.session_state.detection_results:
        label_colors = {"REAL": "green", "FAKE": "red"}
        for post, det in zip(
            st.session_state.posts, st.session_state.detection_results
        ):
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
        st.write("Cliquez sur 'Lancer la detection' apres avoir collecte des posts.")

# --- Emotion ---
with col_emotion:
    st.header("Analyse emotionnelle")
    if st.session_state.posts and st.button("Analyser les emotions"):
        with st.spinner("Analyse VADER + modele entraine..."):
            analyzer = get_emotion_analyzer()
            texts = [
                p.get("cleaned_text", p["text"]) for p in st.session_state.posts
            ]

            monitor = EnergyMonitor()
            monitor.start()
            vader_results = [analyzer.analyze_vader(t) for t in texts]
            bert_results = analyzer.analyze_bert(texts)
            energy = monitor.stop()
            energy["task"] = "emotion"
            energy["timestamp"] = datetime.now(timezone.utc).isoformat()
            energy["n_posts"] = len(texts)
            EnergyMonitor.save_record(energy, EMISSIONS_FILE)
            st.session_state.energy_results.append(energy)

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
            "Negatif": df_emo["neg"].mean(),
            "Neutre": df_emo["neu"].mean(),
        }
        fig_avg = px.pie(
            names=list(avg_scores.keys()),
            values=list(avg_scores.values()),
            title="Repartition moyenne des sentiments (VADER)",
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        if df_emo["bert_emotion"].iloc[0] != "unknown":
            st.subheader("Modele entraine (6 emotions)")
            emotion_colors = {
                "sadness": "#636EFA",
                "joy": "#00CC96",
                "love": "#FF6692",
                "anger": "#EF553B",
                "fear": "#AB63FA",
                "surprise": "#FFA15A",
            }

            # Radar chart
            all_bert_scores = [
                e.get("emotions_bert", {}) for e in st.session_state.emotion_results
            ]
            if all_bert_scores and all_bert_scores[0]:
                avg_emotions = {}
                for scores in all_bert_scores:
                    for emotion, score in scores.items():
                        avg_emotions[emotion] = avg_emotions.get(emotion, 0) + score
                avg_emotions = {
                    k: v / len(all_bert_scores) for k, v in avg_emotions.items()
                }

                emotions_list = list(avg_emotions.keys())
                values = list(avg_emotions.values())
                fig_radar = px.line_polar(
                    r=values + [values[0]],
                    theta=emotions_list + [emotions_list[0]],
                    title="Profil emotionnel moyen",
                )
                fig_radar.update_traces(fill="toself")
                st.plotly_chart(fig_radar, use_container_width=True)

            # Histogram
            emotions = df_emo["bert_emotion"].tolist()
            fig_emotions = px.histogram(
                x=emotions,
                title="Distribution des emotions detectees",
                color=emotions,
                color_discrete_map=emotion_colors,
            )
            fig_emotions.update_layout(xaxis_title="Emotion", yaxis_title="Nombre")
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
                    st.caption(
                        f"... et {len(st.session_state.posts) - 10} autres"
                    )
                    break
    elif st.session_state.posts:
        st.write("Cliquez sur 'Analyser les emotions' apres avoir collecte des posts.")

st.divider()

# --- Model Performance ---
st.header("Performance des modeles")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Detection Fake News")
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
        st.info("Modele non entraine. Lancez : python -m src.training.train_fake_news")

with col_m2:
    st.subheader("Analyse emotionnelle")
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
        st.info("Modele non entraine. Lancez : python -m src.training.train_emotion")

st.divider()

# --- Empreinte Carbone (Green IT) ---
st.header("Empreinte Carbone (Green IT)")

# Current session
if st.session_state.energy_results:
    cols_energy = st.columns(len(st.session_state.energy_results))
    for col, e in zip(cols_energy, st.session_state.energy_results):
        col.metric(
            f"{e['task'].capitalize()}",
            f"{e['emissions_kg'] * 1000:.4f} g CO2",
            delta=f"{e['duration_s']:.1f}s",
        )

# Historical
if os.path.exists(EMISSIONS_FILE):
    with open(EMISSIONS_FILE) as f:
        try:
            all_emissions = json.load(f)
        except json.JSONDecodeError:
            all_emissions = []

    if all_emissions:
        df_emissions = pd.DataFrame(all_emissions)
        total_co2 = df_emissions["emissions_kg"].sum()
        st.metric("Total CO2 emis (cumule)", f"{total_co2 * 1000:.4f} g CO2eq")

        if "task" in df_emissions.columns and "timestamp" in df_emissions.columns:
            fig_carbon = px.bar(
                df_emissions,
                x="timestamp",
                y="emissions_kg",
                color="task",
                title="Historique des emissions par analyse",
            )
            fig_carbon.update_layout(
                yaxis_title="Emissions (kg CO2eq)", xaxis_title=""
            )
            st.plotly_chart(fig_carbon, use_container_width=True)
else:
    st.info("Aucune donnee d'emission. Lancez une analyse pour commencer le suivi.")

st.divider()

# --- Stats BDD ---
st.header("Statistiques BDD")
try:
    repo = get_repository()
    stats = repo.get_stats()

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total posts", stats["total_posts"])
    col_s2.metric("Analyses detection", stats["total_detections"])
    col_s3.metric("Analyses emotion", stats["total_emotions"])

    if stats["label_distribution"]:
        fig_dist = px.bar(
            x=list(stats["label_distribution"].keys()),
            y=list(stats["label_distribution"].values()),
            title="Distribution globale des labels",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
except Exception as e:
    st.warning(f"Base de donnees non disponible : {e}")

st.markdown("---")
st.caption("Thumalien — Projet M1 Data & IA")
