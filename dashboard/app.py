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
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from src.collector.jetstream_collector import JetstreamCollector
from src.config import EMISSIONS_FILE, EMOTION_MODEL_DIR, FAKE_NEWS_MODEL_DIR
from src.database.repository import Repository
from src.detection.classifier import FakeNewsClassifier
from src.emotion.analyzer import EmotionAnalyzer
from src.energy.monitor import EnergyMonitor
from src.filtering.news_filter import NewsFilter
from src.preprocessing.pipeline import PreprocessingPipeline

# --- Page config ---
st.set_page_config(page_title="Thumalien", page_icon="🔍", layout="wide")

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
def get_news_filter():
    return NewsFilter()


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


LABEL_COLORS = {"REAL": "#2ecc71", "FAKE": "#e74c3c", "CASUAL": "#95a5a6"}


# --- Full pipeline ---
def run_full_pipeline(n_posts: int, lang_filter: str, progress_bar, status_text):
    """Execute le pipeline complet : collecte -> filtre -> detection -> emotion -> DB."""
    energy_results = []

    # 1. Collecte
    status_text.text("Collecting posts from Jetstream...")
    progress_bar.progress(0.1)
    collector = JetstreamCollector(lang_filter=lang_filter)
    posts = collector.collect_posts_sync(n=n_posts)

    # 2. Preprocessing
    status_text.text("Cleaning texts...")
    progress_bar.progress(0.25)
    pipeline = get_preprocessing()
    for p in posts:
        p["cleaned_text"] = pipeline.clean_text(p["text"])

    # 3. News filter
    status_text.text("Filtering news-like posts...")
    progress_bar.progress(0.35)
    news_filter = get_news_filter()
    posts = news_filter.filter_posts(posts)

    news_posts = [p for p in posts if p.get("is_news")]
    news_texts = [p.get("cleaned_text", p["text"]) for p in news_posts]
    all_texts = [p.get("cleaned_text", p["text"]) for p in posts]

    # 4. Detection on news-like posts only + energy
    status_text.text(f"Fake news detection on {len(news_texts)}/{len(posts)} news-like posts...")
    progress_bar.progress(0.45)
    detection_results = []
    monitor = EnergyMonitor()
    monitor.start()
    news_detections = []
    if news_texts:
        try:
            classifier = get_classifier()
            news_detections = classifier.predict(news_texts)
        except FileNotFoundError:
            news_detections = [{"label": "N/A", "score": 0.0, "all_scores": {}}] * len(news_texts)
    energy = monitor.stop()
    energy["task"] = "detection"
    energy["timestamp"] = datetime.now(timezone.utc).isoformat()
    energy["n_posts"] = len(news_texts)
    energy["n_filtered"] = len(posts) - len(news_texts)
    energy_results.append(energy)

    # Map back: news posts get their detection, casual posts get CASUAL
    news_idx = 0
    for p in posts:
        if p.get("is_news") and news_idx < len(news_detections):
            detection_results.append(news_detections[news_idx])
            news_idx += 1
        else:
            detection_results.append({"label": "CASUAL", "score": 1.0, "all_scores": {}})

    # 5. Emotion on ALL posts + energy
    status_text.text("Emotion analysis on all posts...")
    progress_bar.progress(0.65)
    monitor2 = EnergyMonitor()
    monitor2.start()
    analyzer = get_emotion_analyzer()
    vader_results = [analyzer.analyze_vader(t) for t in all_texts]
    bert_results = analyzer.analyze_bert(all_texts)
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
    energy2["n_posts"] = len(all_texts)
    energy_results.append(energy2)

    # 6. Save to DB
    status_text.text("Saving to database...")
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
    news_count = sum(1 for p in posts if p.get("is_news"))
    status_text.success(f"Done: {len(posts)} posts ({news_count} news-like, {len(posts) - news_count} casual)")

    return {
        "posts": posts,
        "detection_results": detection_results,
        "emotion_results": emotion_results,
        "energy_results": energy_results,
    }


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Thumalien")
st.sidebar.caption("Fake news detection & emotion analysis on Bluesky")

st.sidebar.markdown("---")
st.sidebar.subheader("Collection settings")
n_posts = st.sidebar.slider("Number of posts", 5, 200, 20)
lang_filter = st.sidebar.text_input("Language (ISO code)", value="en", help="e.g. en, fr, es, pt. Leave empty for all.")

st.sidebar.markdown("---")

# Combined pipeline button
if st.sidebar.button("Collect & Analyze", type="primary", use_container_width=True):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    result = run_full_pipeline(n_posts, lang_filter, progress_bar, status_text)

    st.session_state.posts = result["posts"]
    st.session_state.detection_results = result["detection_results"]
    st.session_state.emotion_results = result["emotion_results"]
    st.session_state.energy_results = result["energy_results"]

st.sidebar.markdown("---")

with st.sidebar.expander("Step by step", expanded=False):
    if st.button("1. Collect only", use_container_width=True):
        with st.spinner(f"Collecting {n_posts} posts via Jetstream..."):
            collector = JetstreamCollector(lang_filter=lang_filter)
            posts = collector.collect_posts_sync(n=n_posts)
            pipeline_pp = get_preprocessing()
            news_filter = get_news_filter()
            for p in posts:
                p["cleaned_text"] = pipeline_pp.clean_text(p["text"])
            posts = news_filter.filter_posts(posts)
            try:
                repo = get_repository()
                for p in posts:
                    repo.save_post(p)
            except Exception:
                pass
            st.session_state.posts = posts
            st.session_state.detection_results = []
            st.session_state.emotion_results = []
        st.sidebar.success(f"{len(posts)} posts collected")

    if st.button("2. Detect fake news", use_container_width=True):
        if st.session_state.posts:
            with st.spinner("Classification in progress..."):
                try:
                    classifier = get_classifier()
                    news_posts = [p for p in st.session_state.posts if p.get("is_news")]
                    news_texts = [p.get("cleaned_text", p["text"]) for p in news_posts]

                    monitor = EnergyMonitor()
                    monitor.start()
                    news_detections = classifier.predict(news_texts) if news_texts else []
                    energy = monitor.stop()
                    energy["task"] = "detection"
                    energy["timestamp"] = datetime.now(timezone.utc).isoformat()
                    energy["n_posts"] = len(news_texts)
                    EnergyMonitor.save_record(energy, EMISSIONS_FILE)
                    st.session_state.energy_results.append(energy)

                    results = []
                    news_idx = 0
                    for p in st.session_state.posts:
                        if p.get("is_news") and news_idx < len(news_detections):
                            results.append(news_detections[news_idx])
                            news_idx += 1
                        else:
                            results.append({"label": "CASUAL", "score": 1.0, "all_scores": {}})
                    st.session_state.detection_results = results
                except FileNotFoundError as e:
                    st.sidebar.error(str(e))
        else:
            st.sidebar.warning("Collect posts first.")

    if st.button("3. Analyze emotions", use_container_width=True):
        if st.session_state.posts:
            with st.spinner("VADER + trained model analysis..."):
                analyzer = get_emotion_analyzer()
                texts = [p.get("cleaned_text", p["text"]) for p in st.session_state.posts]
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
        else:
            st.sidebar.warning("Collect posts first.")

# Sidebar: model info
st.sidebar.markdown("---")
st.sidebar.subheader("Loaded models")
fn_metrics = load_model_metrics(FAKE_NEWS_MODEL_DIR)
emo_metrics = load_model_metrics(EMOTION_MODEL_DIR)
if fn_metrics:
    st.sidebar.success(f"Fake News: {fn_metrics['accuracy']:.1%} acc")
else:
    st.sidebar.warning("Fake News: not trained")
if emo_metrics:
    st.sidebar.success(f"Emotion: {emo_metrics['accuracy']:.1%} acc")
else:
    st.sidebar.warning("Emotion: not trained")


# ============================================================
# MAIN CONTENT
# ============================================================

# --- Header ---
st.title("Thumalien")
st.markdown(
    "Real-time **fake news detection** and **emotion analysis** pipeline "
    "on Bluesky posts. M1 Data & AI project with Green IT carbon tracking."
)

# --- Summary metrics ---
if st.session_state.posts:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Posts collected", len(st.session_state.posts))

    news_count = sum(1 for p in st.session_state.posts if p.get("is_news"))
    casual_count = len(st.session_state.posts) - news_count
    col2.metric("News-like", news_count, delta=f"{news_count/len(st.session_state.posts):.0%}" if st.session_state.posts else "0%")

    if st.session_state.detection_results:
        fake_count = sum(1 for r in st.session_state.detection_results if r["label"] == "FAKE")
        col3.metric("FAKE detected", fake_count, delta_color="inverse")
    else:
        col3.metric("FAKE detected", "—")

    col4.metric("Casual (skipped)", casual_count)

    if st.session_state.emotion_results:
        emotions = [e["bert_emotion"] for e in st.session_state.emotion_results if e.get("bert_emotion") != "unknown"]
        if emotions:
            dominant = Counter(emotions).most_common(1)[0]
            col5.metric("Dominant emotion", dominant[0].capitalize(), delta=f"{dominant[1]} posts")
        else:
            col5.metric("Dominant emotion", "—")
    else:
        col5.metric("Dominant emotion", "—")

    st.divider()

# --- Tabs ---
if st.session_state.posts:
    tab_overview, tab_detection, tab_emotion, tab_carbon, tab_db = st.tabs([
        "Overview",
        "Fake News Detection",
        "Emotion Analysis",
        "Carbon Footprint",
        "Database",
    ])
else:
    st.info(
        "Welcome! Click **Collect & Analyze** in the sidebar "
        "to run the full pipeline on Bluesky posts."
    )
    st.divider()
    tab_detection, tab_emotion, tab_carbon, tab_db = st.tabs([
        "Fake News Detection",
        "Emotion Analysis",
        "Carbon Footprint",
        "Database",
    ])
    tab_overview = None

# ============================================================
# TAB: OVERVIEW
# ============================================================
if st.session_state.posts and tab_overview is not None:
    with tab_overview:
        st.header("Collected posts overview")

        # Posts table
        df_posts = pd.DataFrame([
            {
                "Author": p["did"][-15:],
                "Text": p["text"][:150] + ("..." if len(p["text"]) > 150 else ""),
                "Lang": ", ".join(p.get("langs", [])),
                "Type": "News" if p.get("is_news") else "Casual",
                "Detection": (
                    st.session_state.detection_results[i]["label"]
                    if i < len(st.session_state.detection_results)
                    else "—"
                ),
                "Emotion": (
                    st.session_state.emotion_results[i].get("bert_emotion", "—").capitalize()
                    if i < len(st.session_state.emotion_results)
                    else "—"
                ),
            }
            for i, p in enumerate(st.session_state.posts)
        ])
        st.dataframe(
            df_posts,
            use_container_width=True,
            column_config={
                "Text": st.column_config.TextColumn(width="large"),
                "Type": st.column_config.TextColumn(width="small"),
                "Detection": st.column_config.TextColumn(width="small"),
                "Emotion": st.column_config.TextColumn(width="small"),
            },
        )

        # Two columns: word cloud + time series
        col_wc, col_ts = st.columns(2)

        with col_wc:
            st.subheader("Word cloud")
            all_text = " ".join(p.get("cleaned_text", p["text"]) for p in st.session_state.posts)
            words = all_text.split()
            if len(words) > 10:
                wc = WordCloud(
                    width=800, height=400,
                    background_color="white",
                    colormap="viridis",
                    max_words=100,
                    collocations=False,
                ).generate(all_text)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
                plt.close(fig_wc)
            else:
                st.caption("Not enough words for a word cloud.")

        with col_ts:
            st.subheader("Volume over time")
            timestamps = [p.get("created_at") for p in st.session_state.posts]
            valid_ts = [t for t in timestamps if t]
            if len(valid_ts) > 1:
                df_time = pd.DataFrame({"timestamp": pd.to_datetime(valid_ts)})
                df_time["count"] = 1
                df_time = df_time.set_index("timestamp").resample("1min").sum().reset_index()
                if len(df_time) > 1:
                    fig_time = px.area(
                        df_time, x="timestamp", y="count",
                        title="Posts per minute",
                        labels={"timestamp": "", "count": "Number of posts"},
                    )
                    fig_time.update_layout(showlegend=False)
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.caption("Collection too fast for a time chart.")
            else:
                st.caption("No timestamps available.")

# ============================================================
# TAB: FAKE NEWS DETECTION
# ============================================================
with tab_detection:
    st.header("Fake News Detection")

    st.markdown(
        "Posts are first **filtered** to identify news-like content (URLs, news keywords, claims). "
        "Only news-like posts are classified as REAL or FAKE. Casual posts are skipped."
    )

    # Model info
    fn_metrics = load_model_metrics(FAKE_NEWS_MODEL_DIR)
    if fn_metrics:
        st.markdown(
            f"**Model**: `{fn_metrics.get('base_model', 'distilbert')}` + LoRA | "
            f"**Accuracy**: {fn_metrics['accuracy']:.1%} | "
            f"**F1**: {fn_metrics['f1']:.1%} | "
            f"**Data**: {fn_metrics.get('train_size', '?'):,} train, {fn_metrics.get('val_size', '?'):,} val"
        )
    else:
        st.warning("Model not trained. Run: `python -m src.training.train_fake_news`")

    if st.session_state.detection_results:
        st.divider()

        # Distribution chart — 3 categories
        labels = [r["label"] for r in st.session_state.detection_results]
        label_counts = Counter(labels)

        col_chart, col_details = st.columns([1, 2])

        with col_chart:
            fig_pie = px.pie(
                names=list(label_counts.keys()),
                values=list(label_counts.values()),
                title="Distribution: REAL / FAKE / CASUAL",
                color=list(label_counts.keys()),
                color_discrete_map=LABEL_COLORS,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label+value")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_details:
            # Confidence distribution (news-like only)
            news_results = [r for r in st.session_state.detection_results if r["label"] != "CASUAL"]
            if news_results:
                scores = [r["score"] for r in news_results]
                fig_conf = px.histogram(
                    x=scores, nbins=20,
                    title="Confidence scores (news-like posts only)",
                    labels={"x": "Confidence score", "y": "Number of posts"},
                    color_discrete_sequence=["#3498db"],
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.info("No news-like posts detected in this batch.")

        # Detailed results
        st.subheader("Detail per post")
        for i, (post, det) in enumerate(zip(st.session_state.posts, st.session_state.detection_results)):
            if det["label"] == "FAKE":
                st.markdown(f'🔴 **FAKE** ({det["score"]:.0%}) — {post["text"][:120]}')
            elif det["label"] == "REAL":
                st.markdown(f'🟢 **REAL** ({det["score"]:.0%}) — {post["text"][:120]}')
            else:
                st.markdown(f'⚪ **CASUAL** — {post["text"][:120]}')
            if i >= 19:
                st.caption(f"... and {len(st.session_state.posts) - 20} more posts")
                break

    elif st.session_state.posts:
        st.info("Click **Collect & Analyze** to run detection.")

# ============================================================
# TAB: EMOTION ANALYSIS
# ============================================================
with tab_emotion:
    st.header("Emotion Analysis")

    # Model info
    emo_metrics = load_model_metrics(EMOTION_MODEL_DIR)
    if emo_metrics:
        st.markdown(
            f"**Model**: `{emo_metrics.get('base_model', 'distilbert')}` + LoRA | "
            f"**Accuracy**: {emo_metrics['accuracy']:.1%} | "
            f"**F1 macro**: {emo_metrics['f1_macro']:.1%} | "
            f"**Classes**: {', '.join(emo_metrics.get('labels', []))}"
        )
    else:
        st.warning("Model not trained. Run: `python -m src.training.train_emotion`")

    if st.session_state.emotion_results:
        df_emo = pd.DataFrame(st.session_state.emotion_results)
        st.divider()

        col_vader, col_bert = st.columns(2)

        # --- VADER ---
        with col_vader:
            st.subheader("VADER (baseline)")
            st.caption("Lexical sentiment analysis — positive/negative/neutral")

            avg_scores = {
                "Positive": df_emo["pos"].mean(),
                "Negative": df_emo["neg"].mean(),
                "Neutral": df_emo["neu"].mean(),
            }
            fig_vader_pie = px.pie(
                names=list(avg_scores.keys()),
                values=list(avg_scores.values()),
                title="Average sentiment (VADER)",
                color=list(avg_scores.keys()),
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Negative": "#e74c3c",
                    "Neutral": "#95a5a6",
                },
            )
            fig_vader_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_vader_pie, use_container_width=True)

            fig_compound = px.histogram(
                df_emo, x="compound", nbins=20,
                title="Compound score distribution",
                labels={"compound": "Compound (-1 = negative, +1 = positive)"},
                color_discrete_sequence=["#3498db"],
            )
            st.plotly_chart(fig_compound, use_container_width=True)

        # --- BERT ---
        with col_bert:
            has_bert = df_emo["bert_emotion"].iloc[0] != "unknown" if len(df_emo) > 0 else False

            if has_bert:
                st.subheader("Trained model (6 emotions)")
                st.caption("sadness, joy, love, anger, fear, surprise")

                emotion_colors = {
                    "sadness": "#636EFA", "joy": "#00CC96", "love": "#FF6692",
                    "anger": "#EF553B", "fear": "#AB63FA", "surprise": "#FFA15A",
                }

                # Radar chart
                all_bert_scores = [e.get("emotions_bert", {}) for e in st.session_state.emotion_results]
                if all_bert_scores and all_bert_scores[0]:
                    avg_emotions = {}
                    for scores in all_bert_scores:
                        for emotion, score in scores.items():
                            avg_emotions[emotion] = avg_emotions.get(emotion, 0) + score
                    avg_emotions = {k: v / len(all_bert_scores) for k, v in avg_emotions.items()}

                    emotions_list = list(avg_emotions.keys())
                    values = list(avg_emotions.values())
                    fig_radar = px.line_polar(
                        r=values + [values[0]],
                        theta=emotions_list + [emotions_list[0]],
                        title="Average emotion profile (radar)",
                    )
                    fig_radar.update_traces(fill="toself", fillcolor="rgba(99, 110, 250, 0.2)")
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # Histogram
                emotions = df_emo["bert_emotion"].tolist()
                fig_emotions = px.bar(
                    x=list(Counter(emotions).keys()),
                    y=list(Counter(emotions).values()),
                    title="Detected emotion distribution",
                    color=list(Counter(emotions).keys()),
                    color_discrete_map=emotion_colors,
                    labels={"x": "Emotion", "y": "Number of posts"},
                )
                fig_emotions.update_layout(showlegend=False)
                st.plotly_chart(fig_emotions, use_container_width=True)
            else:
                st.subheader("Trained model")
                st.warning("Emotion model not available — only VADER is used.")

        # Detail per post
        if has_bert:
            st.divider()
            st.subheader("Detail per post")
            emotion_icons = {
                "sadness": "😢", "joy": "😊", "love": "❤️",
                "anger": "😠", "fear": "😨", "surprise": "😲",
            }
            for i, (post, emo) in enumerate(zip(st.session_state.posts, st.session_state.emotion_results)):
                icon = emotion_icons.get(emo["bert_emotion"], "❓")
                compound = emo["compound"]
                compound_icon = "🟢" if compound > 0.05 else ("🔴" if compound < -0.05 else "⚪")
                st.markdown(
                    f'{icon} **{emo["bert_emotion"].capitalize()}** ({emo["bert_score"]:.0%}) '
                    f'{compound_icon} VADER: {compound:+.2f} — '
                    f'{post["text"][:100]}'
                )
                if i >= 14:
                    st.caption(f"... and {len(st.session_state.posts) - 15} more posts")
                    break

    elif st.session_state.posts:
        st.info("Click **Collect & Analyze** to run emotion analysis.")

# ============================================================
# TAB: CARBON FOOTPRINT
# ============================================================
with tab_carbon:
    st.header("Carbon Footprint (Green IT)")
    st.markdown(
        "Energy consumption and CO2 emissions tracking via **CodeCarbon**. "
        "Each analysis step (detection, emotion, training) is measured."
    )

    if st.session_state.energy_results:
        st.subheader("Current session")
        cols_energy = st.columns(len(st.session_state.energy_results))
        for col, e in zip(cols_energy, st.session_state.energy_results):
            col.metric(
                f"{e['task'].capitalize()}",
                f"{e['emissions_kg'] * 1000:.4f} g CO2",
                delta=f"{e['duration_s']:.1f}s | {e.get('n_posts', '?')} posts",
                delta_color="off",
            )
        st.divider()

    st.subheader("Emissions history")
    if os.path.exists(EMISSIONS_FILE):
        with open(EMISSIONS_FILE) as f:
            try:
                all_emissions = json.load(f)
            except json.JSONDecodeError:
                all_emissions = []

        if all_emissions:
            df_emissions = pd.DataFrame(all_emissions)
            total_co2 = df_emissions["emissions_kg"].sum()
            total_energy = df_emissions.get("energy_kwh", pd.Series([0])).sum()
            total_duration = df_emissions["duration_s"].sum()

            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            col_t1.metric("Total CO2", f"{total_co2 * 1000:.4f} g")
            col_t2.metric("Total energy", f"{total_energy * 1000:.4f} Wh")
            col_t3.metric("Total time", f"{total_duration:.0f}s")
            col_t4.metric("Analyses count", len(all_emissions))

            if "task" in df_emissions.columns and "timestamp" in df_emissions.columns:
                fig_carbon = px.bar(
                    df_emissions,
                    x="timestamp",
                    y=df_emissions["emissions_kg"] * 1000,
                    color="task",
                    title="Emissions per analysis (g CO2eq)",
                    labels={"y": "Emissions (g CO2eq)", "timestamp": ""},
                )
                st.plotly_chart(fig_carbon, use_container_width=True)

                if len(df_emissions["task"].unique()) > 1:
                    fig_by_task = px.pie(
                        df_emissions.groupby("task")["emissions_kg"].sum().reset_index(),
                        names="task", values="emissions_kg",
                        title="Emissions by task type",
                    )
                    st.plotly_chart(fig_by_task, use_container_width=True)
        else:
            st.info("No emission data recorded yet.")
    else:
        st.info("No emission data. Run an analysis to start tracking.")

    st.markdown(
        "> **Note**: Emissions are calculated by CodeCarbon based on "
        "your hardware and local energy mix."
    )

# ============================================================
# TAB: DATABASE
# ============================================================
with tab_db:
    st.header("PostgreSQL Database")
    st.caption("Global database statistics")

    try:
        repo = get_repository()
        stats = repo.get_stats()

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Total posts in DB", f"{stats['total_posts']:,}")
        col_s2.metric("Detection analyses", f"{stats['total_detections']:,}")
        col_s3.metric("Emotion analyses", f"{stats['total_emotions']:,}")

        if stats["label_distribution"]:
            st.subheader("Global label distribution (DB)")
            fig_dist = px.bar(
                x=list(stats["label_distribution"].keys()),
                y=list(stats["label_distribution"].values()),
                title="Labels in database",
                color=list(stats["label_distribution"].keys()),
                color_discrete_map=LABEL_COLORS,
                labels={"x": "Label", "y": "Count"},
            )
            fig_dist.update_layout(showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
    except Exception as e:
        st.warning(f"Database not available: {e}")
        st.caption("Check that PostgreSQL is running: `docker-compose up -d`")

# --- Footer ---
st.markdown("---")
st.caption("Thumalien — M1 Data & AI Project | Fake news detection on Bluesky | Green IT")
