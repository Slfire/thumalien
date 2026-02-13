"""Configuration centralisée du projet Thumalien."""

import os
from dotenv import load_dotenv

load_dotenv()

# Jetstream (Bluesky firehose)
JETSTREAM_URL = os.getenv(
    "JETSTREAM_URL", "wss://jetstream1.us-east.bsky.network/subscribe"
)
JETSTREAM_MAX_POSTS = int(os.getenv("JETSTREAM_MAX_POSTS", "1000"))
JETSTREAM_LANG_FILTER = os.getenv("JETSTREAM_LANG_FILTER", "en")
JETSTREAM_RAW_DIR = os.getenv("JETSTREAM_RAW_DIR", "data/raw")

# Base de données
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://thumalien:thumalien@localhost:5432/thumalien"
)

# Modèles
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))

# Modèles fine-tunés
FAKE_NEWS_MODEL_DIR = os.getenv("FAKE_NEWS_MODEL_DIR", "models/fake_news_model")
EMOTION_MODEL_DIR = os.getenv("EMOTION_MODEL_DIR", "models/emotion_model")

# Entraînement
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "16"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))

# Datasets Kaggle
FAKE_NEWS_DATA_DIR = os.getenv("FAKE_NEWS_DATA_DIR", "data/kaggle/fake-news")
EMOTION_DATA_DIR = os.getenv("EMOTION_DATA_DIR", "data/kaggle/emotion")

# CodeCarbon
CODECARBON_COUNTRY = os.getenv("CODECARBON_COUNTRY", "FRA")
EMISSIONS_FILE = os.getenv("EMISSIONS_FILE", "data/emissions.json")
