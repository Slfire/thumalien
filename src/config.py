"""Configuration centralisée du projet Thumalien."""

import os
from dotenv import load_dotenv

load_dotenv()

# Bluesky
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE", "")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD", "")

# Base de données
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://thumalien:thumalien@localhost:5432/thumalien"
)

# Modèle
MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-multilingual-cased")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))

# CodeCarbon
CODECARBON_COUNTRY = os.getenv("CODECARBON_COUNTRY", "FRA")
