"""Module de persistance PostgreSQL pour Thumalien."""

from src.database.models import Base, Post, DetectionResult, EmotionResult
from src.database.repository import Repository

__all__ = ["Base", "Post", "DetectionResult", "EmotionResult", "Repository"]
