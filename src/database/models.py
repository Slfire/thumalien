"""Modèles SQLAlchemy pour Thumalien."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    did = Column(String(255), nullable=False, index=True)
    rkey = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    created_at = Column(String(50), nullable=True)
    langs = Column(JSON, nullable=True)
    collected_at = Column(String(50), nullable=True)

    detection_results = relationship(
        "DetectionResult", back_populates="post", cascade="all, delete-orphan"
    )
    emotion_results = relationship(
        "EmotionResult", back_populates="post", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("did", "rkey", name="uq_post_did_rkey"),
    )


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(
        Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)
    model_name = Column(String(255), nullable=False)
    analyzed_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    post = relationship("Post", back_populates="detection_results")


class EmotionResult(Base):
    __tablename__ = "emotion_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(
        Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    compound = Column(Float, nullable=True)
    positive = Column(Float, nullable=True)
    negative = Column(Float, nullable=True)
    neutral = Column(Float, nullable=True)
    emotions_bert = Column(JSON, nullable=True)
    analyzed_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    post = relationship("Post", back_populates="emotion_results")
