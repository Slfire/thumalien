"""Repository pattern pour les opérations CRUD."""

from loguru import logger
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_URL
from src.database.models import Base, DetectionResult, EmotionResult, Post


class Repository:
    """Encapsule toutes les opérations base de données."""

    def __init__(self, database_url: str | None = None):
        self.engine = create_engine(database_url or DATABASE_URL)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Crée toutes les tables (idempotent)."""
        Base.metadata.create_all(self.engine)
        logger.info("Tables créées/vérifiées")

    def save_post(self, post_dict: dict) -> Post:
        """Sauvegarde un post. Retourne l'existant si (did, rkey) déjà présent."""
        with self.Session() as session:
            existing = session.execute(
                select(Post).where(
                    Post.did == post_dict["did"], Post.rkey == post_dict["rkey"]
                )
            ).scalar_one_or_none()

            if existing:
                return existing

            post = Post(
                did=post_dict["did"],
                rkey=post_dict["rkey"],
                text=post_dict["text"],
                cleaned_text=post_dict.get("cleaned_text"),
                created_at=post_dict.get("created_at"),
                langs=post_dict.get("langs"),
                collected_at=post_dict.get("collected_at"),
            )
            session.add(post)
            session.commit()
            session.refresh(post)
            return post

    def save_detection_result(
        self, post_id: int, label: str, score: float, model_name: str
    ) -> DetectionResult:
        """Sauvegarde un résultat de détection."""
        with self.Session() as session:
            result = DetectionResult(
                post_id=post_id,
                label=label,
                score=score,
                model_name=model_name,
            )
            session.add(result)
            session.commit()
            session.refresh(result)
            return result

    def save_emotion_result(
        self,
        post_id: int,
        compound: float,
        positive: float,
        negative: float,
        neutral: float,
        emotions_bert: dict | None = None,
    ) -> EmotionResult:
        """Sauvegarde un résultat d'analyse émotionnelle."""
        with self.Session() as session:
            result = EmotionResult(
                post_id=post_id,
                compound=compound,
                positive=positive,
                negative=negative,
                neutral=neutral,
                emotions_bert=emotions_bert,
            )
            session.add(result)
            session.commit()
            session.refresh(result)
            return result

    def get_recent_posts(self, limit: int = 50) -> list[Post]:
        """Récupère les posts les plus récents avec leurs résultats."""
        with self.Session() as session:
            posts = (
                session.execute(select(Post).order_by(Post.id.desc()).limit(limit))
                .scalars()
                .all()
            )
            for p in posts:
                _ = p.detection_results
                _ = p.emotion_results
            session.expunge_all()
            return posts

    def get_stats(self) -> dict:
        """Retourne les statistiques agrégées."""
        with self.Session() as session:
            total_posts = session.execute(select(func.count(Post.id))).scalar() or 0
            total_detections = (
                session.execute(select(func.count(DetectionResult.id))).scalar() or 0
            )
            total_emotions = (
                session.execute(select(func.count(EmotionResult.id))).scalar() or 0
            )

            label_counts = {}
            if total_detections > 0:
                rows = session.execute(
                    select(DetectionResult.label, func.count(DetectionResult.id)).group_by(
                        DetectionResult.label
                    )
                ).all()
                label_counts = {row[0]: row[1] for row in rows}

            return {
                "total_posts": total_posts,
                "total_detections": total_detections,
                "total_emotions": total_emotions,
                "label_distribution": label_counts,
            }
