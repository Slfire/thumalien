"""Analyse émotionnelle (VADER baseline + modèle fine-tuné)."""

import os

from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import EMOTION_MODEL_DIR

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


class EmotionAnalyzer:
    """Analyse les émotions dans les posts (VADER + modèle entraîné)."""

    def __init__(self, model_dir: str = EMOTION_MODEL_DIR):
        self.vader = SentimentIntensityAnalyzer()
        self.model_dir = model_dir
        self._pipeline = None
        self._model_available = os.path.exists(
            os.path.join(model_dir, "adapter_config.json")
        )

    @property
    def pipeline(self):
        """Charge le pipeline émotion à la demande."""
        if self._pipeline is None and self._model_available:
            self._load_model()
        return self._pipeline

    def _load_model(self):
        """Charge le modèle fine-tuné d'émotion."""
        from peft import AutoPeftModelForSequenceClassification
        from transformers import AutoConfig, pipeline as hf_pipeline

        logger.info("Chargement du modèle émotion: {}", self.model_dir)
        label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
        id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=len(EMOTION_LABELS),
            id2label=id2label,
            label2id=label2id,
        )
        model.eval()

        self._pipeline = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=self.model_dir,
            device=-1,
            top_k=None,
        )
        logger.info("Modèle émotion chargé")

    def analyze_vader(self, text: str) -> dict:
        """Analyse de sentiment avec VADER (baseline).

        Args:
            text: Texte à analyser.

        Returns:
            Scores de sentiment (neg, neu, pos, compound).
        """
        return self.vader.polarity_scores(text)

    def analyze_bert(self, texts: list[str]) -> list[dict]:
        """Analyse émotionnelle avec le modèle fine-tuné.

        Args:
            texts: Liste de textes.

        Returns:
            Liste de dicts avec emotion, score, all_scores.
        """
        if not self._model_available:
            logger.warning(
                "Modèle émotion non disponible dans {}. "
                "Lancez : python -m src.training.train_emotion",
                self.model_dir,
            )
            return [{"emotion": "unknown", "score": 0.0, "all_scores": {}}] * len(texts)

        logger.info("Analyse émotion sur {} textes", len(texts))
        if not texts:
            return []

        batch_results = self.pipeline(texts)

        if texts and isinstance(batch_results[0], dict):
            batch_results = [batch_results]

        output = []
        for results in batch_results:
            all_scores = {r["label"]: r["score"] for r in results}
            best = max(results, key=lambda x: x["score"])
            output.append({
                "emotion": best["label"],
                "score": best["score"],
                "all_scores": all_scores,
            })
        return output

    def analyze_full(self, text: str) -> dict:
        """Analyse complète : VADER + modèle entraîné.

        Returns:
            Dict avec 'vader' et 'bert' (None si modèle absent).
        """
        result = {"vader": self.analyze_vader(text)}
        if self._model_available:
            result["bert"] = self.analyze_bert([text])[0]
        else:
            result["bert"] = None
        return result
