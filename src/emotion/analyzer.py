"""Analyse émotionnelle (VADER baseline + BERT)."""

from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class EmotionAnalyzer:
    """Analyse les émotions dans les posts."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze_vader(self, text: str) -> dict:
        """Analyse de sentiment avec VADER (baseline).

        Args:
            text: Texte à analyser.

        Returns:
            Scores de sentiment (neg, neu, pos, compound).
        """
        return self.vader.polarity_scores(text)

    def analyze_bert(self, texts: list[str]) -> list[dict]:
        """Analyse émotionnelle fine avec un modèle BERT.

        Args:
            texts: Liste de textes.

        Returns:
            Liste de dicts avec les émotions détectées.
        """
        logger.info("Analyse BERT sur {} textes", len(texts))
        # TODO: charger un modèle d'émotion BERT et prédire
        raise NotImplementedError
