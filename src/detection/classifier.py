"""Classification fake news et score de crédibilité."""

from loguru import logger


class FakeNewsClassifier:
    """Classifie les posts comme fiables ou fake news."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """Charge le modèle de classification."""
        logger.info("Chargement du modèle de détection")
        # TODO: charger le modèle (scikit-learn ou transformers)
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[dict]:
        """Prédit la crédibilité d'une liste de textes.

        Args:
            texts: Liste de textes prétraités.

        Returns:
            Liste de dicts avec 'label' et 'score'.
        """
        logger.info("Prédiction sur {} textes", len(texts))
        # TODO: inférence du modèle
        raise NotImplementedError
