"""Classification fake news via modèle fine-tuné."""

import os

from loguru import logger

from src.config import FAKE_NEWS_MODEL_DIR

DEFAULT_LABELS = ["REAL", "FAKE"]


class FakeNewsClassifier:
    """Classifie les posts comme REAL ou FAKE via le modèle fine-tuné."""

    def __init__(
        self,
        model_dir: str = FAKE_NEWS_MODEL_DIR,
        labels: list[str] | None = None,
    ):
        self.model_dir = model_dir
        self.model_name = f"fine-tuned:{os.path.basename(model_dir)}"
        self.labels = labels or DEFAULT_LABELS
        self._pipeline = None

    @property
    def pipeline(self):
        """Charge le pipeline de classification à la demande."""
        if self._pipeline is None:
            self.load_model()
        return self._pipeline

    def load_model(self):
        """Charge le modèle fine-tuné depuis le disque."""
        from peft import AutoPeftModelForSequenceClassification
        from transformers import pipeline as hf_pipeline

        if not os.path.exists(os.path.join(self.model_dir, "adapter_config.json")):
            raise FileNotFoundError(
                f"Modèle fine-tuné introuvable dans {self.model_dir}. "
                "Lancez d'abord : python -m src.training.train_fake_news"
            )

        logger.info("Chargement du modèle fine-tuné: {}", self.model_dir)
        model = AutoPeftModelForSequenceClassification.from_pretrained(self.model_dir)
        model.eval()

        self._pipeline = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=self.model_dir,
            device=-1,
            top_k=None,
        )
        logger.info("Modèle chargé")

    def predict_single(self, text: str) -> dict:
        """Prédit le label pour un texte.

        Returns:
            Dict avec label, score, all_scores.
        """
        results = self.pipeline(text)
        all_scores = {r["label"]: r["score"] for r in results}
        best = max(results, key=lambda x: x["score"])
        return {
            "label": best["label"],
            "score": best["score"],
            "all_scores": all_scores,
        }

    def predict(self, texts: list[str]) -> list[dict]:
        """Prédit les labels pour une liste de textes.

        Args:
            texts: Liste de textes prétraités.

        Returns:
            Liste de dicts avec label, score, all_scores.
        """
        logger.info("Prédiction sur {} textes", len(texts))
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
                "label": best["label"],
                "score": best["score"],
                "all_scores": all_scores,
            })
        return output
