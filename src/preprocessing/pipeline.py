"""Pipeline de prétraitement : nettoyage, tokenisation, embeddings."""

import re

from loguru import logger


class PreprocessingPipeline:
    """Nettoie et transforme les posts bruts."""

    def __init__(self):
        self.nlp = None  # Pipeline spaCy, chargé à la demande

    def clean_text(self, text: str) -> str:
        """Supprime URLs, mentions, caractères spéciaux.

        Args:
            text: Texte brut du post.

        Returns:
            Texte nettoyé.
        """
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@[\w.]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenise le texte avec spaCy.

        Args:
            text: Texte nettoyé.

        Returns:
            Liste de tokens.
        """
        logger.debug("Tokenisation du texte ({} caractères)", len(text))
        # TODO: charger le modèle spaCy et tokeniser
        raise NotImplementedError

    def compute_embeddings(self, texts: list[str]):
        """Calcule les embeddings via un modèle Transformer.

        Args:
            texts: Liste de textes nettoyés.

        Returns:
            Matrice d'embeddings.
        """
        logger.info("Calcul des embeddings pour {} textes", len(texts))
        # TODO: utiliser transformers pour encoder les textes
        raise NotImplementedError
