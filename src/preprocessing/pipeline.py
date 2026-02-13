"""Pipeline de prétraitement : nettoyage, tokenisation, embeddings."""

import re

from loguru import logger

from src.config import EMBEDDING_MODEL


class PreprocessingPipeline:
    """Nettoie et transforme les posts bruts."""

    def __init__(self):
        self._nlp = None
        self._embedder = None

    @property
    def nlp(self):
        """Charge le modèle spaCy à la demande."""
        if self._nlp is None:
            import spacy

            logger.info("Chargement du modèle spaCy en_core_web_sm")
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    @property
    def embedder(self):
        """Charge le modèle sentence-transformers à la demande."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Chargement du modèle sentence-transformers: {}", EMBEDDING_MODEL)
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

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
            Liste de lemmes (lowercase, sans stopwords ni ponctuation).
        """
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return tokens

    def compute_embeddings(self, texts: list[str]):
        """Calcule les embeddings via sentence-transformers.

        Args:
            texts: Liste de textes nettoyés.

        Returns:
            Matrice numpy (len(texts), embedding_dim).
        """
        logger.info("Calcul des embeddings pour {} textes", len(texts))
        return self.embedder.encode(texts, show_progress_bar=False)

    def process_post(self, post_dict: dict) -> dict:
        """Traite un post du collecteur et l'enrichit.

        Args:
            post_dict: Dict normalisé du collecteur (did, rkey, text, ...).

        Returns:
            Dict enrichi avec cleaned_text, tokens, embedding.
        """
        enriched = dict(post_dict)
        cleaned = self.clean_text(enriched["text"])
        enriched["cleaned_text"] = cleaned
        enriched["tokens"] = self.tokenize(cleaned)
        enriched["embedding"] = self.compute_embeddings([cleaned])[0]
        return enriched
