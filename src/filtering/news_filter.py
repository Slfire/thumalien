"""Filtre pour separer les posts news-like des posts casual."""

import re

from loguru import logger

# Keywords indicating news-like content
NEWS_KEYWORDS = [
    "breaking", "report", "reports", "reported", "reporting",
    "according to", "sources say", "source says",
    "claim", "claims", "claimed",
    "announced", "announces", "announcement",
    "official", "officials", "spokesperson",
    "investigation", "investigators",
    "confirmed", "confirms",
    "breaking news", "just in", "developing",
    "study finds", "study shows", "research shows",
    "government", "president", "minister",
    "election", "vote", "ballot",
    "arrested", "charged", "convicted",
    "scandal", "controversy", "crisis",
    "leaked", "exposed", "revealed",
    "warning", "alert", "emergency",
    "exclusive", "update",
]

URL_PATTERN = re.compile(r"https?://\S+")


class NewsFilter:
    """Filtre les posts pour identifier le contenu news-like vs casual."""

    def __init__(self, min_news_length: int = 60, keywords: list[str] | None = None):
        self.min_news_length = min_news_length
        self.keywords = [kw.lower() for kw in (keywords or NEWS_KEYWORDS)]

    def is_news_like(self, text: str) -> bool:
        """Determine si un post ressemble a du contenu news.

        Criteres :
        1. Contient une URL + mots-cles news
        2. Long + mots-cles news
        3. Citations + URL (partage de source)

        Args:
            text: Texte du post.

        Returns:
            True si le post est news-like.
        """
        text_lower = text.lower()

        has_url = bool(URL_PATTERN.search(text))
        has_news_kw = any(kw in text_lower for kw in self.keywords)
        has_quotes = any(c in text for c in ['"', '\u201c', '\u201d', '\u2018', '\u2019'])
        is_long = len(text) >= self.min_news_length

        # URL + news keywords = strong signal
        if has_url and has_news_kw:
            return True

        # Long post + news keywords
        if is_long and has_news_kw:
            return True

        # Quotes + URL = citing a source
        if has_quotes and has_url:
            return True

        return False

    def filter_posts(self, posts: list[dict]) -> list[dict]:
        """Filtre une liste de posts et ajoute le champ 'is_news'.

        Args:
            posts: Liste de dicts avec au moins 'text'.

        Returns:
            La meme liste avec 'is_news' ajoute a chaque post.
        """
        for post in posts:
            text = post.get("cleaned_text", post.get("text", ""))
            post["is_news"] = self.is_news_like(text)

        news_count = sum(1 for p in posts if p["is_news"])
        logger.info(
            "Filtre news : {}/{} posts identifies comme news-like",
            news_count,
            len(posts),
        )
        return posts
