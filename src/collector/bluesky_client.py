"""Client de collecte de posts Bluesky via AT Protocol."""

from loguru import logger


class BlueskyCollector:
    """Collecte les posts depuis Bluesky."""

    def __init__(self, handle: str, password: str):
        self.handle = handle
        self.password = password
        self.client = None

    def connect(self):
        """Établit la connexion au serveur Bluesky."""
        logger.info("Connexion à Bluesky pour {}", self.handle)
        # TODO: initialiser atproto.Client et s'authentifier
        raise NotImplementedError

    def fetch_recent_posts(self, query: str, limit: int = 50) -> list[dict]:
        """Récupère les posts récents correspondant à une requête.

        Args:
            query: Terme de recherche.
            limit: Nombre maximum de posts à récupérer.

        Returns:
            Liste de posts sous forme de dictionnaires.
        """
        logger.info("Collecte de {} posts pour '{}'", limit, query)
        # TODO: appeler l'API de recherche AT Protocol
        raise NotImplementedError
