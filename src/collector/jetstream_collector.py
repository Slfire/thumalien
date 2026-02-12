"""Collecteur de posts Bluesky via Jetstream WebSocket."""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Callable

import aiofiles
import websockets
from loguru import logger

from src.config import (
    JETSTREAM_LANG_FILTER,
    JETSTREAM_MAX_POSTS,
    JETSTREAM_RAW_DIR,
    JETSTREAM_URL,
)


class JetstreamCollector:
    """Collecte les posts Bluesky en temps réel via Jetstream WebSocket."""

    WANTED_COLLECTION = "app.bsky.feed.post"

    def __init__(
        self,
        endpoint: str = JETSTREAM_URL,
        lang_filter: str = JETSTREAM_LANG_FILTER,
        raw_dir: str = JETSTREAM_RAW_DIR,
    ):
        self.endpoint = endpoint
        self.lang_filter = (
            [lang.strip() for lang in lang_filter.split(",") if lang.strip()]
            if lang_filter
            else []
        )
        self.raw_dir = raw_dir
        self._ws = None
        self._running = False

    @property
    def ws_url(self) -> str:
        """URL WebSocket complète avec paramètres de filtrage."""
        return f"{self.endpoint}?wantedCollections={self.WANTED_COLLECTION}"

    async def connect(self):
        """Ouvre la connexion WebSocket vers Jetstream."""
        logger.info("Connexion Jetstream : {}", self.ws_url)
        self._ws = await websockets.connect(
            self.ws_url,
            ping_interval=30,
            ping_timeout=10,
        )
        self._running = True
        logger.info("Connecté à Jetstream")

    async def disconnect(self):
        """Ferme proprement la connexion WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("Déconnecté de Jetstream")

    def _parse_message(self, raw: str) -> dict | None:
        """Parse un message Jetstream en dict normalisé.

        Retourne None si le message doit être ignoré (pas un create,
        mauvaise collection, ou filtré par langue).
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Message JSON invalide ignoré")
            return None

        if msg.get("kind") != "commit":
            return None

        commit = msg.get("commit", {})
        if commit.get("operation") != "create":
            return None
        if commit.get("collection") != self.WANTED_COLLECTION:
            return None

        record = commit.get("record", {})
        text = record.get("text", "")
        langs = record.get("langs", [])

        if self.lang_filter:
            if not any(lang in self.lang_filter for lang in langs):
                return None

        return {
            "did": msg.get("did", ""),
            "rkey": commit.get("rkey", ""),
            "text": text,
            "created_at": record.get("createdAt", ""),
            "langs": langs,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "raw": msg,
        }

    async def _write_raw(self, writer, post: dict):
        """Écrit un post dans le fichier JSONL."""
        line = json.dumps(post["raw"], ensure_ascii=False) + "\n"
        await writer.write(line)

    async def listen(
        self,
        callback: Callable[[dict], None],
        max_posts: int = 0,
        save_raw: bool = True,
    ):
        """Stream les posts en continu, appelle callback pour chacun.

        Args:
            callback: Appelé avec chaque post normalisé.
            max_posts: Arrêter après N posts (0 = illimité).
            save_raw: Sauvegarder les messages bruts en JSONL.
        """
        count = 0
        backoff = 1
        raw_writer = None

        if save_raw:
            os.makedirs(self.raw_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.raw_dir, f"jetstream_{ts}.jsonl")
            raw_writer = await aiofiles.open(path, "a")
            logger.info("Sauvegarde brute -> {}", path)

        try:
            while self._running:
                try:
                    await self.connect()
                    backoff = 1

                    async for message in self._ws:
                        post = self._parse_message(message)
                        if post is None:
                            continue

                        callback(post)
                        count += 1

                        if raw_writer:
                            await self._write_raw(raw_writer, post)

                        if max_posts > 0 and count >= max_posts:
                            logger.info("Limite atteinte : {} posts", count)
                            self._running = False
                            break

                except websockets.ConnectionClosed as e:
                    if not self._running:
                        break
                    logger.warning(
                        "Connexion perdue : {}. Reconnexion dans {}s...", e, backoff
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)

                except Exception as e:
                    if not self._running:
                        break
                    logger.error(
                        "Erreur inattendue : {}. Reconnexion dans {}s...", e, backoff
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
        finally:
            if raw_writer:
                await raw_writer.close()
            await self.disconnect()

    async def collect_batch(self, n: int | None = None) -> list[dict]:
        """Collecte exactement n posts et les retourne.

        Args:
            n: Nombre de posts. Par défaut JETSTREAM_MAX_POSTS.

        Returns:
            Liste de posts normalisés.
        """
        if n is None:
            n = JETSTREAM_MAX_POSTS

        posts: list[dict] = []

        def _accumulate(post: dict):
            posts.append(post)

        self._running = True
        await self.listen(callback=_accumulate, max_posts=n, save_raw=True)
        return posts

    def collect_posts_sync(self, n: int | None = None) -> list[dict]:
        """Wrapper synchrone autour de collect_batch.

        Pour usage dans les scripts, notebooks et modules non-async.
        """
        return asyncio.run(self.collect_batch(n))
