"""Tests pour le module collector."""

from src.collector.bluesky_client import BlueskyCollector


def test_collector_init():
    collector = BlueskyCollector(handle="test.bsky.social", password="test")
    assert collector.handle == "test.bsky.social"
    assert collector.client is None
