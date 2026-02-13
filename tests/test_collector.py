"""Tests pour le module collector (Jetstream)."""

import json

from src.collector.jetstream_collector import JetstreamCollector


def _make_message(text="Hello world", langs=None, operation="create", kind="commit"):
    """Construit un message Jetstream synthétique."""
    if langs is None:
        langs = ["en"]
    return json.dumps({
        "did": "did:plc:test123",
        "time_us": 1700000000000000,
        "kind": kind,
        "commit": {
            "operation": operation,
            "collection": "app.bsky.feed.post",
            "rkey": "abc123",
            "record": {
                "text": text,
                "createdAt": "2025-11-14T12:00:00.000Z",
                "langs": langs,
            },
        },
    })


def test_collector_init_defaults():
    collector = JetstreamCollector()
    assert "jetstream" in collector.endpoint
    assert collector._ws is None
    assert collector._running is False


def test_collector_init_custom():
    collector = JetstreamCollector(
        endpoint="wss://custom.endpoint/subscribe",
        lang_filter="en,fr",
    )
    assert collector.endpoint == "wss://custom.endpoint/subscribe"
    assert collector.lang_filter == ["en", "fr"]


def test_ws_url_includes_collection():
    collector = JetstreamCollector()
    assert "wantedCollections=app.bsky.feed.post" in collector.ws_url


def test_parse_valid_create():
    collector = JetstreamCollector()
    msg = _make_message(text="This is a test", langs=["en"])
    result = collector._parse_message(msg)
    assert result is not None
    assert result["did"] == "did:plc:test123"
    assert result["rkey"] == "abc123"
    assert result["text"] == "This is a test"
    assert result["langs"] == ["en"]
    assert "collected_at" in result
    assert "raw" in result


def test_parse_delete_returns_none():
    collector = JetstreamCollector()
    msg = _make_message(operation="delete")
    assert collector._parse_message(msg) is None


def test_parse_update_returns_none():
    collector = JetstreamCollector()
    msg = _make_message(operation="update")
    assert collector._parse_message(msg) is None


def test_parse_non_commit_returns_none():
    collector = JetstreamCollector()
    msg = _make_message(kind="identity")
    assert collector._parse_message(msg) is None


def test_parse_lang_filter_accepts():
    collector = JetstreamCollector(lang_filter="en")
    msg = _make_message(langs=["en"])
    assert collector._parse_message(msg) is not None


def test_parse_lang_filter_rejects():
    collector = JetstreamCollector(lang_filter="en")
    msg = _make_message(langs=["fr"])
    assert collector._parse_message(msg) is None


def test_parse_lang_filter_empty_accepts_all():
    collector = JetstreamCollector(lang_filter="")
    msg = _make_message(langs=["ja"])
    assert collector._parse_message(msg) is not None


def test_parse_malformed_json():
    collector = JetstreamCollector()
    assert collector._parse_message("not json at all") is None
