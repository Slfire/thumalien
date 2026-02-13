"""Tests pour le module database (SQLite in-memory)."""

from src.database.repository import Repository


def _make_repo():
    repo = Repository(database_url="sqlite:///:memory:")
    repo.create_tables()
    return repo


def _sample_post():
    return {
        "did": "did:plc:test123",
        "rkey": "abc123",
        "text": "This is a test post",
        "cleaned_text": "This is a test post",
        "created_at": "2026-02-12T09:13:07.209Z",
        "langs": ["en"],
        "collected_at": "2026-02-12T09:15:00.000Z",
    }


def test_save_and_retrieve_post():
    repo = _make_repo()
    post = repo.save_post(_sample_post())
    assert post.id is not None
    assert post.did == "did:plc:test123"
    assert post.text == "This is a test post"


def test_duplicate_post_returns_existing():
    repo = _make_repo()
    p1 = repo.save_post(_sample_post())
    p2 = repo.save_post(_sample_post())
    assert p1.id == p2.id


def test_get_recent_posts():
    repo = _make_repo()
    repo.save_post(_sample_post())
    posts = repo.get_recent_posts(limit=10)
    assert len(posts) == 1


def test_save_detection_result():
    repo = _make_repo()
    post = repo.save_post(_sample_post())
    result = repo.save_detection_result(post.id, "information fiable", 0.85, "xlm-roberta")
    assert result.label == "information fiable"
    assert result.score == 0.85


def test_save_emotion_result():
    repo = _make_repo()
    post = repo.save_post(_sample_post())
    result = repo.save_emotion_result(
        post.id, compound=0.5, positive=0.7, negative=0.1, neutral=0.2
    )
    assert result.compound == 0.5


def test_stats_empty_db():
    repo = _make_repo()
    stats = repo.get_stats()
    assert stats["total_posts"] == 0


def test_stats_with_data():
    repo = _make_repo()
    post = repo.save_post(_sample_post())
    repo.save_detection_result(post.id, "désinformation", 0.9, "xlm-roberta")
    stats = repo.get_stats()
    assert stats["total_posts"] == 1
    assert stats["total_detections"] == 1
    assert stats["label_distribution"]["désinformation"] == 1
