"""Tests pour le module preprocessing."""

from src.preprocessing.pipeline import PreprocessingPipeline


def test_clean_text_removes_urls():
    pipeline = PreprocessingPipeline()
    result = pipeline.clean_text("Voir https://example.com pour plus d'info")
    assert "https" not in result


def test_clean_text_removes_mentions():
    pipeline = PreprocessingPipeline()
    result = pipeline.clean_text("Hello @user.bsky.social !")
    assert "@" not in result
