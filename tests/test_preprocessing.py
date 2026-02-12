"""Tests pour le module preprocessing."""

from unittest.mock import MagicMock

import numpy as np

from src.preprocessing.pipeline import PreprocessingPipeline


def test_clean_text_removes_urls():
    pipeline = PreprocessingPipeline()
    result = pipeline.clean_text("Voir https://example.com pour plus d'info")
    assert "https" not in result


def test_clean_text_removes_mentions():
    pipeline = PreprocessingPipeline()
    result = pipeline.clean_text("Hello @user.bsky.social !")
    assert "@" not in result


def test_clean_text_normalizes_whitespace():
    pipeline = PreprocessingPipeline()
    result = pipeline.clean_text("Beaucoup   d'espaces    ici")
    assert "  " not in result


def _make_mock_nlp():
    mock_nlp = MagicMock()
    token_le = MagicMock(lemma_="le", is_stop=True, is_punct=False, is_space=False)
    token_chat = MagicMock(lemma_="chat", is_stop=False, is_punct=False, is_space=False)
    token_mange = MagicMock(lemma_="manger", is_stop=False, is_punct=False, is_space=False)
    mock_nlp.return_value = [token_le, token_chat, token_mange]
    return mock_nlp


def test_tokenize_filters_stopwords():
    pipeline = PreprocessingPipeline()
    pipeline._nlp = _make_mock_nlp()
    tokens = pipeline.tokenize("Le chat mange")
    assert "le" not in tokens
    assert "chat" in tokens
    assert "manger" in tokens


def test_tokenize_returns_lowercased_lemmas():
    pipeline = PreprocessingPipeline()
    token = MagicMock(lemma_="Manger", is_stop=False, is_punct=False, is_space=False)
    pipeline._nlp = MagicMock(return_value=[token])
    tokens = pipeline.tokenize("Manger")
    assert tokens == ["manger"]


def test_compute_embeddings_shape():
    pipeline = PreprocessingPipeline()
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.rand(2, 384)
    pipeline._embedder = mock_embedder
    result = pipeline.compute_embeddings(["text one", "text two"])
    assert result.shape == (2, 384)


def test_process_post_enriches_dict():
    pipeline = PreprocessingPipeline()
    pipeline._nlp = _make_mock_nlp()
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    pipeline._embedder = mock_embedder

    post = {
        "did": "did:plc:test",
        "rkey": "abc",
        "text": "Le chat mange https://example.com",
        "created_at": "2026-01-01T00:00:00Z",
        "langs": ["fr"],
        "collected_at": "2026-01-01T00:00:01Z",
        "raw": {},
    }
    result = pipeline.process_post(post)
    assert "cleaned_text" in result
    assert "tokens" in result
    assert "embedding" in result
    assert "https" not in result["cleaned_text"]
    assert result["did"] == "did:plc:test"
