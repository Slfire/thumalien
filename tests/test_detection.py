"""Tests pour le module detection."""

from unittest.mock import MagicMock

from src.detection.classifier import DEFAULT_LABELS, FakeNewsClassifier


def test_classifier_init_defaults():
    clf = FakeNewsClassifier()
    assert clf.labels == DEFAULT_LABELS
    assert clf._pipeline is None


def test_classifier_init_custom():
    clf = FakeNewsClassifier(model_dir="/tmp/custom_model", labels=["a", "b"])
    assert clf.model_dir == "/tmp/custom_model"
    assert clf.labels == ["a", "b"]


def _make_mock_pipeline():
    mock = MagicMock()
    mock.return_value = [
        {"label": "REAL", "score": 0.85},
        {"label": "FAKE", "score": 0.15},
    ]
    return mock


def test_predict_single():
    clf = FakeNewsClassifier()
    clf._pipeline = _make_mock_pipeline()
    result = clf.predict_single("Un texte de test")
    assert result["label"] == "REAL"
    assert result["score"] == 0.85
    assert len(result["all_scores"]) == 2


def test_predict_batch():
    clf = FakeNewsClassifier()
    mock = MagicMock()
    mock.return_value = [
        [{"label": "REAL", "score": 0.8}, {"label": "FAKE", "score": 0.2}],
        [{"label": "FAKE", "score": 0.7}, {"label": "REAL", "score": 0.3}],
    ]
    clf._pipeline = mock
    results = clf.predict(["text1", "text2"])
    assert len(results) == 2
    assert results[0]["label"] == "REAL"
    assert results[1]["label"] == "FAKE"


def test_predict_empty_list():
    clf = FakeNewsClassifier()
    assert clf.predict([]) == []


def test_predict_single_text_wraps():
    clf = FakeNewsClassifier()
    mock = MagicMock()
    mock.return_value = [
        {"label": "FAKE", "score": 0.6},
        {"label": "REAL", "score": 0.4},
    ]
    clf._pipeline = mock
    results = clf.predict(["single text"])
    assert len(results) == 1
    assert results[0]["label"] == "FAKE"
