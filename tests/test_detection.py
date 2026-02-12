"""Tests pour le module detection."""

from src.detection.classifier import FakeNewsClassifier


def test_classifier_init():
    clf = FakeNewsClassifier()
    assert clf.model is None
    assert clf.model_path is None


def test_classifier_init_with_path():
    clf = FakeNewsClassifier(model_path="models/fake_news.pkl")
    assert clf.model_path == "models/fake_news.pkl"
