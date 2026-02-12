"""Tests pour le module emotion."""

from src.emotion.analyzer import EmotionAnalyzer


def test_vader_positive():
    analyzer = EmotionAnalyzer()
    scores = analyzer.analyze_vader("This is a great and wonderful day!")
    assert scores["compound"] > 0


def test_vader_negative():
    analyzer = EmotionAnalyzer()
    scores = analyzer.analyze_vader("This is terrible and awful.")
    assert scores["compound"] < 0
