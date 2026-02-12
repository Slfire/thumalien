"""Tests pour le module emotion."""

from unittest.mock import MagicMock

from src.emotion.analyzer import EmotionAnalyzer


def test_vader_positive():
    analyzer = EmotionAnalyzer()
    scores = analyzer.analyze_vader("This is a great and wonderful day!")
    assert scores["compound"] > 0


def test_vader_negative():
    analyzer = EmotionAnalyzer()
    scores = analyzer.analyze_vader("This is terrible and awful.")
    assert scores["compound"] < 0


def test_analyze_bert_no_model():
    analyzer = EmotionAnalyzer(model_dir="/nonexistent/path")
    results = analyzer.analyze_bert(["Some text"])
    assert len(results) == 1
    assert results[0]["emotion"] == "unknown"


def test_analyze_bert_empty():
    analyzer = EmotionAnalyzer()
    analyzer._model_available = True
    analyzer._pipeline = MagicMock(return_value=[])
    results = analyzer.analyze_bert([])
    assert results == []


def test_analyze_bert_with_mock():
    analyzer = EmotionAnalyzer()
    analyzer._model_available = True
    mock = MagicMock()
    mock.return_value = [
        [
            {"label": "joy", "score": 0.7},
            {"label": "sadness", "score": 0.1},
            {"label": "anger", "score": 0.05},
            {"label": "fear", "score": 0.05},
            {"label": "love", "score": 0.05},
            {"label": "surprise", "score": 0.05},
        ]
    ]
    analyzer._pipeline = mock
    results = analyzer.analyze_bert(["I am so happy today"])
    assert len(results) == 1
    assert results[0]["emotion"] == "joy"
    assert results[0]["score"] == 0.7
    assert len(results[0]["all_scores"]) == 6


def test_analyze_full_vader_only():
    analyzer = EmotionAnalyzer(model_dir="/nonexistent/path")
    result = analyzer.analyze_full("Happy day!")
    assert "vader" in result
    assert result["bert"] is None
    assert result["vader"]["compound"] > 0
