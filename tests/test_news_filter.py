"""Tests pour le module filtering (news filter)."""

from src.filtering.news_filter import NewsFilter


def test_casual_short_post():
    nf = NewsFilter()
    assert nf.is_news_like("just had the best coffee ever!") is False


def test_casual_personal():
    nf = NewsFilter()
    assert nf.is_news_like("feeling great today, going to the park") is False


def test_news_url_and_keyword():
    nf = NewsFilter()
    assert nf.is_news_like("Breaking: new policy announced https://news.com/article") is True


def test_news_long_with_keyword():
    nf = NewsFilter()
    text = (
        "According to officials, the government has confirmed that "
        "a new investigation will be launched into the matter this week."
    )
    assert nf.is_news_like(text) is True


def test_news_quotes_and_url():
    nf = NewsFilter()
    assert nf.is_news_like('"This is unprecedented" says expert https://t.co/abc') is True


def test_url_without_keywords_is_casual():
    nf = NewsFilter()
    assert nf.is_news_like("check out my new blog https://myblog.com") is False


def test_filter_posts_adds_is_news():
    nf = NewsFilter()
    posts = [
        {"text": "Breaking news: report confirms https://example.com"},
        {"text": "lol this is so funny"},
        {"text": "According to sources, the president announced a new policy today"},
    ]
    result = nf.filter_posts(posts)
    assert result[0]["is_news"] is True
    assert result[1]["is_news"] is False
    assert result[2]["is_news"] is True


def test_filter_uses_cleaned_text():
    nf = NewsFilter()
    posts = [
        {"text": "raw text", "cleaned_text": "Breaking: officials confirmed https://t.co/x"},
    ]
    result = nf.filter_posts(posts)
    assert result[0]["is_news"] is True


def test_empty_list():
    nf = NewsFilter()
    assert nf.filter_posts([]) == []
