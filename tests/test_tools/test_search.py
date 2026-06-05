"""Offline tests for the pluggable search providers (Phase 2).

Uses ``httpx.MockTransport`` so no real network calls are made.
"""

from __future__ import annotations

import httpx
import pytest

from genesis.tools import search as search_mod
from genesis.tools.search import (
    DuckDuckGoSearchProvider,
    SearchError,
    SearchHit,
    TavilySearchProvider,
    get_search_provider,
)


# ── provider selection ───────────────────────────────────────────────────────


def test_default_provider_is_duckduckgo_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = get_search_provider()
    assert provider.name == "duckduckgo"


def test_tavily_selected_when_key_present(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
    provider = get_search_provider()
    assert provider.name == "tavily"


def test_tavily_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert TavilySearchProvider().available is False


# ── Tavily provider ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_search_parses_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.tavily.com" in str(request.url)
        return httpx.Response(200, json={
            "results": [
                {"title": "Doc A", "url": "https://a.example", "content": "snippet a", "score": 0.9},
                {"title": "Doc B", "url": "https://b.example", "content": "snippet b", "score": 0.5},
            ]
        })

    transport = httpx.MockTransport(handler)
    orig = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return orig(*args, **kwargs)

    monkeypatch.setattr(search_mod.httpx, "AsyncClient", patched)

    hits = await TavilySearchProvider().search("azure functions", max_results=5)
    assert len(hits) == 2
    assert hits[0].title == "Doc A"
    assert hits[0].source == "tavily"
    assert hits[0].citation()["url"] == "https://a.example"


@pytest.mark.asyncio
async def test_tavily_raises_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SearchError):
        await TavilySearchProvider().search("anything")


# ── DuckDuckGo provider ──────────────────────────────────────────────────────


_DDG_HTML = """
<html><body>
  <div class="result">
    <a class="result__title">First Result</a>
    <a class="result__snippet">snippet one</a>
    <a class="result__url" href="https://one.example">one.example</a>
  </div>
  <div class="result">
    <a class="result__title">Second Result</a>
    <a class="result__snippet">snippet two</a>
    <a class="result__url" href="https://two.example">two.example</a>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_duckduckgo_scrapes_results(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_DDG_HTML)

    transport = httpx.MockTransport(handler)
    orig = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return orig(*args, **kwargs)

    monkeypatch.setattr(search_mod.httpx, "AsyncClient", patched)

    hits = await DuckDuckGoSearchProvider().search("query", max_results=5)
    assert len(hits) == 2
    assert hits[0].title == "First Result"
    assert hits[0].url == "https://one.example"
    assert hits[0].source == "duckduckgo"


def test_search_hit_citation_shape():
    hit = SearchHit(title="T", url="https://x", snippet="s", source="web")
    cite = hit.citation()
    assert cite == {"title": "T", "url": "https://x", "source": "web"}
