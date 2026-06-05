"""Pluggable web-search providers with graceful, keyless fallback.

Genesis grounds generated agents in *real* data. This module exposes a single
``SearchProvider`` interface with two concrete implementations:

  * ``TavilySearchProvider`` — production-grade search via the Tavily API
    (used automatically when ``TAVILY_API_KEY`` is set).
  * ``DuckDuckGoSearchProvider`` — keyless HTML-scrape fallback so the system
    works out of the box (and so tests run offline with a mocked transport).

``get_search_provider()`` picks the best available provider. Every result
carries a citation (title + URL + source) so downstream prompts and the UI can
show *where* information came from — no unattributed claims.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """A single search result with provenance for citations."""
    title: str
    url: str
    snippet: str = ""
    source: str = "web"
    score: float = 0.0

    def citation(self) -> dict:
        return {"title": self.title, "url": self.url, "source": self.source}


class SearchError(RuntimeError):
    """Raised when a search provider fails in a way callers should surface."""


class SearchProvider(ABC):
    """Abstract web-search provider."""

    name: str = "base"
    requires_key: bool = False
    env_var: str = ""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SearchHit]:
        ...

    @property
    def available(self) -> bool:
        if not self.requires_key:
            return True
        return bool(os.getenv(self.env_var, ""))


class TavilySearchProvider(SearchProvider):
    """Production search via the Tavily API (https://docs.tavily.com)."""

    name = "tavily"
    requires_key = True
    env_var = "TAVILY_API_KEY"
    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> List[SearchHit]:
        key = os.getenv(self.env_var, "")
        if not key:
            raise SearchError("TAVILY_API_KEY not configured")

        payload = {
            "api_key": key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()

        hits: List[SearchHit] = []
        for item in data.get("results", [])[:max_results]:
            hits.append(SearchHit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="tavily",
                score=float(item.get("score", 0.0)),
            ))
        logger.info("Tavily search '%s': %d results", query[:50], len(hits))
        return hits


class DuckDuckGoSearchProvider(SearchProvider):
    """Keyless fallback that scrapes the DuckDuckGo HTML endpoint."""

    name = "duckduckgo"
    requires_key = False
    ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> List[SearchHit]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; rv:120.0) "
                "Gecko/20100101 Firefox/120.0"
            )
        }
        url = f"{self.ENDPOINT}?q={quote_plus(query)}"
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        hits: List[SearchHit] = []
        for item in soup.select(".result")[:max_results]:
            title_el = item.select_one(".result__title")
            snippet_el = item.select_one(".result__snippet")
            link_el = item.select_one(".result__url")
            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            link = link_el.get("href", "") if link_el else ""
            if title and link:
                hits.append(SearchHit(title=title, url=link, snippet=snippet, source="duckduckgo"))
        logger.info("DuckDuckGo search '%s': %d results", query[:50], len(hits))
        return hits


def get_search_provider(prefer: Optional[str] = None) -> SearchProvider:
    """Return the best available search provider.

    Tavily is used when ``TAVILY_API_KEY`` is set; otherwise the keyless
    DuckDuckGo provider is returned so the system always works.
    """
    if prefer == "tavily" or (prefer is None and os.getenv("TAVILY_API_KEY")):
        tavily = TavilySearchProvider()
        if tavily.available:
            return tavily
    return DuckDuckGoSearchProvider()


__all__ = [
    "SearchHit",
    "SearchError",
    "SearchProvider",
    "TavilySearchProvider",
    "DuckDuckGoSearchProvider",
    "get_search_provider",
]
