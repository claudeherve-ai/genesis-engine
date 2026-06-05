"""Web intelligence tools — search, fetch, and grounding.

Provides real-time web search and content extraction to ground
LLM outputs in actual data. No hallucinations. No lying.

v0.2.0 — Added MCP server grounding for agent tool generation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from genesis.tools.search import SearchHit, get_search_provider

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"

    def citation(self) -> dict:
        return {"title": self.title, "url": self.url, "source": self.source}


@dataclass
class WebContext:
    query: str
    results: List[SearchResult] = field(default_factory=list)
    fetched_content: List[str] = field(default_factory=list)
    provider: str = ""
    error: Optional[str] = None

    def citations(self) -> List[dict]:
        """Structured citations for every result (provenance, no unattributed claims)."""
        return [r.citation() for r in self.results]


async def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search the web via the best available provider (Tavily or DuckDuckGo).

    Never raises — failures are logged with context and surfaced as an empty
    list so the pipeline degrades gracefully instead of crashing.
    """
    provider = get_search_provider()
    try:
        hits: List[SearchHit] = await provider.search(query, max_results=max_results)
        results = [
            SearchResult(title=h.title, url=h.url, snippet=h.snippet, source=h.source)
            for h in hits
        ]
        logger.info(
            "Web search via %s '%s': %d results",
            provider.name, query[:50], len(results),
        )
        return results
    except Exception as e:
        logger.warning(
            "Web search failed (provider=%s, query=%r): %s",
            provider.name, query[:50], e,
        )
        return []


async def web_fetch(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract clean text content from a URL."""
    if url.startswith("//"):
        url = "https:" + url
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; GenesisEngine/1.0)"}
            r = await client.get(url, headers=headers)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)

            return text[:max_chars]
    except Exception as e:
        logger.warning("Web fetch failed for %s: %s", url[:60], e)
        return ""


async def research_topic(query: str, fetch_depth: int = 2) -> WebContext:
    """Research a topic: search + fetch top results for deep context."""
    provider = get_search_provider()
    context = WebContext(query=query, provider=provider.name)

    context.results = await web_search(query, max_results=5)
    if not context.results:
        context.error = "no_results"
        return context

    tasks = [web_fetch(r.url) for r in context.results[:fetch_depth]]
    contents = await asyncio.gather(*tasks)
    context.fetched_content = [c for c in contents if c]

    return context


def format_context_for_prompt(context: WebContext, max_chars: int = 3000) -> str:
    """Format web research results into a prompt-ready string with citations."""
    if not context.results:
        return ""

    provider_note = f" (source: {context.provider})" if context.provider else ""
    lines = [f"## Web Research: \"{context.query}\"{provider_note}", ""]

    for i, result in enumerate(context.results[:5]):
        lines.append(f"[{i+1}] {result.title}")
        lines.append(f"    URL: {result.url}")
        if result.snippet:
            lines.append(f"    {result.snippet[:200]}")
        lines.append("")

    if context.fetched_content:
        lines.append("## Deep Context (fetched pages)")
        for i, content in enumerate(context.fetched_content[:2]):
            truncated = content[:max_chars // 2]
            lines.append(f"[Source {i+1}] {truncated}")
            lines.append("")

    lines.append("## Citations (cite these — do not invent sources)")
    for i, result in enumerate(context.results[:5]):
        lines.append(f"[{i+1}] {result.url}")

    return "\n".join(lines)


__all__ = [
    "SearchResult",
    "WebContext",
    "web_search",
    "web_fetch",
    "research_topic",
    "format_context_for_prompt",
]
