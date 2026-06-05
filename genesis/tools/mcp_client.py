"""MCP (Model Context Protocol) client for Genesis Engine.

Provides authoritative grounding from vendor documentation servers so that
generated agents are grounded in real, up-to-date documentation rather than
LLM hallucinations about tool capabilities.

The flagship integration is the **official Microsoft Learn MCP server**
(``https://learn.microsoft.com/api/mcp``), which speaks the MCP
*streamable-HTTP* transport: an ``initialize`` handshake, an ``initialized``
notification, then ``tools/call`` against ``microsoft_docs_search``. Responses
may arrive as JSON or as Server-Sent-Events frames; both are parsed.

Every transport failure degrades gracefully (logged with context, returns an
empty result) so the pipeline never crashes when a server is unreachable —
and so the whole stack runs offline in tests with a mocked transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2025-06-18"
_CLIENT_INFO = {"name": "genesis-engine", "version": "0.4.0"}


@dataclass
class MCPServer:
    name: str
    url: str
    description: str
    tier: int
    search_tool: str = "search"
    query_arg: str = "query"
    tools: List[str] = field(default_factory=list)
    auth_env: str = ""
    enabled: bool = True


# Only servers with REAL, reachable endpoints are enabled by default. The
# Microsoft Learn MCP server is public and keyless. Others remain registered
# for extension but stay disabled until a real endpoint/credential is wired,
# so we never pretend to query a server that does not exist.
MCP_SERVERS: Dict[str, MCPServer] = {
    "microsoft_learn": MCPServer(
        name="microsoft_learn",
        url="https://learn.microsoft.com/api/mcp",
        description="Official Microsoft Learn documentation — Azure, .NET, M365, Power Platform",
        tier=1,
        search_tool="microsoft_docs_search",
        query_arg="query",
        tools=["microsoft_docs_search", "microsoft_docs_fetch"],
    ),
    "github": MCPServer(
        name="github",
        url="https://api.githubcopilot.com/mcp/",
        description="GitHub MCP — repos, issues, PRs, code search",
        tier=2,
        search_tool="search_code",
        query_arg="query",
        tools=["search_code", "get_repository", "search_issues"],
        auth_env="GITHUB_TOKEN",
        enabled=False,
    ),
}


class MCPClient:
    """MCP streamable-HTTP client for querying documentation servers."""

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
        return self._http

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def search_all(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Query every enabled MCP server and merge ranked results."""
        results: List[Dict[str, Any]] = []
        client = await self._get_client()

        for name, server in MCP_SERVERS.items():
            if not server.enabled:
                continue
            if server.auth_env and not os.getenv(server.auth_env):
                logger.debug("MCP server %s skipped: %s not set", name, server.auth_env)
                continue
            try:
                server_results = await self._query_server(client, server, query, max_results)
                results.extend(server_results)
            except Exception as e:  # never let one server crash grounding
                logger.info("MCP server %s unavailable (query=%r): %s", name, query[:50], e)

        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results[: max_results * 2]

    async def _query_server(
        self,
        client: httpx.AsyncClient,
        server: MCPServer,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if server.auth_env:
            token = os.getenv(server.auth_env, "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        # 1) initialize handshake (captures session id if the server issues one)
        init_resp = await client.post(
            server.url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _CLIENT_INFO,
                },
            },
        )
        init_resp.raise_for_status()
        session_id = init_resp.headers.get("Mcp-Session-Id") or init_resp.headers.get("mcp-session-id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        # 2) initialized notification (best-effort; ignore failures)
        try:
            await client.post(
                server.url,
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
        except httpx.HTTPError:
            pass

        # 3) tools/call against the server's search tool
        call_resp = await client.post(
            server.url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": server.search_tool,
                    "arguments": {server.query_arg: query},
                },
            },
        )
        call_resp.raise_for_status()
        data = self._parse_response(call_resp)
        return self._normalize_results(data, server.name)

    @staticmethod
    def _parse_response(resp: httpx.Response) -> Dict[str, Any]:
        """Parse a JSON or text/event-stream MCP response into a dict."""
        content_type = resp.headers.get("content-type", "")
        text = resp.text
        if "text/event-stream" in content_type or text.lstrip().startswith("event:"):
            # SSE: collect the last JSON payload from `data:` lines.
            payload: Dict[str, Any] = {}
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    chunk = line[len("data:"):].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        payload = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
            return payload
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return {}

    @staticmethod
    def _normalize_results(data: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not isinstance(data, dict):
            return results

        # JSON-RPC error → surface nothing (logged upstream).
        if "error" in data and "result" not in data:
            logger.debug("MCP %s returned error: %s", source, data.get("error"))
            return results

        raw = data.get("result", data)

        # MCP tools/call result → {"content": [{"type": "text", "text": "..."}]}
        if isinstance(raw, dict) and "content" in raw:
            for block in raw.get("content", []):
                if not isinstance(block, dict):
                    continue
                text = block.get("text", "")
                if not text:
                    continue
                # Some servers return a JSON string list inside the text block.
                parsed = MCPClient._maybe_json(text)
                if isinstance(parsed, list):
                    for item in parsed:
                        results.append(MCPClient._coerce_item(item, source))
                else:
                    results.append({"content": text, "source": source, "score": 5.0})
            return results

        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("results", raw.get("items", []))
        else:
            return results

        for item in items:
            results.append(MCPClient._coerce_item(item, source))
        return results

    @staticmethod
    def _maybe_json(text: str) -> Any:
        stripped = text.strip()
        if stripped[:1] in ("[", "{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return text
        return text

    @staticmethod
    def _coerce_item(item: Any, source: str) -> Dict[str, Any]:
        if isinstance(item, str):
            return {"content": item, "source": source, "score": 5.0}
        if isinstance(item, dict):
            return {
                "title": item.get("title", item.get("name", "")),
                "content": item.get("content", item.get("text", item.get("snippet", ""))),
                "url": item.get("url", item.get("contentUrl", item.get("link", ""))),
                "source": source,
                "score": float(item.get("score", item.get("relevance", 5.0))),
            }
        return {"content": str(item), "source": source, "score": 1.0}


async def mcp_grounding(query: str) -> str:
    """Query MCP servers for authoritative grounding on a topic."""
    client = MCPClient()
    try:
        results = await client.search_all(query)
    finally:
        await client.close()

    if not results:
        return ""

    lines = [f"## Authoritative Documentation (MCP): \"{query}\"", ""]
    seen = set()

    for r in results[:10]:
        title = r.get("title", "Untitled") or "Untitled"
        content = str(r.get("content", ""))[:400]
        url = r.get("url", "")
        source = r.get("source", "unknown")

        key = (title, content[:100])
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"### [{source.upper()}] {title}")
        if url:
            lines.append(f"    URL: {url}")
        lines.append(f"    {content}")
        lines.append("")

    return "\n".join(lines)


async def research_agent_tools(agent_role: str, needed_tools: List[str]) -> str:
    """Research real tool capabilities for an agent being generated.

    Queries the MCP servers and the web to find actual tool documentation,
    APIs, and best practices — ensuring generated agents use real tools
    rather than hallucinated ones. Failures are logged (not silently
    swallowed) but never abort generation.
    """
    parts: List[str] = []

    for tool in needed_tools:
        query = f"{tool} API documentation {agent_role}"
        try:
            mcp_text = await mcp_grounding(query)
            if mcp_text:
                parts.append(mcp_text)
        except Exception as e:
            logger.info("MCP grounding failed for %r: %s", tool, e)

        try:
            from genesis.tools import research_topic, format_context_for_prompt
            web_ctx = await research_topic(query)
            web_text = format_context_for_prompt(web_ctx)
            if web_text:
                parts.append(web_text)
        except Exception as e:
            logger.info("Web research failed for %r: %s", tool, e)

    return "\n\n".join(parts) if parts else ""
