"""Offline pure-function tests for the MCP client parsers (Phase 2).

These exercise ``_parse_response``, ``_normalize_results`` and ``_coerce_item``
without any network — the streamable-HTTP transport is never touched.
"""

from __future__ import annotations

import json

import httpx

from genesis.tools.mcp_client import MCP_SERVERS, MCPClient


# ── _parse_response ──────────────────────────────────────────────────────────


def _resp(text: str, content_type: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=text,
        headers={"content-type": content_type},
        request=httpx.Request("POST", "https://learn.microsoft.com/api/mcp"),
    )


def test_parse_plain_json_response():
    body = {"result": {"content": [{"type": "text", "text": "hello"}]}}
    data = MCPClient._parse_response(_resp(json.dumps(body), "application/json"))
    assert data == body


def test_parse_sse_response_takes_last_data_frame():
    sse = (
        "event: message\n"
        'data: {"result": {"content": [{"type": "text", "text": "first"}]}}\n'
        "\n"
        "event: message\n"
        'data: {"result": {"content": [{"type": "text", "text": "second"}]}}\n'
        "\n"
        "data: [DONE]\n"
    )
    data = MCPClient._parse_response(_resp(sse, "text/event-stream"))
    assert data["result"]["content"][0]["text"] == "second"


def test_parse_garbage_returns_empty_dict():
    data = MCPClient._parse_response(_resp("not json at all", "application/json"))
    assert data == {}


# ── _normalize_results ───────────────────────────────────────────────────────


def test_normalize_mcp_content_blocks():
    data = {"result": {"content": [
        {"type": "text", "text": "Azure Functions scale automatically."},
        {"type": "text", "text": "Use Durable Functions for orchestration."},
    ]}}
    results = MCPClient._normalize_results(data, "microsoft_learn")
    assert len(results) == 2
    assert results[0]["content"].startswith("Azure Functions")
    assert results[0]["source"] == "microsoft_learn"


def test_normalize_content_with_embedded_json_list():
    inner = json.dumps([
        {"title": "Doc1", "content": "c1", "url": "https://d1"},
        {"title": "Doc2", "content": "c2", "url": "https://d2"},
    ])
    data = {"result": {"content": [{"type": "text", "text": inner}]}}
    results = MCPClient._normalize_results(data, "microsoft_learn")
    assert len(results) == 2
    assert results[0]["title"] == "Doc1"
    assert results[1]["url"] == "https://d2"


def test_normalize_error_response_returns_empty():
    data = {"error": {"code": -32601, "message": "Method not found"}}
    assert MCPClient._normalize_results(data, "x") == []


def test_normalize_non_dict_returns_empty():
    assert MCPClient._normalize_results([], "x") == []  # type: ignore[arg-type]


# ── _coerce_item ─────────────────────────────────────────────────────────────


def test_coerce_string_item():
    item = MCPClient._coerce_item("just text", "src")
    assert item["content"] == "just text"
    assert item["source"] == "src"


def test_coerce_dict_item_maps_aliases():
    item = MCPClient._coerce_item(
        {"name": "Title", "snippet": "body", "contentUrl": "https://x", "relevance": 0.8},
        "src",
    )
    assert item["title"] == "Title"
    assert item["content"] == "body"
    assert item["url"] == "https://x"
    assert item["score"] == 0.8


# ── registry sanity ──────────────────────────────────────────────────────────


def test_microsoft_learn_server_is_real_and_enabled():
    server = MCP_SERVERS["microsoft_learn"]
    assert server.url == "https://learn.microsoft.com/api/mcp"
    assert server.enabled is True
    assert server.search_tool == "microsoft_docs_search"


def test_no_fake_placeholder_endpoints():
    # Guard against the leaked/fake endpoints scrubbed in Phase 0.
    for server in MCP_SERVERS.values():
        assert "tedcherve" not in server.url.lower()
        assert "mangoflower" not in server.url.lower()
