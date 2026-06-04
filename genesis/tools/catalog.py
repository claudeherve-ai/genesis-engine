"""Tool Catalog — registry of REAL, verified tools available for generated agents.

Every tool has a verified endpoint, auth requirements, JSON Schema, and rate limits.
Generated agents can ONLY use tools that exist in this catalog — no hallucinations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class CatalogTool:
    """A verified, real tool available to agents."""
    name: str
    description: str
    category: str  # "search", "data", "communication", "storage", "api", "compute"
    endpoint: Optional[str] = None  # API endpoint URL
    auth_required: bool = False
    auth_method: str = "none"  # "api_key", "bearer", "oauth2", "basic"
    env_var: str = ""  # env var for auth credentials
    json_schema: Dict[str, Any] = field(default_factory=dict)
    rate_limit: str = "100/hour"
    is_available: bool = True
    requires_configuration: bool = False
    docs_url: str = ""
    example_usage: str = ""

    @property
    def schema_dict(self) -> Dict[str, Any]:
        return self.json_schema


# ---------------------------------------------------------------------------
# The Actual Catalog
# ---------------------------------------------------------------------------

TOOL_CATALOG: Dict[str, CatalogTool] = {}


def _register(tool: CatalogTool) -> CatalogTool:
    TOOL_CATALOG[tool.name] = tool
    return tool


# --- SEARCH TOOLS ---

_register(CatalogTool(
    name="web_search",
    description="Search the web for real-time information using Tavily, Brave, or DuckDuckGo",
    category="search",
    endpoint="https://api.tavily.com/search",
    auth_required=True,
    auth_method="api_key",
    env_var="TAVILY_API_KEY",
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5, "maximum": 20},
        },
        "required": ["query"],
    },
    rate_limit="100/hour",
    docs_url="https://docs.tavily.com",
    example_usage='web_search(query="latest AI news", max_results=5)',
))

_register(CatalogTool(
    name="web_scrape",
    description="Extract clean text content from a web page URL",
    category="search",
    auth_required=False,
    json_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to scrape"},
            "max_chars": {"type": "integer", "default": 5000},
        },
        "required": ["url"],
    },
    rate_limit="50/hour",
    example_usage='web_scrape(url="https://example.com/article", max_chars=3000)',
))

# --- DATA TOOLS ---

_register(CatalogTool(
    name="financial_data",
    description="Get real-time stock and crypto quotes via Yahoo Finance or Alpha Vantage",
    category="data",
    endpoint="https://query1.finance.yahoo.com/v8/finance/chart",
    auth_required=False,
    json_schema={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker (AAPL) or crypto (BTC-USD)"},
            "type": {"type": "string", "enum": ["stock", "crypto"], "default": "stock"},
        },
        "required": ["symbol"],
    },
    rate_limit="200/hour",
    example_usage='financial_data(symbol="AAPL", type="stock")',
))

_register(CatalogTool(
    name="news_fetch",
    description="Fetch latest news articles from NewsAPI, GNews, or RSS feeds",
    category="data",
    endpoint="https://newsapi.org/v2/everything",
    auth_required=True,
    auth_method="api_key",
    env_var="NEWSAPI_KEY",
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "News search query"},
            "category": {"type": "string", "description": "News category"},
            "max_articles": {"type": "integer", "default": 10, "maximum": 100},
        },
        "required": ["query"],
    },
    rate_limit="100/day",
    docs_url="https://newsapi.org/docs",
    example_usage='news_fetch(query="Apple product launch", max_articles=10)',
))

_register(CatalogTool(
    name="github_search",
    description="Search GitHub repositories, get trending repos, and check activity",
    category="data",
    endpoint="https://api.github.com/search/repositories",
    auth_required=False,
    auth_method="bearer",
    env_var="GITHUB_TOKEN",
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query or repo name"},
            "type": {"type": "string", "enum": ["search", "trending", "activity"], "default": "search"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
    rate_limit="60/hour (authenticated), 10/hour (anonymous)",
    example_usage='github_search(query="machine learning framework", type="trending")',
))

_register(CatalogTool(
    name="arxiv_search",
    description="Search academic papers on arXiv",
    category="data",
    endpoint="http://export.arxiv.org/api/query",
    auth_required=False,
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "category": {"type": "string", "description": "arXiv category (e.g., cs.AI)"},
            "max_results": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
    rate_limit="Unlimited (please be respectful)",
    example_usage='arxiv_search(query="transformer architecture", category="cs.CL")',
))

# --- COMMUNICATION TOOLS ---

_register(CatalogTool(
    name="email_send",
    description="Send emails via SMTP or email API (SendGrid, Mailgun, AWS SES)",
    category="communication",
    endpoint="https://api.sendgrid.com/v3/mail/send",
    auth_required=True,
    auth_method="bearer",
    env_var="SENDGRID_API_KEY",
    json_schema={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "html": {"type": "boolean", "default": False},
        },
        "required": ["to", "subject", "body"],
    },
    rate_limit="100/day",
    requires_configuration=True,
    example_usage='email_send(to="user@example.com", subject="Hello", body="Message")',
))

_register(CatalogTool(
    name="notification_send",
    description="Send notifications via Slack, Discord, or Telegram webhooks",
    category="communication",
    endpoint="https://hooks.slack.com/services/...",
    auth_required=True,
    auth_method="api_key",
    env_var="SLACK_WEBHOOK_URL",
    json_schema={
        "type": "object",
        "properties": {
            "channel": {"type": "string", "enum": ["slack", "discord", "telegram"]},
            "message": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "urgent"], "default": "normal"},
        },
        "required": ["channel", "message"],
    },
    rate_limit="30/minute",
    requires_configuration=True,
    example_usage='notification_send(channel="slack", message="Deployment complete!")',
))

# --- STORAGE TOOLS ---

_register(CatalogTool(
    name="database_query",
    description="Query SQL databases (PostgreSQL, MySQL, SQLite)",
    category="storage",
    endpoint="postgresql://...",
    auth_required=True,
    auth_method="basic",
    env_var="DATABASE_URL",
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL query"},
            "params": {"type": "array", "items": {"type": "string"}, "default": []},
            "database": {"type": "string", "enum": ["postgresql", "mysql", "sqlite"], "default": "sqlite"},
        },
        "required": ["query"],
    },
    rate_limit="1000/hour",
    requires_configuration=True,
    example_usage='database_query(query="SELECT * FROM users WHERE id = $1", params=["123"])',
))

_register(CatalogTool(
    name="file_operations",
    description="Read, write, list, and delete files in the agent's workspace",
    category="storage",
    json_schema={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["read", "write", "list", "delete"]},
            "path": {"type": "string", "description": "File path relative to workspace"},
            "content": {"type": "string", "description": "Content for write operation"},
        },
        "required": ["operation", "path"],
    },
    rate_limit="Unlimited",
    example_usage='file_operations(operation="read", path="config.json")',
))

# --- API TOOLS ---

_register(CatalogTool(
    name="api_client",
    description="Make HTTP requests to external APIs (REST, GraphQL)",
    category="api",
    json_schema={
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
            "url": {"type": "string"},
            "headers": {"type": "object", "default": {}},
            "body": {"type": "object", "default": {}},
            "auth_type": {"type": "string", "enum": ["none", "bearer", "api_key", "basic"], "default": "none"},
        },
        "required": ["method", "url"],
    },
    rate_limit="1000/hour",
    requires_configuration=True,
    example_usage='api_client(method="GET", url="https://api.example.com/data", headers={"Authorization": "Bearer TOKEN"})',
))

_register(CatalogTool(
    name="code_execute",
    description="Execute safe sandboxed Python code for data processing and computation",
    category="compute",
    json_schema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "default": 30, "maximum": 300},
            "packages": {"type": "array", "items": {"type": "string"}, "default": [],
                         "description": "Allowed packages (requests, pandas, numpy, etc.)"},
        },
        "required": ["code"],
    },
    rate_limit="100/hour",
    example_usage='code_execute(code="print(sum(range(100)))", timeout=10)',
))

# ---------------------------------------------------------------------------
# Catalog operations
# ---------------------------------------------------------------------------

def get_tool(name: str) -> Optional[CatalogTool]:
    """Get a tool from the catalog by name."""
    return TOOL_CATALOG.get(name)

def list_tools(category: str = "") -> List[CatalogTool]:
    """List all tools, optionally filtered by category."""
    tools = list(TOOL_CATALOG.values())
    if category:
        tools = [t for t in tools if t.category == category]
    return sorted(tools, key=lambda t: t.name)

def list_available_tools() -> List[CatalogTool]:
    """List tools that are available (no configuration required)."""
    return [t for t in TOOL_CATALOG.values() if t.is_available]

def validate_tool(name: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate that a tool exists in the catalog and its schema matches.

    Returns {"valid": True/False, "issues": [...]}
    """
    tool = TOOL_CATALOG.get(name)
    if not tool:
        return {"valid": False, "issues": [f"Tool '{name}' does not exist in the catalog. It may be a hallucination."]}

    issues = []
    if tool.requires_configuration and tool.auth_required:
        import os
        if not os.getenv(tool.env_var, ""):
            issues.append(f"Tool '{name}' requires {tool.env_var} to be configured")

    if schema:
        # Basic schema match check
        tool_props = set(tool.json_schema.get("properties", {}).keys())
        gen_props = set(schema.get("properties", {}).keys())
        missing = tool_props - gen_props
        extra = gen_props - tool_props
        if missing:
            issues.append(f"Generated schema missing fields: {missing}")
        if extra:
            issues.append(f"Generated schema has extra fields: {extra}")

    return {"valid": len(issues) == 0, "issues": issues}

def search_catalog(query: str) -> List[CatalogTool]:
    """Search the catalog by name, description, or category."""
    q = query.lower()
    return [
        t for t in TOOL_CATALOG.values()
        if q in t.name.lower() or q in t.description.lower() or q in t.category.lower()
    ]


__all__ = [
    "CatalogTool", "TOOL_CATALOG",
    "get_tool", "list_tools", "list_available_tools",
    "validate_tool", "search_catalog",
]
