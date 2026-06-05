"""Offline tests for the anti-hallucination tool catalog validation (Phase 2).

Covers ``suggest_tools``, ``ToolValidationReport`` and ``validate_agent_tools``
across AgentDefinition objects, raw dicts, and bare string tool names.
"""

from __future__ import annotations

from genesis.models.agent import AgentDefinition, ToolConfig
from genesis.tools.catalog import (
    ToolValidationReport,
    suggest_tools,
    validate_agent_tools,
)


def make_agent(name: str, tools):
    return AgentDefinition(
        name=name,
        role=f"{name} role",
        system_prompt=f"You are {name}.",
        tools=[ToolConfig(name=t, description=f"{t} tool") for t in tools],
    )


# ── suggest_tools ────────────────────────────────────────────────────────────


def test_suggest_tools_returns_real_catalog_names():
    suggestions = suggest_tools("send_slack_message")
    assert suggestions  # non-empty
    # Every suggestion must be a real catalog tool.
    from genesis.tools.catalog import get_tool
    for s in suggestions:
        assert get_tool(s) is not None


def test_suggest_tools_close_match_for_typo():
    # 'web_serch' should fuzzy-match 'web_search'.
    assert "web_search" in suggest_tools("web_serch")


def test_suggest_tools_respects_limit():
    assert len(suggest_tools("anything", limit=2)) <= 2


# ── validate_agent_tools: valid ──────────────────────────────────────────────


def test_all_real_tools_pass():
    agents = [make_agent("a", ["web_search", "web_scrape"])]
    report = validate_agent_tools(agents)
    assert report.valid is True
    assert report.checked == 2
    assert report.hallucinated == {}
    assert report.feedback_text() == ""


def test_empty_agents_is_valid():
    report = validate_agent_tools([])
    assert report.valid is True
    assert report.checked == 0


# ── validate_agent_tools: hallucinations ─────────────────────────────────────


def test_hallucinated_tool_flagged_with_suggestions():
    agents = [make_agent("researcher", ["web_search", "magic_oracle_tool"])]
    report = validate_agent_tools(agents)
    assert report.valid is False
    assert "magic_oracle_tool" in report.hallucinated["researcher"]
    assert "magic_oracle_tool" in report.suggestions
    assert report.checked == 2


def test_feedback_text_lists_fake_tool_and_real_catalog():
    agents = [make_agent("a", ["totally_fake_tool"])]
    report = validate_agent_tools(agents)
    text = report.feedback_text()
    assert "totally_fake_tool" in text
    assert "does not exist" in text
    # Real catalog must be advertised for correction.
    assert "web_search" in text


# ── input shape flexibility ──────────────────────────────────────────────────


def test_accepts_dict_agents_and_string_tools():
    agents = [{"name": "d", "tools": ["web_search", "fake_thing"]}]
    report = validate_agent_tools(agents)
    assert report.valid is False
    assert "fake_thing" in report.hallucinated["d"]


def test_accepts_dict_tools():
    agents = [{"name": "d", "tools": [{"name": "web_search"}, {"name": "nope"}]}]
    report = validate_agent_tools(agents)
    assert report.valid is False
    assert "nope" in report.hallucinated["d"]
    assert report.checked == 2


def test_report_is_dataclass_instance():
    assert isinstance(validate_agent_tools([]), ToolValidationReport)
