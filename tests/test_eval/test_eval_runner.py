"""Tests for the evaluation regression suite.

These tests inject a deterministic scripted LLM (never a global provider, so no
API keys are needed) and assert the EvalRunner scores produced agents against
golden expectations. A negative-control case wires a *broken* LLM (drops a
required tool) and asserts the runner REPORTS the failure — proving the eval
measures reality rather than rubber-stamping a mock.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from genesis.llm.provider import LLMProvider, LLMResponse
from genesis.eval import EvalRunner, EvalReport, load_datasets


def _agent(name: str, role: str, tools: List[str]) -> Dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "system_prompt": f"You are {name}.",
        "tools": [
            {
                "name": t,
                "description": "a real catalog tool",
                "schema": {"type": "object", "properties": {}},
                "endpoint": None,
                "auth_required": False,
            }
            for t in tools
        ],
        "skills": [],
        "coordination_rules": {
            "handoff_format": "json",
            "shared_context": [],
            "escalation_path": [],
        },
        "config_yaml": f"name: {name}",
    }


class ScriptedLLM(LLMProvider):
    """Returns a fixed agents payload regardless of prompt."""

    def __init__(self, agents: List[Dict[str, Any]]) -> None:
        self._payload = json.dumps({"agents": agents})

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        return LLMResponse(content=self._payload, model="scripted", usage={})

    async def health_check(self) -> bool:
        return True


def _support_case() -> Dict[str, Any]:
    return {
        "name": "support_triage",
        "architecture": {
            "topology": "router",
            "agents": [
                {"name": "triage_router", "role": "triage tickets", "tools": []},
                {"name": "account_lookup", "role": "look up accounts", "tools": []},
                {"name": "billing_specialist", "role": "resolve billing", "tools": []},
            ],
        },
        "expected_roles": ["triage", "account", "billing"],
        "expected_tools": ["database_query"],
        "min_agents": 3,
    }


@pytest.mark.asyncio
async def test_competent_llm_passes():
    case = _support_case()
    good_agents = [
        _agent("triage_router", "triage incoming tickets", []),
        _agent("account_lookup", "account record lookup", ["database_query"]),
        _agent("billing_specialist", "billing dispute resolution", ["database_query"]),
    ]
    runner = EvalRunner(ScriptedLLM(good_agents))
    result = await runner.evaluate(case)

    assert result.passed is True
    assert result.role_coverage == 1.0
    assert result.tool_coverage == 1.0
    assert result.hallucinated_tools == []
    assert result.agent_count == 3


@pytest.mark.asyncio
async def test_negative_control_missing_tool_fails():
    # Broken LLM drops the required database_query tool entirely.
    case = _support_case()
    broken_agents = [
        _agent("triage_router", "triage incoming tickets", []),
        _agent("account_lookup", "account record lookup", []),
        _agent("billing_specialist", "billing dispute resolution", []),
    ]
    runner = EvalRunner(ScriptedLLM(broken_agents))
    result = await runner.evaluate(case)

    assert result.passed is False
    assert "database_query" in result.missing_tools
    assert result.tool_coverage < 1.0


@pytest.mark.asyncio
async def test_negative_control_missing_role_fails():
    # Broken LLM omits the billing specialist entirely.
    case = _support_case()
    broken_agents = [
        _agent("triage_router", "triage incoming tickets", ["database_query"]),
        _agent("account_lookup", "account record lookup", ["database_query"]),
    ]
    runner = EvalRunner(ScriptedLLM(broken_agents))
    result = await runner.evaluate(case)

    assert result.passed is False
    assert "billing" in result.missing_roles


@pytest.mark.asyncio
async def test_run_suite_aggregates_and_flags_regressions():
    case = _support_case()
    good = [
        _agent("triage_router", "triage incoming tickets", ["database_query"]),
        _agent("account_lookup", "account record lookup", ["database_query"]),
        _agent("billing_specialist", "billing dispute resolution", ["database_query"]),
    ]
    runner = EvalRunner(ScriptedLLM(good))
    report = await runner.run_suite([case])

    assert isinstance(report, EvalReport)
    assert report.total == 1
    assert report.passed == 1
    assert report.regressions == []
    d = report.to_dict()
    assert d["pass_rate"] == 1.0
    assert "results" in d


def test_load_datasets_reads_golden_files():
    cases = load_datasets()
    assert len(cases) >= 2
    names = {c["name"] for c in cases}
    assert "support_triage_router" in names
    for case in cases:
        assert "architecture" in case
        assert "expected_roles" in case
        assert "expected_tools" in case


@pytest.mark.asyncio
async def test_golden_datasets_pass_with_matching_llm():
    # Drive each golden case with an LLM scripted to satisfy its expectations,
    # proving the datasets are internally consistent and runnable.
    cases = load_datasets()
    for case in cases:
        agents = []
        arch_agents = case["architecture"]["agents"]
        expected_tools = case["expected_tools"]
        # Give every required tool to the first agent; map roles 1:1.
        for i, spec in enumerate(arch_agents):
            tools = expected_tools if i == 0 else []
            agents.append(_agent(spec["name"], spec["role"], tools))
        runner = EvalRunner(ScriptedLLM(agents))
        result = await runner.evaluate(case)
        assert result.passed is True, f"{case['name']} failed: {result.to_dict()}"
