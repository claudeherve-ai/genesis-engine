"""Offline tests for TestStage real-execution vs simulation dispatch.

Verifies that when concrete scenarios are set via ``set_scenarios`` the stage
runs the instrumented runtime (mode="execution", transcripts populated, numeric
metrics), and that with no scenarios it falls back to the legacy LLM-roleplay
simulation (mode="simulation"). All offline — scripted fake LLM, no network.
"""

from __future__ import annotations

import json
from typing import Callable, List

import pytest

from genesis.llm.provider import LLMResponse
from genesis.models.agent import AgentDefinition, ToolConfig, TestScenario
from genesis.pipeline.test import TestStage


class FakeLLM:
    def __init__(self, responder: Callable[[str, str], str]) -> None:
        self._responder = responder
        self.calls: List[dict] = []

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format=None,
    ) -> LLMResponse:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return LLMResponse(
            content=self._responder(system_prompt, user_prompt),
            model="fake-model",
            usage={},
        )


def make_agent(name: str, role: str, tools: List[str] | None = None) -> AgentDefinition:
    return AgentDefinition(
        name=name,
        role=role,
        system_prompt=f"You are {name}.",
        tools=[ToolConfig(name=t, description=f"{t} tool") for t in (tools or [])],
    )


# ── Execution path ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execution_path_produces_real_results_and_transcripts():
    llm = FakeLLM(lambda _s, _u: "ANSWER: the subscription costs $20 per month")
    stage = TestStage(llm)
    agents = [make_agent("pricing_agent", "Answer pricing questions")]
    stage.set_scenarios([
        TestScenario(
            name="price",
            input="How much per month?",
            expected_outputs=["$20", "per month"],
        )
    ])

    results = await stage.run(agents)

    assert results.mode == "execution"
    assert results.scenarios_run == 1
    assert results.scenarios_passed == 1
    assert results.overall_score == pytest.approx(1.0)
    assert len(results.transcripts) == 1
    assert results.transcripts[0]["agent"] == "pricing_agent"
    # Numeric-only metrics from real execution.
    assert set(results.metrics.keys()) == {
        "precision", "tool_accuracy", "routing_accuracy", "avg_latency_ms",
    }


@pytest.mark.asyncio
async def test_execution_failure_emits_structured_failure():
    llm = FakeLLM(lambda _s, _u: "ANSWER: I don't know")
    stage = TestStage(llm)
    agents = [make_agent("pricing_agent", "Answer pricing questions")]
    stage.set_scenarios([
        TestScenario(
            name="price",
            input="How much?",
            expected_outputs=["$20", "per month"],
        )
    ])

    results = await stage.run(agents)

    assert results.mode == "execution"
    assert results.scenarios_passed == 0
    assert results.failure_count == 1
    failure = results.failures[0]
    assert failure.scenario == "price"
    assert failure.metric == "precision"


@pytest.mark.asyncio
async def test_set_scenarios_empty_restores_simulation():
    """An empty scenario list must restore simulation behavior."""
    sim_payload = json.dumps({
        "scenarios_run": 10,
        "scenarios_passed": 9,
        "overall_score": 0.9,
        "metrics": {"resolution_rate": 0.9},
        "failures": [],
        "passed": True,
    })
    llm = FakeLLM(lambda _s, _u: sim_payload)
    stage = TestStage(llm)
    agents = [make_agent("agent", "General assistant")]

    stage.set_scenarios([TestScenario(name="s", input="x", expected_outputs=["x"])])
    stage.set_scenarios([])  # cleared → simulation
    results = await stage.run(agents)

    assert results.mode == "simulation"
    assert results.scenarios_run == 10


# ── Simulation path (back-compat) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_simulation_path_when_no_scenarios():
    sim_payload = json.dumps({
        "scenarios_run": 12,
        "scenarios_passed": 11,
        "overall_score": 0.88,
        "metrics": {"resolution_rate": 0.9, "handoff_correctness": 0.85},
        "failures": [
            {
                "scenario": "edge case",
                "agent": "triage_agent",
                "expected": "route to billing",
                "actual": "routed to technical",
                "metric": "handoff_correctness",
            }
        ],
        "passed": True,
    })
    llm = FakeLLM(lambda _s, _u: sim_payload)
    stage = TestStage(llm)
    agents = [make_agent("triage_agent", "Route requests")]

    results = await stage.run(agents)

    assert results.mode == "simulation"
    assert results.scenarios_run == 12
    assert results.scenarios_passed == 11
    assert results.failure_count == 1
    assert results.transcripts == []


@pytest.mark.asyncio
async def test_run_signature_unchanged_accepts_agents_only():
    """Orchestrator calls stage.run(agents) positionally — keep that contract."""
    llm = FakeLLM(lambda _s, _u: "ANSWER: ok response here")
    stage = TestStage(llm)
    agents = [make_agent("agent", "General")]
    stage.set_scenarios([TestScenario(name="s", input="hello")])

    results = await stage.run(agents)  # positional, single arg

    assert results.mode in ("execution", "heuristic")
    assert results.scenarios_run == 1
