"""Offline tests for the instrumented AgentRuntime sandbox.

These tests use a scripted fake LLM (no API keys, no network) and assert that
the runtime scores agents from *reality*: required output substrings must
actually appear, declared+catalog tools must actually run, hallucinated/
undeclared tool calls are rejected, and routing failures are folded into the
pass/score so they cannot be masked.
"""

from __future__ import annotations

from typing import Callable, List

import pytest

from genesis.llm.provider import LLMResponse
from genesis.models.agent import AgentDefinition, ToolConfig, TestScenario
from genesis.runtime.sandbox import AgentRuntime, ExecutionResult, RuntimeReport


# ── Scripted fake LLM ────────────────────────────────────────────────────────


class FakeLLM:
    """Deterministic, offline LLM stub.

    ``responder(system_prompt, user_prompt) -> str`` lets each test script the
    exact text the "agent" emits, so we can drive CALL/ANSWER protocol paths.
    """

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
        content = self._responder(system_prompt, user_prompt)
        return LLMResponse(content=content, model="fake-model", usage={})


def const(text: str) -> Callable[[str, str], str]:
    return lambda _s, _u: text


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_agent(
    name: str,
    role: str,
    tool_names: List[str] | None = None,
) -> AgentDefinition:
    tools = [ToolConfig(name=t, description=f"{t} tool") for t in (tool_names or [])]
    return AgentDefinition(
        name=name,
        role=role,
        system_prompt=f"You are {name}.",
        tools=tools,
    )


# ── Normalization ────────────────────────────────────────────────────────────


def test_normalize_passthrough_test_scenario():
    rt = AgentRuntime(FakeLLM(const("ANSWER: ok")))
    scenario = TestScenario(name="s", input="hello", expected_outputs=["x"])
    assert rt._normalize(scenario, 0) is scenario


def test_normalize_string_to_scenario():
    rt = AgentRuntime(FakeLLM(const("ANSWER: ok")))
    s = rt._normalize("I need a refund please", 0)
    assert isinstance(s, TestScenario)
    assert s.input == "I need a refund please"
    assert s.name == "I need a refund please"[:50]
    assert s.expected_outputs == []
    assert s.expected_tools == []


def test_normalize_dict_reads_input_and_criteria():
    rt = AgentRuntime(FakeLLM(const("ANSWER: ok")))
    s = rt._normalize(
        {
            "name": "refund_case",
            "input": "refund my order",
            "expected_outputs": ["refund"],
            "expected_tools": ["web_search"],
            "route_to": "billing_agent",
        },
        0,
    )
    assert s.name == "refund_case"
    assert s.input == "refund my order"
    assert s.expected_outputs == ["refund"]
    assert s.expected_tools == ["web_search"]
    assert s.route_to == "billing_agent"


def test_normalize_dict_scenario_key_fallback():
    rt = AgentRuntime(FakeLLM(const("ANSWER: ok")))
    s = rt._normalize({"scenario": "legacy text"}, 2)
    assert s.input == "legacy text"


# ── Tokenization & routing ───────────────────────────────────────────────────


def test_tokens_drops_stopwords_and_singletons():
    toks = AgentRuntime._tokens("I want a refund for my billing")
    assert "refund" in toks
    assert "billing" in toks
    assert "i" not in toks
    assert "a" not in toks
    assert "for" not in toks


def test_route_picks_best_overlap():
    rt = AgentRuntime(FakeLLM(const("ANSWER: ok")))
    agents = [
        make_agent("triage_agent", "Classify and route incoming requests"),
        make_agent("billing_agent", "Handle billing questions and refunds"),
    ]
    routed = rt._route("I have a billing question about a refund", agents)
    assert routed.name == "billing_agent"


def test_route_zero_overlap_falls_back_to_first():
    rt = AgentRuntime(FakeLLM(const("ANSWER: ok")))
    agents = [
        make_agent("fallback_agent", "Default handler"),
        make_agent("billing_agent", "Handle billing questions and refunds"),
    ]
    routed = rt._route("zzzqqq wwwxxx", agents)
    assert routed.name == "fallback_agent"


# ── Deterministic scoring (expected_outputs) ────────────────────────────────


@pytest.mark.asyncio
async def test_execution_passes_when_phrases_present():
    llm = FakeLLM(const("ANSWER: The subscription costs $20 per month."))
    rt = AgentRuntime(llm)
    agents = [make_agent("pricing_agent", "Answer pricing questions")]
    scenarios = [
        TestScenario(
            name="price",
            input="How much does the subscription cost?",
            expected_outputs=["$20", "per month"],
        )
    ]
    report = await rt.execute(agents, scenarios)
    assert isinstance(report, RuntimeReport)
    r = report.results[0]
    assert r.mode == "execution"
    assert r.expected_found == 2
    assert r.expected_total == 2
    assert r.score == pytest.approx(1.0)
    assert r.passed is True


@pytest.mark.asyncio
async def test_execution_fails_when_phrases_missing():
    llm = FakeLLM(const("ANSWER: I'm not sure about that."))
    rt = AgentRuntime(llm)
    agents = [make_agent("pricing_agent", "Answer pricing questions")]
    scenarios = [
        TestScenario(
            name="price",
            input="How much does it cost?",
            expected_outputs=["$20", "per month"],
        )
    ]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert r.expected_found == 0
    assert r.score == pytest.approx(0.0)
    assert r.passed is False


# ── Tool execution (instrumented) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_declared_catalog_tool_is_executed():
    llm = FakeLLM(const(
        'CALL web_search {"query": "azure pricing"}\n'
        "ANSWER: I searched and found the pricing."
    ))
    rt = AgentRuntime(llm)
    agents = [make_agent("research_agent", "Research things", ["web_search"])]
    scenarios = [
        TestScenario(
            name="search",
            input="Find the latest azure pricing",
            expected_outputs=["found"],
            expected_tools=["web_search"],
        )
    ]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert "web_search" in r.tools_executed
    assert r.tool_accuracy == pytest.approx(1.0)
    # score = 0.6*precision(1.0) + 0.4*tool_accuracy(1.0) = 1.0
    assert r.score == pytest.approx(1.0)
    assert r.passed is True


@pytest.mark.asyncio
async def test_undeclared_tool_is_rejected():
    # Agent declares no tools but tries to call one.
    llm = FakeLLM(const(
        'CALL web_search {"query": "x"}\nANSWER: done'
    ))
    rt = AgentRuntime(llm)
    agents = [make_agent("plain_agent", "Answer questions")]  # no tools
    scenarios = [TestScenario(name="s", input="hi", expected_outputs=["done"])]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert r.tools_executed == []
    assert any("undeclared tool" in e for e in r.errors)


@pytest.mark.asyncio
async def test_hallucinated_tool_not_in_catalog_is_rejected():
    # Tool is declared on the agent but does not exist in the catalog.
    llm = FakeLLM(const(
        'CALL imaginary_tool {"q": "x"}\nANSWER: done'
    ))
    rt = AgentRuntime(llm)
    agents = [make_agent("agent", "Answer", ["imaginary_tool"])]
    scenarios = [TestScenario(name="s", input="hi", expected_outputs=["done"])]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert r.tools_executed == []
    assert any("hallucinated tool" in e for e in r.errors)


@pytest.mark.asyncio
async def test_missing_expected_tool_fails_even_if_phrases_match():
    # Phrase is present but the required tool was never called.
    llm = FakeLLM(const("ANSWER: found the answer"))
    rt = AgentRuntime(llm)
    agents = [make_agent("research_agent", "Research", ["web_search"])]
    scenarios = [
        TestScenario(
            name="s",
            input="find something",
            expected_outputs=["found"],
            expected_tools=["web_search"],
        )
    ]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert r.expected_found == 1
    assert r.tools_executed == []
    assert r.tool_accuracy == pytest.approx(0.0)
    assert r.passed is False


# ── Routing failure folding ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wrong_routing_fails_and_halves_score():
    llm = FakeLLM(const("ANSWER: handled"))
    rt = AgentRuntime(llm)
    agents = [
        make_agent("triage_agent", "Classify and route incoming requests"),
        make_agent("billing_agent", "Handle billing questions and refunds"),
    ]
    # Input clearly matches triage, but the scenario says billing should win.
    scenarios = [
        TestScenario(
            name="route_case",
            input="classify and route this incoming request",
            expected_outputs=["handled"],
            route_to="billing_agent",
        )
    ]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert r.routed_agent == "triage_agent"
    assert r.routing_correct is False
    assert r.passed is False
    # Base precision score 1.0, halved to 0.5 by the routing penalty.
    assert r.score == pytest.approx(0.5)
    assert any("Router selected" in e for e in r.errors)


@pytest.mark.asyncio
async def test_correct_routing_executes_declared_agent():
    llm = FakeLLM(const("ANSWER: billing handled"))
    rt = AgentRuntime(llm)
    agents = [
        make_agent("triage_agent", "Classify and route incoming requests"),
        make_agent("billing_agent", "Handle billing questions and refunds"),
    ]
    scenarios = [
        TestScenario(
            name="route_case",
            input="I have a billing question about my refund",
            expected_outputs=["billing"],
            route_to="billing_agent",
        )
    ]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert r.routed_agent == "billing_agent"
    assert r.routing_correct is True
    assert r.agent_name == "billing_agent"
    assert r.passed is True


# ── Heuristic fallback (no explicit criteria) ────────────────────────────────


@pytest.mark.asyncio
async def test_heuristic_mode_when_no_expected_outputs():
    # Agent answer first, then QA evaluator returns JSON score.
    def responder(system_prompt: str, _user: str) -> str:
        if system_prompt.startswith("You are a QA evaluator"):
            return '{"score": 0.9, "passed": true}'
        return "ANSWER: a thoughtful, detailed reply to the user"

    rt = AgentRuntime(FakeLLM(responder))
    agents = [make_agent("agent", "General assistant")]
    scenarios = [TestScenario(name="open", input="tell me about cats")]
    report = await rt.execute(agents, scenarios)
    r = report.results[0]
    assert r.mode == "heuristic"
    assert r.score == pytest.approx(0.9)
    assert r.passed is True


# ── Report shape & edge cases ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_agents_or_scenarios_returns_empty_report():
    rt = AgentRuntime(FakeLLM(const("ANSWER: ok")))
    empty_a = await rt.execute([], [TestScenario(name="s", input="x")])
    empty_s = await rt.execute([make_agent("a", "role")], [])
    assert empty_a.scenario == "empty"
    assert empty_a.results == []
    assert empty_s.scenario == "empty"


@pytest.mark.asyncio
async def test_transcripts_shape():
    llm = FakeLLM(const("ANSWER: hi there"))
    rt = AgentRuntime(llm)
    agents = [make_agent("agent", "General")]
    scenarios = [
        TestScenario(name="s", input="hello", expected_outputs=["hi"])
    ]
    report = await rt.execute(agents, scenarios)
    transcripts = report.transcripts()
    assert len(transcripts) == 1
    t = transcripts[0]
    for key in (
        "scenario", "agent", "input", "output", "passed", "score",
        "latency_ms", "mode", "tools_expected", "tools_executed",
        "routed_agent", "routing_correct", "errors",
    ):
        assert key in t
    assert report.scenarios_run == 1
    assert report.scenarios_passed == 1
