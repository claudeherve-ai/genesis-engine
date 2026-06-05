"""Offline test for the BUILD anti-hallucination gate (Phase 2).

A scripted FakeLLM first emits an agent declaring a hallucinated tool, then a
clean agent on retry. The gate must detect the fake tool, retry with correction
feedback, and ultimately ship only real catalog tools. A second scenario keeps
emitting the fake tool on every attempt and asserts the stage strips it rather
than shipping a fake tool.

Architecture agents declare NO tools so ``_research_tools`` short-circuits and
no network call is made — fully offline.
"""

from __future__ import annotations

import json
from typing import Callable, List

import pytest

from genesis.llm.provider import LLMResponse
from genesis.models.agent import AgentArchitecture
from genesis.pipeline.build import BuildStage


class FakeLLM:
    def __init__(self, responder: Callable[[int], str]) -> None:
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
        idx = len(self.calls)
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return LLMResponse(content=self._responder(idx), model="fake", usage={})


def _agent_json(tool_name: str) -> str:
    return json.dumps({
        "agents": [{
            "name": "researcher",
            "role": "research things",
            "system_prompt": "You research.",
            "tools": [{
                "name": tool_name,
                "description": "a tool",
                "schema": {"type": "object", "properties": {}},
                "endpoint": None,
                "auth_required": False,
            }],
            "skills": [],
            "coordination_rules": {"handoff_format": "json", "shared_context": [], "escalation_path": []},
            "config_yaml": "name: researcher",
        }]
    })


def _architecture() -> AgentArchitecture:
    # Empty tools → _research_tools returns "" with no network.
    return AgentArchitecture(
        topology="router",
        agents=[{"name": "researcher", "role": "research things", "tools": []}],
    )


@pytest.mark.asyncio
async def test_gate_retries_then_accepts_real_tool():
    # Attempt 1: hallucinated tool. Attempt 2: real tool.
    def responder(idx: int) -> str:
        return _agent_json("magic_oracle_tool") if idx == 0 else _agent_json("web_search")

    llm = FakeLLM(responder)
    agents = await BuildStage(llm).run(_architecture())

    assert len(llm.calls) == 2  # retried once
    # The correction feedback must have been injected on the retry.
    assert "does not exist" in llm.calls[1]["user"]
    tool_names = [t.name for t in agents[0].tools]
    assert tool_names == ["web_search"]


@pytest.mark.asyncio
async def test_gate_strips_fake_tool_when_unfixable():
    # Always emit the fake tool — gate must strip it on the final attempt.
    llm = FakeLLM(lambda _idx: _agent_json("magic_oracle_tool"))
    agents = await BuildStage(llm).run(_architecture())

    assert len(llm.calls) == 3  # exhausted MAX_RETRIES
    # Fake tool stripped → agent ships with zero (but only real) tools.
    assert all(t.name != "magic_oracle_tool" for t in agents[0].tools)
    assert agents[0].tools == []


@pytest.mark.asyncio
async def test_gate_passes_real_tool_first_try():
    llm = FakeLLM(lambda _idx: _agent_json("web_search"))
    agents = await BuildStage(llm).run(_architecture())

    assert len(llm.calls) == 1  # no retry needed
    assert [t.name for t in agents[0].tools] == ["web_search"]
