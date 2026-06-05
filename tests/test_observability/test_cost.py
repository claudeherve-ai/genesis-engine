"""Tests for cost/token tracking and budget enforcement."""

import pytest

from genesis.llm.provider import LLMProvider, LLMResponse
from genesis.observability.cost import (
    BudgetExceeded,
    CostTracker,
    CostTrackingProvider,
    StageBudget,
    estimate_cost,
    set_active_context,
)


def test_estimate_cost_known_model():
    # gpt-4o: (0.0025, 0.01) per 1K tokens.
    cost = estimate_cost("gpt-4o", 1000, 1000)
    assert cost == pytest.approx(0.0025 + 0.01)


def test_estimate_cost_prefix_match():
    cost = estimate_cost("gpt-4o-2024-08-06", 1000, 0)
    assert cost == pytest.approx(0.0025)


def test_estimate_cost_unknown_uses_default():
    # _DEFAULT_PRICE = (0.003, 0.015)
    cost = estimate_cost("some-random-model", 1000, 1000)
    assert cost == pytest.approx(0.003 + 0.015)


def test_record_and_snapshot():
    tracker = CostTracker(budget=StageBudget())
    tracker.record("gpt-4o", 1000, 500, stage="BUILD", build_id="b1")
    snap = tracker.snapshot(build_id="b1")
    assert snap["total"]["calls"] == 1
    assert snap["total"]["prompt_tokens"] == 1000
    assert snap["total"]["completion_tokens"] == 500
    assert snap["total"]["total_tokens"] == 1500
    assert "gpt-4o" in snap["by_model"]
    assert "BUILD" in snap["by_stage"]
    assert snap["build"]["calls"] == 1


def test_build_budget_exceeded():
    tracker = CostTracker(budget=StageBudget(build_usd=0.001))
    with pytest.raises(BudgetExceeded):
        tracker.record("gpt-4o", 10000, 10000, build_id="b1")
    # Usage still recorded before raising.
    assert tracker.snapshot(build_id="b1")["build"]["calls"] == 1


def test_stage_budget_exceeded():
    tracker = CostTracker(budget=StageBudget(stage_usd=0.0001))
    with pytest.raises(BudgetExceeded):
        tracker.record("gpt-4o", 10000, 10000, stage="ANALYZE")


def test_no_budget_when_zero():
    tracker = CostTracker(budget=StageBudget())
    for _ in range(50):
        tracker.record("gpt-4o", 10000, 10000, build_id="b1", stage="BUILD")
    # Never raises.
    assert tracker.snapshot()["total"]["calls"] == 50


def test_reset():
    tracker = CostTracker(budget=StageBudget())
    tracker.record("gpt-4o", 100, 100)
    tracker.reset()
    assert tracker.snapshot()["total"]["calls"] == 0


class _FakeLLM(LLMProvider):
    def __init__(self, usage):
        self._usage = usage

    async def complete(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="hi", model="gpt-4o", usage=self._usage)

    async def health_check(self) -> bool:
        return True


async def test_cost_tracking_provider_meters_completion():
    tracker = CostTracker(budget=StageBudget())
    inner = _FakeLLM({"prompt_tokens": 200, "completion_tokens": 100})
    provider = CostTrackingProvider(inner, tracker=tracker)
    set_active_context(stage="ARCHITECT", build_id="b42")
    resp = await provider.complete("sys", "user")
    assert resp.content == "hi"
    snap = tracker.snapshot(build_id="b42")
    assert snap["total"]["prompt_tokens"] == 200
    assert snap["total"]["completion_tokens"] == 100
    assert "ARCHITECT" in snap["by_stage"]


async def test_cost_tracking_provider_handles_input_output_keys():
    tracker = CostTracker(budget=StageBudget())
    inner = _FakeLLM({"input_tokens": 50, "output_tokens": 25})
    provider = CostTrackingProvider(inner, tracker=tracker)
    await provider.complete("sys", "user")
    snap = tracker.snapshot()
    assert snap["total"]["prompt_tokens"] == 50
    assert snap["total"]["completion_tokens"] == 25


async def test_cost_tracking_provider_health_passthrough():
    inner = _FakeLLM({})
    provider = CostTrackingProvider(inner, tracker=CostTracker(budget=StageBudget()))
    assert await provider.health_check() is True
