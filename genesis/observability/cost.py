"""Token usage and cost tracking with per-stage budget enforcement.

A process-wide :class:`CostTracker` accumulates token usage and estimated
dollar cost per model, per pipeline stage, and per build. The
:class:`CostTrackingProvider` wraps any :class:`~genesis.llm.provider.LLMProvider`
so every completion is metered transparently — no changes to call sites
required. Pricing is a best-effort table (USD per 1K tokens) covering the
common Azure OpenAI / OpenAI / Anthropic models, with a safe default for
unknown models.

Budgets are opt-in via environment variables:

* ``GENESIS_BUILD_BUDGET_USD`` — hard cap per build (raises
  :class:`BudgetExceeded` when exceeded). ``0`` / unset disables the cap.
* ``GENESIS_STAGE_BUDGET_USD`` — hard cap per pipeline stage.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from genesis.llm.provider import LLMProvider, LLMResponse

# USD per 1,000 tokens — (prompt, completion). Best-effort, October-2024 list.
_PRICING: Dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-35-turbo": (0.0005, 0.0015),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.06),
    "o1-mini": (0.003, 0.012),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-haiku": (0.0008, 0.004),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
}

# Fallback price for an unrecognised model (mid-range, avoids under-counting).
_DEFAULT_PRICE = (0.003, 0.015)

# Tracks the active pipeline stage for the current async context so the
# wrapped provider can attribute usage without threading the value through
# every call signature.
_active_stage: ContextVar[Optional[str]] = ContextVar("genesis_active_stage", default=None)
_active_build: ContextVar[Optional[str]] = ContextVar("genesis_active_build", default=None)


def _price_for(model: str) -> tuple[float, float]:
    name = (model or "").lower()
    if name in _PRICING:
        return _PRICING[name]
    # Prefix/contains match (e.g. "gpt-4o-2024-08-06").
    for key, price in _PRICING.items():
        if key in name:
            return price
    return _DEFAULT_PRICE


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single completion."""
    p_in, p_out = _price_for(model)
    return (prompt_tokens / 1000.0) * p_in + (completion_tokens / 1000.0) * p_out


class BudgetExceeded(RuntimeError):
    """Raised when a build or stage exceeds its configured USD budget."""


@dataclass
class _Counter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class StageBudget:
    """Resolved per-build / per-stage USD budgets."""

    build_usd: float = 0.0
    stage_usd: float = 0.0

    @classmethod
    def from_env(cls) -> "StageBudget":
        def _read(name: str) -> float:
            try:
                return max(0.0, float(os.getenv(name, "0")))
            except ValueError:
                return 0.0

        return cls(
            build_usd=_read("GENESIS_BUILD_BUDGET_USD"),
            stage_usd=_read("GENESIS_STAGE_BUDGET_USD"),
        )


class CostTracker:
    """Thread-safe accumulator of token usage and cost."""

    def __init__(self, budget: Optional[StageBudget] = None):
        self._lock = threading.Lock()
        self._total = _Counter()
        self._by_model: Dict[str, _Counter] = defaultdict(_Counter)
        self._by_stage: Dict[str, _Counter] = defaultdict(_Counter)
        self._by_build: Dict[str, _Counter] = defaultdict(_Counter)
        self.budget = budget or StageBudget.from_env()

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        stage: Optional[str] = None,
        build_id: Optional[str] = None,
    ) -> float:
        """Record a completion and return its estimated cost.

        Raises :class:`BudgetExceeded` if a configured budget would be
        surpassed (the usage is still recorded first so dashboards stay
        accurate).
        """
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        with self._lock:
            self._total.add(prompt_tokens, completion_tokens, cost)
            self._by_model[model or "unknown"].add(prompt_tokens, completion_tokens, cost)
            if stage:
                self._by_stage[stage].add(prompt_tokens, completion_tokens, cost)
            if build_id:
                self._by_build[build_id].add(prompt_tokens, completion_tokens, cost)

            build_cost = self._by_build[build_id].cost_usd if build_id else 0.0
            stage_cost = self._by_stage[stage].cost_usd if stage else 0.0

        if self.budget.build_usd and build_id and build_cost > self.budget.build_usd:
            raise BudgetExceeded(
                f"Build {build_id} exceeded budget "
                f"(${build_cost:.4f} > ${self.budget.build_usd:.4f})."
            )
        if self.budget.stage_usd and stage and stage_cost > self.budget.stage_usd:
            raise BudgetExceeded(
                f"Stage {stage} exceeded budget "
                f"(${stage_cost:.4f} > ${self.budget.stage_usd:.4f})."
            )
        return cost

    def snapshot(self, build_id: Optional[str] = None) -> Dict[str, Any]:
        """Return a JSON-serialisable view of accumulated usage."""
        with self._lock:
            data: Dict[str, Any] = {
                "total": self._total.as_dict(),
                "by_model": {k: v.as_dict() for k, v in self._by_model.items()},
                "by_stage": {k: v.as_dict() for k, v in self._by_stage.items()},
                "budget": {
                    "build_usd": self.budget.build_usd,
                    "stage_usd": self.budget.stage_usd,
                },
            }
            if build_id is not None:
                data["build"] = self._by_build.get(build_id, _Counter()).as_dict()
            return data

    def reset(self) -> None:
        with self._lock:
            self._total = _Counter()
            self._by_model.clear()
            self._by_stage.clear()
            self._by_build.clear()


_tracker: Optional[CostTracker] = None
_tracker_lock = threading.Lock()


def get_cost_tracker() -> CostTracker:
    """Return the process-wide cost tracker (created on first use)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = CostTracker()
    return _tracker


class CostTrackingProvider(LLMProvider):
    """Decorator that meters every completion of a wrapped provider.

    Stage and build attribution is read from context variables set by the
    orchestrator (:func:`set_active_context`), so existing call sites need no
    changes.
    """

    def __init__(self, inner: LLMProvider, tracker: Optional[CostTracker] = None):
        self._inner = inner
        self._tracker = tracker or get_cost_tracker()

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = await self._inner.complete(*args, **kwargs)
        usage = response.usage or {}
        prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        self._tracker.record(
            response.model,
            prompt,
            completion,
            stage=_active_stage.get(),
            build_id=_active_build.get(),
        )
        return response

    async def health_check(self) -> bool:
        return await self._inner.health_check()

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._inner, item)


def set_active_context(*, stage: Optional[str] = None, build_id: Optional[str] = None) -> None:
    """Set the stage/build attribution for cost tracking in this context."""
    if stage is not None:
        _active_stage.set(stage)
    if build_id is not None:
        _active_build.set(build_id)
