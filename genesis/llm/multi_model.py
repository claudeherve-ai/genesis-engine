"""Multi-model pipeline — different models for different cognitive tasks.

Routes each pipeline stage to a model suited to that cognitive task, with
graceful degradation to the primary provider when secondaries are absent.

- ANALYZE:   primary (reasoning) → Claude / Gemini if configured
- ARCHITECT: Claude (creativity) → primary
- BUILD:     primary (generation) → Claude / DeepSeek if configured
- TEST:      Gemini (evaluation) → primary

The primary provider (Azure OpenAI) is always registered. Anthropic Claude
is a real, first-class secondary that activates when ANTHROPIC_API_KEY is set.
Gemini and DeepSeek are honest extension points — drop in a provider module
(genesis/llm/gemini.py / deepseek.py) and set the key to enable them.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from genesis.llm.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage model assignments
# ---------------------------------------------------------------------------

class PipelineStage_:
    ANALYZE = "analyze"
    ARCHITECT = "architect"
    BUILD = "build"
    TEST = "test"
    VERIFY = "verify"
    DEPLOY = "deploy"


# Primary: Azure OpenAI (always registered). Secondaries activate by key.

class ModelRegistry:
    """Registry of available models with their specializations."""

    def __init__(self):
        self._models: Dict[str, LLMProvider] = {}
        self._specialties: Dict[str, List[str]] = {}  # stage → [model_names]

    def register(self, name: str, provider: LLMProvider, stages: List[str]):
        self._models[name] = provider
        self._specialties[name] = stages
        logger.info("Registered model '%s' for stages: %s", name, stages)

    def get_primary(self, stage: str) -> LLMProvider:
        """Get the primary model for a stage."""
        candidates = [
            (n, p) for n, p in self._models.items()
            if stage in self._specialties.get(n, [])
        ]
        if candidates:
            return candidates[0][1]
        # Fallback to first available
        if self._models:
            return list(self._models.values())[0]
        raise RuntimeError(f"No models available for stage '{stage}'")

    def get_fallback(self, stage: str, exclude: str) -> Optional[LLMProvider]:
        """Get a fallback model for a stage, excluding the given model."""
        candidates = [
            (n, p) for n, p in self._models.items()
            if n != exclude and stage in self._specialties.get(n, [])
        ]
        return candidates[0][1] if candidates else None

    @property
    def available_models(self) -> List[str]:
        return list(self._models.keys())

    @property
    def model_count(self) -> int:
        return len(self._models)


def create_model_registry(primary: LLMProvider) -> ModelRegistry:
    """Create a model registry with the primary provider + any configured secondaries.

    Primary: the OpenAI/Azure provider you already have (always registered).
    Secondaries: real providers that activate when their API key is present
    (Anthropic today; Gemini/DeepSeek when their provider modules are added).
    """
    registry = ModelRegistry()

    # Primary model — always available
    registry.register("openai", primary, [
        PipelineStage_.ANALYZE,
        PipelineStage_.ARCHITECT,
        PipelineStage_.BUILD,
        PipelineStage_.TEST,
        PipelineStage_.VERIFY,
        PipelineStage_.DEPLOY,
    ])

    # Secondary model: Anthropic Claude — a real, first-class provider in
    # genesis. Activates automatically when ANTHROPIC_API_KEY is set.
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and "PLACEHOLDER" not in anthropic_key:
        try:
            from genesis.llm.anthropic import AnthropicProvider

            registry.register("anthropic", AnthropicProvider(), [
                PipelineStage_.ANALYZE, PipelineStage_.ARCHITECT,
                PipelineStage_.BUILD,
            ])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Anthropic provider unavailable: %s", exc)

    # Gemini / DeepSeek are honest extension points: a provider module is
    # imported only if it exists. Drop a `genesis/llm/gemini.py` (exporting
    # GeminiProvider) and set GEMINI_API_KEY to light it up — no code change
    # here required. Until then these branches are inert no-ops.
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and "PLACEHOLDER" not in gemini_key:
        try:
            from genesis.llm.gemini import GeminiProvider  # type: ignore

            registry.register("gemini", GeminiProvider(), [
                PipelineStage_.ANALYZE, PipelineStage_.TEST,
            ])
        except ImportError:
            logger.info(
                "GEMINI_API_KEY set but genesis.llm.gemini not installed — "
                "skipping (add the provider module to enable)."
            )

    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    if deepseek_key and "PLACEHOLDER" not in deepseek_key:
        try:
            from genesis.llm.deepseek import DeepSeekProvider  # type: ignore

            registry.register("deepseek", DeepSeekProvider(), [
                PipelineStage_.ANALYZE, PipelineStage_.BUILD,
            ])
        except ImportError:
            logger.info(
                "DEEPSEEK_API_KEY set but genesis.llm.deepseek not installed — "
                "skipping (add the provider module to enable)."
            )

    logger.info(
        "Model registry: %d models — %s",
        registry.model_count,
        ", ".join(registry.available_models),
    )
    return registry


# ---------------------------------------------------------------------------
# Stage-specific model selection
# ---------------------------------------------------------------------------

# Which model is best for each cognitive task
STAGE_MODEL_PREFERENCES: Dict[str, List[str]] = {
    PipelineStage_.ANALYZE: ["openai", "anthropic", "gemini"],
    PipelineStage_.ARCHITECT: ["anthropic", "openai"],
    PipelineStage_.BUILD: ["openai", "anthropic", "deepseek"],
    PipelineStage_.TEST: ["gemini", "openai"],
    PipelineStage_.VERIFY: ["openai"],
    PipelineStage_.DEPLOY: ["openai"],
}


def get_stage_model(registry: ModelRegistry, stage: str) -> LLMProvider:
    """Get the best available model for a stage."""
    preferences = STAGE_MODEL_PREFERENCES.get(stage, ["openai"])
    for model_name in preferences:
        if model_name in registry.available_models:
            return registry._models[model_name]
    return registry.get_primary(stage)


__all__ = [
    "ModelRegistry", "create_model_registry",
    "get_stage_model", "STAGE_MODEL_PREFERENCES",
    "PipelineStage_",
]
