"""Multi-model pipeline — different models for different cognitive tasks.

Uses specialized models per pipeline stage to avoid single-model bias:
- ANALYZE: GPT-5.4 (reasoning) + Claude (placeholder) + Gemini (placeholder)
- ARCHITECT: Claude (creativity) + GPT-5.4
- BUILD: GPT-5.4 (generation) + Claude (placeholder)
- TEST: Gemini (evaluation) + GPT-5.4

CITATION: Built to avoid single-model pipeline bias.
Session: Hermes Agent, 2026-06-01.
BACK-LINK: /home/tedch/genesis-engine/
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


# Primary: what we have (OpenAI/Azure via gpt-5.4)
# Secondary: placeholders for Anthropic Claude, Google Gemini, DeepSeek

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
    """Create a model registry with primary + placeholder secondaries.

    Primary: the OpenAI/Azure provider you already have.
    Secondaries: placeholders that activate when you add API keys.
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

    # Check for secondary models
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and "PLACEHOLDER" not in anthropic_key:
        try:
            from oracle.prediction.ensemble import AnthropicProvider
            registry.register("anthropic", AnthropicProvider(), [
                PipelineStage_.ANALYZE, PipelineStage_.ARCHITECT,
                PipelineStage_.BUILD,
            ])
        except ImportError:
            logger.debug("Anthropic provider not available")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and "PLACEHOLDER" not in gemini_key:
        try:
            from oracle.prediction.ensemble import GeminiProvider
            registry.register("gemini", GeminiProvider(), [
                PipelineStage_.ANALYZE, PipelineStage_.TEST,
            ])
        except ImportError:
            logger.debug("Gemini provider not available")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    if deepseek_key and "PLACEHOLDER" not in deepseek_key:
        try:
            from oracle.prediction.ensemble import DeepSeekProvider
            registry.register("deepseek", DeepSeekProvider(), [
                PipelineStage_.ANALYZE, PipelineStage_.BUILD,
            ])
        except ImportError:
            logger.debug("DeepSeek provider not available")

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
