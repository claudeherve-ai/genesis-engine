"""Tests for multi-model routing — graceful degradation + key-gated secondaries."""

from typing import Optional, Dict, Any

import pytest

from genesis.llm.provider import LLMProvider, LLMResponse
from genesis.llm.multi_model import (
    ModelRegistry,
    create_model_registry,
    get_stage_model,
    STAGE_MODEL_PREFERENCES,
    PipelineStage_,
)


class _FakePrimary(LLMProvider):
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        return LLMResponse(content="{}", model="fake", usage={})

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def primary() -> _FakePrimary:
    return _FakePrimary()


def _clear_keys(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_primary_only_when_no_secondary_keys(monkeypatch, primary):
    _clear_keys(monkeypatch)
    registry = create_model_registry(primary)

    assert registry.model_count == 1
    assert registry.available_models == ["openai"]
    # Every stage routes to the primary.
    for stage in (
        PipelineStage_.ANALYZE,
        PipelineStage_.ARCHITECT,
        PipelineStage_.BUILD,
        PipelineStage_.TEST,
    ):
        assert get_stage_model(registry, stage) is primary


def test_anthropic_registers_when_key_present(monkeypatch, primary):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    registry = create_model_registry(primary)

    assert "anthropic" in registry.available_models
    assert registry.model_count == 2
    # ARCHITECT prefers anthropic when available.
    assert get_stage_model(registry, PipelineStage_.ARCHITECT) is not primary


def test_placeholder_key_does_not_register(monkeypatch, primary):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "PLACEHOLDER")
    registry = create_model_registry(primary)

    assert registry.available_models == ["openai"]


def test_missing_provider_module_skips_cleanly(monkeypatch, primary):
    # Gemini/DeepSeek provider modules do not exist; setting their keys must
    # NOT crash registry creation — the ImportError is caught and logged.
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test-123")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d-test-123")
    registry = create_model_registry(primary)

    # Only the primary registered; no exception raised.
    assert registry.available_models == ["openai"]


def test_get_stage_model_falls_back_to_primary(monkeypatch, primary):
    _clear_keys(monkeypatch)
    registry = create_model_registry(primary)
    # An unknown stage still resolves to a usable provider.
    assert get_stage_model(registry, "nonexistent_stage") is primary


def test_registry_fallback_excludes_named_model(primary):
    registry = ModelRegistry()
    secondary = _FakePrimary()
    registry.register("openai", primary, [PipelineStage_.BUILD])
    registry.register("anthropic", secondary, [PipelineStage_.BUILD])

    fb = registry.get_fallback(PipelineStage_.BUILD, exclude="openai")
    assert fb is secondary


def test_stage_preferences_cover_all_stages():
    for stage in (
        PipelineStage_.ANALYZE,
        PipelineStage_.ARCHITECT,
        PipelineStage_.BUILD,
        PipelineStage_.TEST,
        PipelineStage_.VERIFY,
        PipelineStage_.DEPLOY,
    ):
        assert stage in STAGE_MODEL_PREFERENCES
