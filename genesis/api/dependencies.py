"""FastAPI dependency injection."""

import os
from functools import lru_cache
from genesis.storage.database import get_session
from genesis.storage.repository import ProjectRepository, BuildRepository
from genesis.llm.openai import OpenAIProvider
from genesis.llm.anthropic import AnthropicProvider
from genesis.llm.provider import LLMProvider
from genesis.adapters.base import DeploymentTarget
from genesis.adapters.agentsystem import create_agentsystem_adapter


@lru_cache()
def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider.

    Uses Azure Foundry (Azure OpenAI) with AZURE_OPENAI_API_KEY.
    """
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return OpenAIProvider()
    raise RuntimeError(
        "No LLM provider configured. Set AZURE_OPENAI_API_KEY."
    )


@lru_cache()
def get_deployment_target() -> DeploymentTarget:
    """Get the configured deployment target (AgentSystem)."""
    return create_agentsystem_adapter()


def get_project_repo():
    """Get a ProjectRepository with a fresh session."""
    session = get_session()
    try:
        yield ProjectRepository(session)
    finally:
        session.close()


def get_build_repo():
    """Get a BuildRepository with a fresh session."""
    session = get_session()
    try:
        yield BuildRepository(session)
    finally:
        session.close()
