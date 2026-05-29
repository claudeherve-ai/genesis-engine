"""LLM provider abstraction layer."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Structured response from an LLM provider."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base for LLM providers (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Send a completion request and return a structured response."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the provider is reachable and authenticated."""
        ...
