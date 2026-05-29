"""Anthropic Claude LLM provider implementation."""

import os
from typing import Optional, Dict, Any
from anthropic import AsyncAnthropic
from genesis.llm.provider import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic's Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
    ):
        self.client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
        self.default_model = default_model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        # Anthropic doesn't support native JSON mode, so we prefix the
        # system prompt if JSON output is requested
        system = system_prompt
        prompt = user_prompt
        if response_format and response_format.get("type") == "json_object":
            system = f"{system_prompt}\n\nYou MUST respond with valid JSON only. No markdown, no explanation."

        response = await self.client.messages.create(
            model=model or self.default_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        content = response.content[0].text if response.content else ""
        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
        )

    async def health_check(self) -> bool:
        try:
            await self.complete(
                system_prompt="Respond with 'ok'",
                user_prompt="Ping",
                max_tokens=10,
            )
            return True
        except Exception:
            return False
