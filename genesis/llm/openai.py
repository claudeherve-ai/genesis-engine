"""Azure Foundry (Azure OpenAI) LLM provider implementation."""

import os
from typing import Optional, Dict, Any
from openai import AsyncAzureOpenAI
from genesis.llm.provider import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """LLM provider using Azure Foundry (Azure OpenAI Service) with gpt-5.4."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        default_model: str = "gpt-5.4",
    ):
        self.client = AsyncAzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint or os.getenv(
                "AZURE_OPENAI_ENDPOINT",
                "https://tedcherve-6038-resource.cognitiveservices.azure.com/"
            ),
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
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
        kwargs: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        content = choice.message.content or ""
        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
