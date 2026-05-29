"""Stage 1: Domain Analysis.

Takes a natural-language problem description and uses an LLM to produce
a structured DomainModel with actors, intents, constraints, edge cases,
and success criteria.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from genesis.llm.provider import LLMProvider
from genesis.models.agent import DomainModel

logger = logging.getLogger(__name__)

ANALYZE_SYSTEM_PROMPT = """You are a domain analyst. Given a problem description,
decompose it into a structured domain model. Output valid JSON only.

Return format:
{
  "domain": "short domain name",
  "actors": ["list of actors/roles involved"],
  "intents": [{"actor": "actor_name", "intent": "what they want", "priority": "high|medium|low"}],
  "constraints": ["list of constraints or requirements"],
  "edge_cases": ["list of edge cases to handle"],
  "success_criteria": ["how to know if the system is working"]
}"""

STRICTER_RETRY_PROMPT = """You are a domain analyst. Given a problem description,
decompose it into a structured domain model. You MUST output ONLY valid JSON.
Do not wrap the JSON in markdown code fences (no ```json). Do not include any
explanatory text before or after the JSON object. Output exactly a JSON object
with these keys: domain, actors, intents, constraints, edge_cases, success_criteria.

Return format:
{
  "domain": "short domain name",
  "actors": ["list of actors/roles involved"],
  "intents": [{"actor": "actor_name", "intent": "what they want", "priority": "high|medium|low"}],
  "constraints": ["list of constraints or requirements"],
  "edge_cases": ["list of edge cases to handle"],
  "success_criteria": ["how to know if the system is working"]
}"""

MAX_RETRIES = 3


class AnalyzeStage:
    """Stage 1: Decompose a problem description into a DomainModel.

    Uses an LLM to extract actors, intents, constraints, edge cases, and
    success criteria from natural language. Retries up to MAX_RETRIES times
    on JSON parse failure, escalating the system prompt each time.

    Args:
        llm: An LLMProvider instance for making completion requests.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def run(self, problem_description: str) -> DomainModel:
        """Run the ANALYZE stage.

        Args:
            problem_description: Natural language description of the problem.

        Returns:
            A validated DomainModel instance.

        Raises:
            ValueError: If the LLM fails to produce valid JSON after all retries.
        """
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                system_prompt = (
                    ANALYZE_SYSTEM_PROMPT if attempt == 1 else STRICTER_RETRY_PROMPT
                )
                response = await self.llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=problem_description,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                data = self._parse_json(response.content)
                domain_model = DomainModel(**data)
                logger.info(
                    "ANALYZE stage complete — domain=%s, actors=%d, intents=%d",
                    domain_model.domain,
                    domain_model.actor_count,
                    domain_model.intent_count,
                )
                return domain_model
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                last_error = e
                logger.warning(
                    "ANALYZE stage parse failure (attempt %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    e,
                )

        raise ValueError(
            f"ANALYZE stage failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present.

        Args:
            content: Raw text content from the LLM.

        Returns:
            Parsed JSON as a dict.

        Raises:
            json.JSONDecodeError: If the content is not valid JSON.
        """
        text = content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove opening fence (```json or ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
