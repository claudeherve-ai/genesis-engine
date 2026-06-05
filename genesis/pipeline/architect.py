"""Stage 2: Agent Topology Design.

Takes a DomainModel and uses an LLM to design the agent architecture:
topology type, agent count and responsibilities, communication patterns,
tool assignments, and escalation paths.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from genesis.llm.provider import LLMProvider
from genesis.models.agent import DomainModel, AgentArchitecture

logger = logging.getLogger(__name__)

ARCHITECT_SYSTEM_PROMPT = """You are a systems architect specializing in multi-agent
systems. Given a domain model and the original problem description, design the
optimal agent topology.

CRITICAL RULE: If the user explicitly asks for multiple agents (e.g., "triage,
tech support, AND billing"), you MUST create a separate agent for EACH distinct
function. Do not collapse multiple functions into a single agent.

Output valid JSON only with this structure:
{
  "topology": "router|sequential|parallel|swarm",
  "agents": [
    {
      "name": "agent_name_snake_case",
      "role": "role description",
      "triggers": ["what triggers this agent"],
      "tools": ["tools this agent needs"],
      "escalates_to": "next agent or null"
    }
  ],
  "routing": {
    "strategy": "intent_based|round_robin|llm_judge",
    "confidence_threshold": 0.7,
    "fallback_agent": "agent_name or null"
  },
  "test_scenarios": [
    {
      "name": "short_scenario_name",
      "input": "a realistic user message that exercises the system",
      "expected_outputs": ["substring that MUST appear in the reply"],
      "expected_tools": ["tool the agent MUST actually call, or omit"],
      "route_to": "the agent that SHOULD handle this input"
    }
  ]
}

Design principles:
- Each agent should have a single, clear responsibility
- Create separate agents for each distinct function or department mentioned
  (e.g., triage, technical support, billing, sales, HR — each gets its own agent)
- Only combine functions if they are truly identical in scope
- Define clear escalation paths for edge cases
- Choose topology: router for triage, sequential for pipelines, parallel for
  independent tasks, swarm for collaborative problem-solving
- Include a default/fallback agent for unrecognized intents
- Provide 5-10 concrete test_scenarios grounded in the domain's intents and
  success criteria. Each scenario MUST be independently checkable: name a
  routing target (route_to), and where possible list expected_outputs
  (exact substrings) and expected_tools (tools that must actually run)."""

STRICTER_RETRY_PROMPT = """You are a systems architect specializing in multi-agent
systems. Given a domain model, design the optimal agent topology.

You MUST output ONLY valid JSON. Do not wrap the JSON in markdown code fences
(no ```json). Do not include any explanatory text. Output exactly a JSON object
with keys: topology, agents (array of objects), routing (object).

Design principles:
- Each agent should have a single, clear responsibility
- Minimize agent count while maintaining separation of concerns
- Define clear escalation paths for edge cases
- Choose topology: router for triage, sequential for pipelines, parallel for
  independent tasks, swarm for collaborative problem-solving"""

MAX_RETRIES = 3


class ArchitectStage:
    """Stage 2: Design agent topology from a domain model.

    Uses an LLM to determine the optimal multi-agent topology, agent
    roles and responsibilities, communication patterns, and escalation
    paths. Retries up to MAX_RETRIES times on JSON parse failure.

    Args:
        llm: An LLMProvider instance for making completion requests.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def run(
        self,
        domain_model: DomainModel,
        original_problem: str = "",
    ) -> AgentArchitecture:
        """Run the ARCHITECT stage.

        Args:
            domain_model: Structured domain analysis from ANALYZE stage.
            original_problem: The original problem description for context.

        Returns:
            A validated AgentArchitecture instance.

        Raises:
            ValueError: If the LLM fails to produce valid JSON after all retries.
        """
        user_prompt = self._build_prompt(domain_model, original_problem)
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                system_prompt = (
                    ARCHITECT_SYSTEM_PROMPT
                    if attempt == 1
                    else STRICTER_RETRY_PROMPT
                )
                response = await self.llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
                data = self._parse_json(response.content)
                architecture = AgentArchitecture(**data)
                if not architecture.test_scenarios:
                    architecture.test_scenarios = self._synthesize_scenarios(
                        domain_model, architecture
                    )
                logger.info(
                    "ARCHITECT stage complete — topology=%s, agents=%d, "
                    "scenarios=%d",
                    architecture.topology,
                    architecture.agent_count,
                    len(architecture.test_scenarios),
                )
                return architecture
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                last_error = e
                logger.warning(
                    "ARCHITECT stage parse failure (attempt %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    e,
                )

        raise ValueError(
            f"ARCHITECT stage failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _build_prompt(
        domain_model: DomainModel,
        original_problem: str = "",
    ) -> str:
        """Build the user prompt from a DomainModel.

        Args:
            domain_model: The domain model to serialize.
            original_problem: The original problem description.

        Returns:
            A formatted prompt string for the LLM.
        """
        lines = [
            f"Domain: {domain_model.domain}",
        ]
        if original_problem:
            lines.append(f"Original request: {original_problem}")
        lines.extend([
            "",
            "Actors:",
            *[f"  - {a}" for a in domain_model.actors],
            "",
            "Intents:",
            *[
                f"  - [{i.get('priority', 'medium').upper()}] {i.get('actor', '?')}: "
                f"{i.get('intent', '?')}"
                for i in domain_model.intents
            ],
            "",
            "Constraints:",
            *[f"  - {c}" for c in domain_model.constraints],
            "",
            "Edge Cases:",
            *[f"  - {e}" for e in domain_model.edge_cases],
            "",
            "Success Criteria:",
            *[f"  - {s}" for s in domain_model.success_criteria],
        ])
        return "\n".join(lines)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    # ------------------------------------------------------------------
    # Deterministic scenario synthesis (fallback when LLM omits scenarios)
    # ------------------------------------------------------------------

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "to", "for", "of", "and", "or", "my",
        "i", "me", "you", "it", "this", "that", "with", "on", "in", "at",
        "can", "do", "does", "how", "what", "please", "need", "want", "help",
        "agent", "user", "request", "handle", "handles", "system",
    }

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        import re

        words = re.findall(r"[a-z0-9]+", str(text).lower())
        return {w for w in words if w not in cls._STOPWORDS and len(w) > 1}

    @classmethod
    def _match_agent(
        cls,
        intent: dict[str, Any],
        agents: list[dict[str, Any]],
    ) -> str | None:
        """Pick the ground-truth agent for an intent via trigger/role overlap.

        This is intentionally richer than the runtime router (which only sees
        name + role) so the synthesized ``route_to`` acts as an independent
        ground truth the router is tested against.
        """
        if not agents:
            return None
        intent_tokens = cls._tokens(
            f"{intent.get('intent', '')} {intent.get('actor', '')}"
        )
        best = agents[0].get("name")
        best_score = -1.0
        for agent in agents:
            triggers = " ".join(agent.get("triggers", []) or [])
            agent_tokens = cls._tokens(
                f"{agent.get('name', '')} {agent.get('role', '')} {triggers}"
            )
            overlap = float(len(intent_tokens & agent_tokens))
            if overlap > best_score:
                best_score = overlap
                best = agent.get("name")
        return best

    @classmethod
    def _synthesize_scenarios(
        cls,
        domain_model: DomainModel,
        architecture: AgentArchitecture,
    ) -> list[dict[str, Any]]:
        """Build concrete, checkable scenarios directly from the domain model.

        Each domain intent becomes one routing scenario whose ``route_to`` is
        the best-matching agent by declared triggers. ``expected_outputs`` and
        ``expected_tools`` are deliberately left empty here — the synthesizer
        will not fabricate success substrings it cannot guarantee, so these
        runs exercise real routing + real execution and are scored honestly
        (heuristic mode) rather than against invented criteria.
        """
        agents = architecture.agents
        if not agents:
            return []

        scenarios: list[dict[str, Any]] = []
        seen_inputs: set[str] = set()
        for idx, intent in enumerate(domain_model.intents[:10]):
            intent_text = (intent.get("intent") or "").strip()
            if not intent_text or intent_text.lower() in seen_inputs:
                continue
            seen_inputs.add(intent_text.lower())
            route_to = cls._match_agent(intent, agents)
            scenarios.append({
                "name": f"intent_{idx + 1}"[:50],
                "input": intent_text,
                "expected_outputs": [],
                "expected_tools": [],
                "route_to": route_to,
            })
        return scenarios
