"""Stage 3: Prompt & Tool Generation.

Takes an AgentArchitecture and uses an LLM to generate fully specified
AgentDefinition objects, including system prompts, tool configurations,
skill files, coordination rules, and target-platform YAML.

Supports an optional feedback parameter from the TEST stage for retry loops.
When feedback is provided, the BUILD stage incorporates specific failure
details to improve generated agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from genesis.llm.provider import LLMProvider
from genesis.models.agent import (
    AgentArchitecture,
    AgentDefinition,
    ToolConfig,
    SkillFile,
    CoordinationConfig,
)

logger = logging.getLogger(__name__)

BUILD_SYSTEM_PROMPT = """You are a prompt engineer and tool designer for multi-agent
systems. Given an agent architecture specification, generate fully specified
agent definitions ready for deployment.

CRITICAL RULE: Generate ALL agents listed in the architecture. If the architecture
specifies 3 agents (e.g., triage, technical, billing), you MUST output all 3.
Do not skip, merge, or collapse agents.

Output valid JSON only — an object with an "agents" array:

{
  "agents": [{
    "name": "snake_case_agent_name",
    "role": "concise role description",
    "system_prompt": "detailed system prompt with instructions, tone, and boundaries",
    "tools": [
      {
        "name": "tool_name",
        "description": "what this tool does",
        "schema": {"type": "object", "properties": {}},
        "endpoint": null,
        "auth_required": false
      }
    ],
    "skills": [
      {
        "name": "skill_name",
        "content": "detailed skill instructions/procedures",
        "category": "generated"
      }
    ],
    "coordination_rules": {
      "handoff_format": "json",
      "shared_context": ["context_item_1"],
      "escalation_path": ["next_agent_name"]
    },
    "config_yaml": "# target-platform YAML config string"
  }]
}

Guidelines:
- System prompts should be detailed (2-4 paragraphs), defining the agent's
  personality, capabilities, boundaries, and response format
- Tools should have realistic schemas with proper JSON Schema definitions
- Skills should cover specific procedural knowledge the agent needs
- coordination_rules should match the architecture's topology and escalation paths
- config_yaml should be a valid YAML string for the target platform
- The "agents" array MUST contain exactly one entry per agent in the architecture"""

STRICTER_RETRY_PROMPT = """You are a prompt engineer and tool designer for multi-agent
systems. Given an agent architecture specification, generate fully specified
agent definitions ready for deployment.

You MUST output ONLY valid JSON — a JSON array. Do not wrap in markdown code
fences. Do not include explanatory text. Output exactly a JSON array of objects
with keys: name, role, system_prompt, tools, skills, coordination_rules, config_yaml.

Guidelines:
- System prompts should be detailed (2-4 paragraphs)
- Tools should have realistic JSON Schema definitions
- Skills should cover specific procedural knowledge
- coordination_rules should include handoff_format, shared_context, escalation_path
- config_yaml should be a valid YAML string for agentsystem-compatible format"""

MAX_RETRIES = 3


class BuildStage:
    """Stage 3: Generate agent prompts, tools, and skills from architecture.

    Uses an LLM to produce fully specified AgentDefinition objects including
    system prompts, tool configurations, skill files, coordination rules,
    and target-platform YAML. Supports a feedback-driven retry loop from the
    TEST stage.

    Args:
        llm: An LLMProvider instance for making completion requests.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def run(
        self,
        architecture: AgentArchitecture,
        feedback: Optional[Any] = None,
    ) -> List[AgentDefinition]:
        """Run the BUILD stage.

        Args:
            architecture: Agent topology from the ARCHITECT stage.
            feedback: Optional TestResults or failure details from the TEST
                stage, used to refine agent definitions during retry loops.

        Returns:
            A list of fully defined AgentDefinition instances.

        Raises:
            ValueError: If the LLM fails to produce valid JSON after all retries.
        """
        user_prompt = self._build_prompt(architecture, feedback)
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                system_prompt = (
                    BUILD_SYSTEM_PROMPT
                    if attempt == 1
                    else STRICTER_RETRY_PROMPT
                )
                response = await self.llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.5,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
                data = self._parse_json(response.content)

                # Response is always a JSON object: {"agents": [...]}
                if not isinstance(data, dict):
                    raise ValueError(
                        f"Expected a JSON object with 'agents' key, got {type(data).__name__}"
                    )
                raw_agents = data.get("agents")
                if not isinstance(raw_agents, list):
                    raise ValueError(
                        "Expected 'agents' key containing an array of agent definitions"
                    )

                agents = [self._build_agent_definition(a) for a in raw_agents]
                logger.info(
                    "BUILD stage complete — generated %d agents: %s",
                    len(agents),
                    [a.name for a in agents],
                )
                return agents
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                last_error = e
                logger.warning(
                    "BUILD stage parse failure (attempt %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    e,
                )

        raise ValueError(
            f"BUILD stage failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        architecture: AgentArchitecture,
        feedback: Optional[Any],
    ) -> str:
        """Build the user prompt including architecture details and optional
        test feedback for retry loops.

        Args:
            architecture: The agent architecture to build from.
            feedback: Optional test results for refinement.

        Returns:
            Formatted prompt string.
        """
        lines = [
            f"Topology: {architecture.topology}",
            "",
            "Agents to build:",
        ]

        for a in architecture.agents:
            lines.append(
                f"  - {a.get('name', 'unnamed')}: {a.get('role', 'no role')}"
            )
            lines.append(f"    Triggers: {a.get('triggers', [])}")
            lines.append(f"    Tools needed: {a.get('tools', [])}")
            lines.append(f"    Escalates to: {a.get('escalates_to', 'none')}")
            lines.append("")

        lines.append(f"Routing strategy: {architecture.routing.get('strategy', 'intent_based')}")
        fallback = architecture.routing.get("fallback_agent")
        if fallback:
            lines.append(f"Fallback agent: {fallback}")

        # Include test feedback if in retry loop
        if feedback is not None:
            lines.append("")
            lines.append("=" * 40)
            lines.append("TEST FEEDBACK — fix these issues:")
            feedback_text = self._format_feedback(feedback)
            lines.append(feedback_text)

        return "\n".join(lines)

    def _format_feedback(self, feedback: Any) -> str:
        """Format test feedback for inclusion in the BUILD prompt.

        Args:
            feedback: TestResults or failure data.

        Returns:
            Formatted feedback string.
        """
        # Try to extract structured feedback
        try:
            if hasattr(feedback, "failures"):
                lines = [f"Overall score: {getattr(feedback, 'overall_score', 'N/A')}"]
                for f in getattr(feedback, "failures", []):
                    if hasattr(f, "scenario"):
                        lines.append(
                            f"  - [{f.agent}] Scenario '{f.scenario}': "
                            f"expected '{f.expected}', got '{f.actual}' "
                            f"(metric: {f.metric})"
                        )
                    elif isinstance(f, dict):
                        lines.append(
                            f"  - [{f.get('agent', '?')}] "
                            f"Scenario '{f.get('scenario', '?')}': "
                            f"expected '{f.get('expected', '?')}', "
                            f"got '{f.get('actual', '?')}'"
                        )
                return "\n".join(lines)

            if isinstance(feedback, dict):
                return json.dumps(feedback, indent=2)

            return str(feedback)
        except Exception:
            return str(feedback)

    def _build_agent_definition(self, raw: dict[str, Any]) -> AgentDefinition:
        """Construct an AgentDefinition from a raw dict.

        Validates and normalizes the dict into proper nested models.

        Args:
            raw: Raw agent data from the LLM.

        Returns:
            A validated AgentDefinition instance.
        """
        # Build tools
        tools: list[ToolConfig] = []
        for t in raw.get("tools", []):
            if isinstance(t, dict):
                tools.append(ToolConfig(**t))
            elif isinstance(t, ToolConfig):
                tools.append(t)

        # Build skills
        skills: list[SkillFile] = []
        for s in raw.get("skills", []):
            if isinstance(s, dict):
                skills.append(SkillFile(**s))
            elif isinstance(s, SkillFile):
                skills.append(s)

        # Build coordination rules
        coord_raw = raw.get("coordination_rules", {})
        if isinstance(coord_raw, CoordinationConfig):
            coordination = coord_raw
        elif isinstance(coord_raw, dict):
            coordination = CoordinationConfig(**coord_raw)
        else:
            coordination = CoordinationConfig()

        return AgentDefinition(
            name=raw["name"],
            role=raw["role"],
            system_prompt=raw["system_prompt"],
            tools=tools,
            skills=skills,
            coordination_rules=coordination,
            config_yaml=raw.get("config_yaml", ""),
        )

    @staticmethod
    def _parse_json(content: str) -> Any:
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
