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

IMPORTANT: Use ONLY real, documented tools. The required tools section below
contains verified tool documentation sourced from official APIs and MCP servers.
Do NOT hallucinate tool capabilities — use only what's documented.

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
        # Research real tool capabilities via MCP + web
        tool_research = await self._research_tools(architecture)

        user_prompt = self._build_prompt(architecture, feedback, tool_research)
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
                    max_tokens=16384,  # Need more tokens for multi-agent JSON
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
        tool_research: str = "",
    ) -> str:
        """Build the user prompt including architecture details, optional
        test feedback for retry loops, and tool research for grounding.

        Args:
            architecture: The agent architecture to build from.
            feedback: Optional test results for refinement.
            tool_research: Optional tool documentation from MCP/web research.

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

        # Include tool research (MCP + web grounding)
        if tool_research:
            lines.append("")
            lines.append("=" * 40)
            lines.append("VERIFIED TOOL DOCUMENTATION (use only these real tools):")
            lines.append(tool_research)
            lines.append("=" * 40)

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

    async def _research_tools(self, architecture: AgentArchitecture) -> str:
        """Research real tool capabilities via MCP servers and web search.

        Queries MCP servers (Context7, Microsoft Docs, DeepWiki) and web
        for each tool needed by the architecture's agents. Returns
        formatted documentation to ground the LLM and prevent hallucinations.
        """
        # Collect all unique tools needed
        all_tools: set = set()
        for agent in architecture.agents:
            for tool in agent.get("tools", []):
                all_tools.add(tool)

        if not all_tools:
            return ""

        research_parts = []

        # Try MCP grounding first for key tools
        try:
            from genesis.tools.mcp_client import mcp_grounding
            for tool in list(all_tools)[:3]:  # Limit to avoid long prompts
                tool_query = f"{tool} API documentation usage examples"
                mcp_text = await mcp_grounding(tool_query)
                if mcp_text:
                    research_parts.append(mcp_text)
        except Exception as e:
            logger.debug("MCP grounding unavailable for tool research: %s", e)

        # Supplement with web search
        try:
            from genesis.tools import research_topic, format_context_for_prompt
            for tool in list(all_tools)[:5]:
                ctx = await research_topic(f"{tool} tool API reference")
                text = format_context_for_prompt(ctx, max_chars=1500)
                if text:
                    research_parts.append(text)
        except Exception as e:
            logger.debug("Web research unavailable for tools: %s", e)

        return "\n\n".join(research_parts) if research_parts else ""

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
        """Parse JSON from LLM response, repairing truncation if needed."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to repair truncated JSON (common with large multi-agent outputs)
            logger.warning("JSON parse failed at char %d, attempting repair", e.pos)
            repaired = _repair_truncated_json(text)
            if repaired is not None:
                logger.info("JSON repaired successfully")
                return repaired
            raise


def _repair_truncated_json(text: str):
    """Attempt to repair truncated JSON by closing unclosed braces/brackets."""
    import json as _json
    braces = brackets = 0
    in_string = escape = False
    for ch in text:
        if escape: escape = False; continue
        if ch == "\\": escape = True; continue
        if ch == '"' and not escape: in_string = not in_string; continue
        if in_string: continue
        if ch == "{": braces += 1
        elif ch == "}": braces -= 1
        elif ch == "[": brackets += 1
        elif ch == "]": brackets -= 1
    repair = text.rstrip().rstrip(",")
    repair += "]" * max(0, brackets)
    repair += "}" * max(0, braces)
    try:
        return _json.loads(repair)
    except _json.JSONDecodeError:
        return None
