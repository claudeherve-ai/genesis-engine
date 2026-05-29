"""Stage 4: Simulation & Evaluation.

Takes a list of AgentDefinition objects and uses an LLM to generate test
scenarios, run simulated interactions, and produce TestResults with a
quality score and detailed failure information for the BUILD feedback loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

from genesis.llm.provider import LLMProvider
from genesis.models.agent import AgentDefinition
from genesis.models.test_results import TestResults, TestFailure

logger = logging.getLogger(__name__)

TEST_SYSTEM_PROMPT = """You are a QA engineer evaluating a multi-agent system.
Given a set of agent definitions, generate test scenarios and evaluate the
agents' expected behavior.

Output valid JSON only:

{
  "scenarios_run": <int>,
  "scenarios_passed": <int>,
  "overall_score": <float 0.0-1.0>,
  "metrics": {
    "intent_classification": <float>,
    "resolution_rate": <float>,
    "handoff_correctness": <float>,
    "tool_usage": <float>,
    "response_quality": <float>,
    "escalation_propriety": <float>,
    "coordination_efficiency": <float>
  },
  "failures": [
    {
      "scenario": "description of the test scenario",
      "agent": "agent_name",
      "expected": "what should have happened",
      "actual": "what the assessment found",
      "metric": "which metric failed"
    }
  ],
  "passed": <bool>
}

Evaluation guidelines:
- Generate 8-15 test scenarios covering normal flows, edge cases, and error paths
- Score each metric from 0.0 to 1.0
- overall_score should be a weighted average reflecting real-world priorities:
  resolution_rate and handoff_correctness are most important
- For each failure, provide specific, actionable feedback the BUILD stage can use
- Be critical but fair — flag real issues, not nitpicks
- Check that escalation paths are correctly followed
- Verify that tool schemas are appropriate for agent responsibilities
- Ensure system prompts are clear, scoped, and include appropriate boundaries"""

STRICTER_RETRY_PROMPT = """You are a QA engineer evaluating a multi-agent system.
Given a set of agent definitions, generate test scenarios and evaluate the
agents' expected behavior.

You MUST output ONLY valid JSON. Do not wrap in markdown code fences.
Do not include explanatory text. Output exactly a JSON object with keys:
scenarios_run, scenarios_passed, overall_score, metrics, failures, passed.

Be critical but fair — flag real issues, not nitpicks. Provide specific,
actionable feedback in each failure entry."""

MAX_RETRIES = 3


class TestStage:
    """Stage 4: Simulate and evaluate a multi-agent system.

    Uses an LLM to generate test scenarios, run simulated evaluations,
    and produce TestResults with a quality score. The results include
    detailed failure information that can be fed back to the BUILD stage
    for refinement.

    Args:
        llm: An LLMProvider instance for making completion requests.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def run(self, agents: List[AgentDefinition]) -> TestResults:
        """Run the TEST stage.

        Args:
            agents: List of agent definitions to evaluate.

        Returns:
            A TestResults instance with scores and failure details.

        Raises:
            ValueError: If the LLM fails to produce valid JSON after all retries.
        """
        user_prompt = self._build_prompt(agents)
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                system_prompt = (
                    TEST_SYSTEM_PROMPT
                    if attempt == 1
                    else STRICTER_RETRY_PROMPT
                )
                response = await self.llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
                data = self._parse_json(response.content)
                test_results = self._build_test_results(data)
                logger.info(
                    "TEST stage complete — score=%.2f, passed=%s, "
                    "scenarios=%d/%d, failures=%d",
                    test_results.overall_score,
                    test_results.passed,
                    test_results.scenarios_passed,
                    test_results.scenarios_run,
                    test_results.failure_count,
                )
                return test_results
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                last_error = e
                logger.warning(
                    "TEST stage parse failure (attempt %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    e,
                )

        raise ValueError(
            f"TEST stage failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(agents: List[AgentDefinition]) -> str:
        """Build a prompt that describes the agents for evaluation.

        Args:
            agents: Agent definitions to describe.

        Returns:
            Formatted prompt string.
        """
        lines = ["Evaluate the following multi-agent system:\n"]
        lines.append(f"Topology has {len(agents)} agents:\n")

        for i, agent in enumerate(agents, 1):
            lines.append(f"{'=' * 50}")
            lines.append(f"Agent {i}: {agent.name}")
            lines.append(f"  Role: {agent.role}")
            lines.append("")
            lines.append("  System Prompt:")
            lines.append(f"    {agent.system_prompt[:500]}")
            lines.append("")
            lines.append(f"  Tools ({len(agent.tools)}):")
            for tool in agent.tools:
                lines.append(f"    - {tool.name}: {tool.description}")
            lines.append(f"  Skills ({len(agent.skills)}):")
            for skill in agent.skills:
                lines.append(f"    - {skill.name}")
            coord = agent.coordination_rules
            lines.append(f"  Handoff format: {coord.handoff_format}")
            if coord.escalation_path:
                lines.append(
                    f"  Escalation path: {' -> '.join(coord.escalation_path)}"
                )
            if coord.shared_context:
                lines.append(f"  Shared context: {', '.join(coord.shared_context)}")
            lines.append("")

        lines.append(
            "Generate test scenarios and score this system. Focus on whether "
            "the agents collectively handle their domain correctly, whether "
            "handoffs make sense, and whether each agent's prompt is clear "
            "and scoped appropriately."
        )

        return "\n".join(lines)

    @staticmethod
    def _build_test_results(data: dict[str, Any]) -> TestResults:
        """Build a TestResults object from parsed JSON data.

        Args:
            data: Parsed JSON from the LLM response.

        Returns:
            A validated TestResults instance.
        """
        failures: list[TestFailure] = []
        for f in data.get("failures", []):
            if isinstance(f, dict):
                failures.append(TestFailure(**f))
            elif isinstance(f, TestFailure):
                failures.append(f)

        results = TestResults(
            scenarios_run=data.get("scenarios_run", 0),
            scenarios_passed=data.get("scenarios_passed", 0),
            overall_score=data.get("overall_score", 0.0),
            metrics=data.get("metrics", {}),
            failures=failures,
            passed=data.get("passed", False),
        )
        return results

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
