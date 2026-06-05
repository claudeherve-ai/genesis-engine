"""Runtime sandbox — actually EXECUTE generated agents against test inputs.

This replaces opinion-based "LLM judges the agent" testing with a real,
instrumented runtime that:

1. Routes each scenario to an agent using a deterministic router (so routing
   failures surface instead of being masked by "best agent wins").
2. Exposes the agent's *allowed* tools and parses the structured tool calls the
   agent actually emits (``CALL <tool> {<json args>}``).
3. Independently validates each call against the catalog and the agent's own
   tool config, then executes a deterministic handler — so ``tools_executed``
   reflects what the runtime *actually ran*, not what the model claims.
4. Scores from reality: required output substrings must appear, required tools
   must actually run, and the router must pick the declared agent.

When a scenario carries no explicit success criteria (legacy free-form string),
the runtime falls back to a heuristic/LLM quality score and labels the run
honestly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from genesis.llm.provider import LLMProvider
from genesis.models.agent import AgentDefinition, TestScenario
from genesis.tools.catalog import get_tool

logger = logging.getLogger(__name__)

# Matches a structured tool call the agent emits, e.g.:
#   CALL web_search {"query": "azure pricing"}
#   CALL send_email {}
_CALL_RE = re.compile(
    r"CALL\s+([a-zA-Z_][a-zA-Z0-9_\-]*)\s*(\{.*?\})?",
    re.IGNORECASE,
)

_PASS_THRESHOLD = 0.6
_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "for", "of", "and", "or", "my",
    "i", "me", "you", "it", "this", "that", "with", "on", "in", "at",
    "can", "do", "does", "how", "what", "please", "need", "want", "help",
}

ScenarioInput = Union[str, Dict[str, Any], TestScenario]


@dataclass
class ExecutionResult:
    """Result of running a single scenario against the routed agent."""

    agent_name: str
    scenario_name: str
    input_text: str
    output_text: str = ""
    passed: bool = False
    score: float = 0.0
    latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    tools_executed: List[str] = field(default_factory=list)
    expected_found: int = 0
    expected_total: int = 0
    route_to: Optional[str] = None
    routed_agent: str = ""
    routing_correct: bool = True
    mode: str = "execution"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Back-compat alias used by some readers expecting the old field name.
    @property
    def test_scenario(self) -> str:
        return self.scenario_name

    @property
    def precision(self) -> float:
        if not self.expected_total:
            return 1.0
        return self.expected_found / self.expected_total

    @property
    def tool_accuracy(self) -> float:
        if not self.tools_expected:
            return 1.0
        expected = set(self.tools_expected)
        executed = set(self.tools_executed)
        return len(expected & executed) / len(expected)

    def to_transcript(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "agent": self.agent_name,
            "input": self.input_text,
            "output": self.output_text,
            "passed": self.passed,
            "score": round(self.score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "mode": self.mode,
            "expected_found": self.expected_found,
            "expected_total": self.expected_total,
            "tools_expected": list(self.tools_expected),
            "tools_executed": list(self.tools_executed),
            "route_to": self.route_to,
            "routed_agent": self.routed_agent,
            "routing_correct": self.routing_correct,
            "errors": list(self.errors),
        }


@dataclass
class RuntimeReport:
    """Full execution report for a multi-agent system."""

    scenario: str
    results: List[ExecutionResult] = field(default_factory=list)
    overall_score: float = 0.0
    total_latency_ms: float = 0.0
    agents_executed: int = 0
    agents_passed: int = 0
    executed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def scenarios_run(self) -> int:
        return len(self.results)

    @property
    def scenarios_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def transcripts(self) -> List[Dict[str, Any]]:
        return [r.to_transcript() for r in self.results]


class AgentRuntime:
    """Sandboxed agent execution runtime.

    Takes agent definitions and concrete test scenarios, actually runs the
    routed agent with its system prompt, executes the tools it calls, and
    measures real, deterministic performance.
    """

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def execute(
        self,
        agents: List[AgentDefinition],
        scenarios: List[ScenarioInput],
        *,
        timeout_per_agent: float = 30.0,
    ) -> RuntimeReport:
        """Route and execute every scenario, scoring from reality."""
        if not agents or not scenarios:
            return RuntimeReport(scenario="empty")

        normalized = [self._normalize(s, i) for i, s in enumerate(scenarios)]
        logger.info(
            "Executing %d scenarios across %d agents...",
            len(normalized), len(agents),
        )

        results: List[ExecutionResult] = []
        for scenario in normalized:
            results.append(
                await self._run_scenario(agents, scenario, timeout_per_agent)
            )

        total_score = sum(r.score for r in results) / max(len(results), 1)
        passed = sum(1 for r in results if r.passed)
        total_latency = sum(r.latency_ms for r in results)

        return RuntimeReport(
            scenario=normalized[0].name[:50],
            results=results,
            overall_score=round(total_score, 4),
            total_latency_ms=round(total_latency, 2),
            agents_executed=len(agents),
            agents_passed=passed,
        )

    # ------------------------------------------------------------------
    # Scenario normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(scenario: ScenarioInput, index: int) -> TestScenario:
        """Coerce str | dict | TestScenario into a TestScenario."""
        if isinstance(scenario, TestScenario):
            return scenario
        if isinstance(scenario, dict):
            data = dict(scenario)
            text = data.get("input") or data.get("scenario") or ""
            name = data.get("name") or (text[:50] or f"scenario_{index + 1}")
            return TestScenario(
                name=name,
                input=text,
                expected_outputs=data.get("expected_outputs", []) or [],
                expected_tools=data.get("expected_tools", []) or [],
                route_to=data.get("route_to"),
            )
        # Plain string (legacy free-form scenario)
        text = str(scenario)
        return TestScenario(name=text[:50] or f"scenario_{index + 1}", input=text)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 1}

    def _route(
        self,
        input_text: str,
        agents: List[AgentDefinition],
    ) -> AgentDefinition:
        """Deterministically pick the agent that best matches the input.

        Scores token overlap between the input and each agent's name + role.
        Ties (including zero-overlap) fall back to the first agent — which is
        the conventional default/fallback slot.
        """
        input_tokens = self._tokens(input_text)
        best = agents[0]
        best_score = -1.0
        for agent in agents:
            agent_tokens = self._tokens(f"{agent.name} {agent.role}")
            overlap = float(len(input_tokens & agent_tokens))
            if overlap > best_score:
                best_score = overlap
                best = agent
        return best

    # ------------------------------------------------------------------
    # Per-scenario execution
    # ------------------------------------------------------------------

    async def _run_scenario(
        self,
        agents: List[AgentDefinition],
        scenario: TestScenario,
        timeout: float,
    ) -> ExecutionResult:
        routed = self._route(scenario.input, agents)

        # Decide which agent actually executes (the declared correct one, if any).
        target = routed
        routing_correct = True
        if scenario.route_to:
            routing_correct = routed.name == scenario.route_to
            declared = next(
                (a for a in agents if a.name == scenario.route_to), None
            )
            if declared is not None:
                target = declared

        result = await self._execute_agent(target, scenario, timeout)
        result.route_to = scenario.route_to
        result.routed_agent = routed.name
        result.routing_correct = routing_correct

        # Fold routing failure into pass/score so it can't be hidden.
        if scenario.route_to and not routing_correct:
            result.passed = False
            result.score = round(result.score * 0.5, 4)
            result.errors.append(
                f"Router selected '{routed.name}' but scenario expects "
                f"'{scenario.route_to}'"
            )
        return result

    async def _execute_agent(
        self,
        agent: AgentDefinition,
        scenario: TestScenario,
        timeout: float,
    ) -> ExecutionResult:
        start = time.monotonic()
        allowed = {t.name for t in agent.tools}
        result = ExecutionResult(
            agent_name=agent.name,
            scenario_name=scenario.name,
            input_text=scenario.input,
            tools_expected=list(scenario.expected_tools),
            expected_total=len(scenario.expected_outputs),
        )

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    system_prompt=agent.system_prompt,
                    user_prompt=self._build_user_prompt(agent, scenario.input),
                    temperature=0.3,
                    max_tokens=1024,
                ),
                timeout=timeout,
            )
            raw = (response.content or "").strip()
            answer = self._extract_answer(raw)
            result.output_text = answer[:1000]

            tools_executed, tool_errors = self._run_tool_calls(raw, agent, allowed)
            result.tools_executed = tools_executed
            result.errors.extend(tool_errors)

            result.latency_ms = (time.monotonic() - start) * 1000

            if scenario.expected_outputs:
                # Deterministic requirement checks.
                found = sum(
                    1 for exp in scenario.expected_outputs
                    if exp.lower() in answer.lower()
                )
                result.expected_found = found
                precision = found / len(scenario.expected_outputs)
                if scenario.expected_tools:
                    result.score = round(
                        0.6 * precision + 0.4 * result.tool_accuracy, 4
                    )
                else:
                    result.score = round(precision, 4)

                precision_ok = precision >= _PASS_THRESHOLD
                tools_ok = (
                    not scenario.expected_tools
                    or set(scenario.expected_tools) <= set(tools_executed)
                )
                result.passed = precision_ok and tools_ok
                result.mode = "execution"
            else:
                # No explicit criteria — fall back to a quality heuristic.
                score, passed = await self._score_output(
                    agent, scenario.input, answer
                )
                result.score = score
                result.passed = passed
                result.mode = "heuristic"

            return result

        except asyncio.TimeoutError:
            result.latency_ms = (time.monotonic() - start) * 1000
            result.output_text = "[TIMEOUT]"
            result.errors.append(f"Agent timed out after {timeout}s")
            return result
        except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
            result.latency_ms = (time.monotonic() - start) * 1000
            result.output_text = "[ERROR]"
            result.errors.append(str(e))
            logger.warning(
                "Agent '%s' execution error on scenario '%s': %s",
                agent.name, scenario.name, e,
            )
            return result

    # ------------------------------------------------------------------
    # Tool execution (instrumented)
    # ------------------------------------------------------------------

    def _run_tool_calls(
        self,
        raw_output: str,
        agent: AgentDefinition,
        allowed: set[str],
    ) -> tuple[List[str], List[str]]:
        """Parse, validate, and execute the tool calls the agent emitted.

        Returns ``(tools_executed, errors)``. A tool is recorded as executed
        only if it is (a) declared in the agent's config and (b) present in the
        catalog and (c) its deterministic handler runs without raising.
        """
        executed: List[str] = []
        errors: List[str] = []

        for match in _CALL_RE.finditer(raw_output):
            name = match.group(1)
            args_blob = match.group(2)

            if name not in allowed:
                errors.append(
                    f"Agent called undeclared tool '{name}' "
                    f"(not in its tool config)"
                )
                continue

            catalog_tool = get_tool(name)
            if catalog_tool is None:
                errors.append(
                    f"Agent called hallucinated tool '{name}' "
                    f"(not in catalog)"
                )
                continue

            args: Dict[str, Any] = {}
            if args_blob:
                try:
                    args = json.loads(args_blob)
                except json.JSONDecodeError:
                    errors.append(f"Tool '{name}' called with invalid JSON args")
                    continue

            try:
                self._execute_tool_handler(name, args)
            except Exception as e:  # noqa: BLE001
                errors.append(f"Tool '{name}' handler failed: {e}")
                continue

            if name not in executed:
                executed.append(name)

        return executed, errors

    @staticmethod
    def _execute_tool_handler(name: str, args: Dict[str, Any]) -> str:
        """Deterministic mock handler for a catalog tool.

        Real network-backed execution is intentionally out of scope for the
        sandbox; this records that the call was *actually dispatched* with
        validated arguments and returns a synthetic observation.
        """
        return f"[{name}] executed with args={json.dumps(args, sort_keys=True)}"

    # ------------------------------------------------------------------
    # Prompting & parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(agent: AgentDefinition, scenario: str) -> str:
        if agent.tools:
            tool_lines = "\n".join(
                f"  - {t.name}: {t.description}" for t in agent.tools
            )
            tool_block = (
                "You have access to these tools:\n"
                f"{tool_lines}\n\n"
                "To use a tool, output a line exactly like:\n"
                '  CALL <tool_name> {<json args>}\n'
                "You may call multiple tools (one per line). After any tool "
                "calls, write your final reply on a line beginning with "
                "'ANSWER:'.\n\n"
            )
        else:
            tool_block = (
                "You have no tools. Write your reply on a line beginning with "
                "'ANSWER:'.\n\n"
            )

        return (
            f"You are agent '{agent.name}' ({agent.role}).\n\n"
            f"{tool_block}"
            f"User input: {scenario}\n"
        )

    @staticmethod
    def _extract_answer(raw: str) -> str:
        """Return the text after the last 'ANSWER:' marker, or the whole body.

        CALL lines are stripped so the scored answer doesn't accidentally
        match expected substrings via the tool-call syntax itself.
        """
        marker = "ANSWER:"
        idx = raw.rfind(marker)
        body = raw[idx + len(marker):] if idx != -1 else raw
        cleaned = "\n".join(
            line for line in body.splitlines()
            if not line.strip().upper().startswith("CALL ")
        )
        return cleaned.strip()

    # ------------------------------------------------------------------
    # Heuristic / LLM scoring (legacy fallback)
    # ------------------------------------------------------------------

    async def _score_output(
        self,
        agent: AgentDefinition,
        scenario: str,
        output: str,
    ) -> tuple[float, bool]:
        """Score output quality when no deterministic criteria exist."""
        if not output or output in ("[TIMEOUT]", "[ERROR]"):
            return 0.0, False

        try:
            response = await self._llm.complete(
                system_prompt=(
                    "You are a QA evaluator. Score the agent's response from "
                    "0.0 to 1.0. Return JSON: "
                    '{"score": 0.8, "passed": true}. Pass threshold: 0.6.'
                ),
                user_prompt=(
                    f"Agent: {agent.name} ({agent.role})\n"
                    f"User input: {scenario}\n"
                    f"Agent response: {output[:1000]}\n\nScore this response."
                ),
                temperature=0.1,
                max_tokens=200,
            )
            text = response.content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            passed = bool(data.get("passed", score >= _PASS_THRESHOLD))
            return round(score, 4), passed
        except Exception as e:  # noqa: BLE001
            logger.debug("Heuristic score fallback: %s", e)
            has_content = len(output) > 20
            relevant = any(
                kw in output.lower() for kw in scenario.lower().split()[:3]
            )
            score = 0.5 if has_content else 0.2
            score += 0.2 if relevant else 0.0
            return round(score, 4), score >= _PASS_THRESHOLD


__all__ = ["AgentRuntime", "ExecutionResult", "RuntimeReport"]
