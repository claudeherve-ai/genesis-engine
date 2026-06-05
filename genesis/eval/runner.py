"""Evaluation regression suite — score produced agents against golden expectations.

The runner drives the *real* BUILD stage with an injected LLM provider (a
deterministic scripted fake in CI, a real model in the cloud) and scores the
agents it produces against a golden dataset: required agent roles, required
tools, and a zero-hallucination guarantee (no tool outside the verified
catalog). It is dependency-injected end-to-end so tests never reach for a
global provider that would require API keys.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from genesis.llm.provider import LLMProvider
from genesis.models.agent import AgentArchitecture
from genesis.pipeline.build import BuildStage
from genesis.tools.catalog import TOOL_CATALOG

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


@dataclass
class EvalCaseResult:
    """Score for a single golden case."""

    case_name: str
    role_coverage: float
    tool_coverage: float
    hallucinated_tools: List[str]
    matched_roles: List[str]
    missing_roles: List[str]
    matched_tools: List[str]
    missing_tools: List[str]
    agent_count: int
    passed: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case_name,
            "role_coverage": round(self.role_coverage, 4),
            "tool_coverage": round(self.tool_coverage, 4),
            "hallucinated_tools": self.hallucinated_tools,
            "matched_roles": self.matched_roles,
            "missing_roles": self.missing_roles,
            "matched_tools": self.matched_tools,
            "missing_tools": self.missing_tools,
            "agent_count": self.agent_count,
            "passed": self.passed,
            "error": self.error,
        }


@dataclass
class EvalReport:
    """Aggregate report across a suite of golden cases."""

    results: List[EvalCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 1.0

    @property
    def regressions(self) -> List[str]:
        return [r.case_name for r in self.results if not r.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "regressions": self.regressions,
            "results": [r.to_dict() for r in self.results],
        }


def _matches_role(expected: str, produced_roles: List[str]) -> bool:
    needle = expected.lower().strip()
    return any(needle in role.lower() for role in produced_roles)


class EvalRunner:
    """Run golden eval cases against produced agents.

    Parameters
    ----------
    llm:
        An LLM provider (injected). In offline CI this is a scripted fake so
        evaluation is deterministic; in production it can be a real model.
    min_role_coverage / min_tool_coverage:
        Pass thresholds (default 1.0 — every required role/tool must appear).
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        min_role_coverage: float = 1.0,
        min_tool_coverage: float = 1.0,
    ) -> None:
        self.llm = llm
        self.min_role_coverage = min_role_coverage
        self.min_tool_coverage = min_tool_coverage

    async def evaluate(self, case: Dict[str, Any]) -> EvalCaseResult:
        name = case.get("name", "unnamed")
        expected_roles: List[str] = case.get("expected_roles", [])
        expected_tools: List[str] = case.get("expected_tools", [])

        try:
            architecture = AgentArchitecture(**case["architecture"])
            agents = await BuildStage(self.llm).run(architecture)
        except Exception as exc:  # pragma: no cover - defensive
            return EvalCaseResult(
                case_name=name,
                role_coverage=0.0,
                tool_coverage=0.0,
                hallucinated_tools=[],
                matched_roles=[],
                missing_roles=expected_roles,
                matched_tools=[],
                missing_tools=expected_tools,
                agent_count=0,
                passed=False,
                error=str(exc),
            )

        produced_roles = [f"{a.name} {a.role}" for a in agents]
        produced_tools: List[str] = []
        for agent in agents:
            produced_tools.extend(t.name for t in agent.tools)
        produced_tool_set = set(produced_tools)

        matched_roles = [r for r in expected_roles if _matches_role(r, produced_roles)]
        missing_roles = [r for r in expected_roles if r not in matched_roles]
        matched_tools = [t for t in expected_tools if t in produced_tool_set]
        missing_tools = [t for t in expected_tools if t not in produced_tool_set]

        hallucinated = sorted(
            {t for t in produced_tool_set if t not in TOOL_CATALOG}
        )

        role_coverage = (
            len(matched_roles) / len(expected_roles) if expected_roles else 1.0
        )
        tool_coverage = (
            len(matched_tools) / len(expected_tools) if expected_tools else 1.0
        )

        passed = (
            role_coverage >= self.min_role_coverage
            and tool_coverage >= self.min_tool_coverage
            and not hallucinated
            and len(agents) >= case.get("min_agents", 1)
        )

        return EvalCaseResult(
            case_name=name,
            role_coverage=role_coverage,
            tool_coverage=tool_coverage,
            hallucinated_tools=hallucinated,
            matched_roles=matched_roles,
            missing_roles=missing_roles,
            matched_tools=matched_tools,
            missing_tools=missing_tools,
            agent_count=len(agents),
            passed=passed,
        )

    async def run_suite(self, cases: List[Dict[str, Any]]) -> EvalReport:
        report = EvalReport()
        for case in cases:
            report.results.append(await self.evaluate(case))
        return report


def load_datasets(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load every ``*.json`` golden case from the datasets directory."""
    directory = directory or DATASETS_DIR
    cases: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            cases.append(json.load(fh))
    return cases
