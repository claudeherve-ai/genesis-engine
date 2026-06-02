"""Tool validation stage — checks generated tools exist in the catalog.

Runs between BUILD and TEST stages to catch hallucinated tools
before they reach the test phase.

CITATION: Built to eliminate hallucinated tool generation.
Session: Hermes Agent, 2026-06-01.
BACK-LINK: /home/tedch/genesis-engine/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from genesis.models.agent import AgentDefinition, ToolConfig
from genesis.tools.catalog import validate_tool, search_catalog

logger = logging.getLogger(__name__)


@dataclass
class ToolValidationResult:
    """Validation result for an agent's tools."""
    agent_name: str
    tools_validated: int = 0
    tools_passed: int = 0
    tools_failed: int = 0
    issues: List[str] = field(default_factory=list)
    valid: bool = True


@dataclass
class ValidationReport:
    """Full validation report for all agents in a build."""
    agents_validated: int = 0
    total_tools: int = 0
    tools_passed: int = 0
    tools_failed: int = 0
    results: List[ToolValidationResult] = field(default_factory=list)
    valid: bool = True


async def validate_agent_tools(agents: List[AgentDefinition]) -> ValidationReport:
    """Validate all tools across all agents against the catalog."""
    if not agents:
        return ValidationReport(agents_validated=0, valid=True)

    results: List[ToolValidationResult] = []

    for agent in agents:
        result = ToolValidationResult(agent_name=agent.name)
        result.tools_validated = len(agent.tools)

        for tool in agent.tools:
            validation = validate_tool(tool.name, tool.schema_dict if hasattr(tool, 'schema_dict') else tool.tool_schema)
            if validation["valid"]:
                result.tools_passed += 1
            else:
                result.tools_failed += 1
                result.issues.extend(validation["issues"])
                # Suggest alternatives
                alternatives = search_catalog(tool.name)
                if alternatives:
                    alt_names = [a.name for a in alternatives[:3] if a.name != tool.name]
                    if alt_names:
                        result.issues.append(f"Did you mean: {', '.join(alt_names)}?")

        result.valid = result.tools_failed == 0
        results.append(result)

    total = sum(r.tools_validated for r in results)
    passed = sum(r.tools_passed for r in results)
    failed = sum(r.tools_failed for r in results)

    report = ValidationReport(
        agents_validated=len(agents),
        total_tools=total,
        tools_passed=passed,
        tools_failed=failed,
        results=results,
        valid=failed == 0,
    )

    if not report.valid:
        logger.warning(
            "Tool validation: %d/%d tools INVALID across %d agents",
            failed, total, len(agents),
        )
        for result in results:
            if result.issues:
                logger.warning("  %s: %s", result.agent_name, "; ".join(result.issues[:3]))
    else:
        logger.info("Tool validation: all %d tools valid across %d agents", total, len(agents))

    return report


__all__ = ["validate_agent_tools", "ToolValidationResult", "ValidationReport"]
