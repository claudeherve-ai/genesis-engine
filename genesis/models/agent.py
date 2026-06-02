"""Agent models — definitions, tools, skills, coordination, plus new models.

CITATION: Extended for Genesis Engine upgrade with validation, runtime, deployment,
and feedback models. 2026-06-01.
BACK-LINK: /home/tedch/genesis-engine/genesis/pipeline/validate_tools.py,
           /home/tedch/genesis-engine/genesis/runtime/,
           /home/tedch/genesis-engine/genesis/deployment/,
           /home/tedch/genesis-engine/genesis/feedback/
Session: Hermes Agent, 2026-06-01.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Configuration for an agent tool."""

    name: str
    description: str = ""
    tool_schema: Dict[str, Any] = Field(default_factory=dict, alias="schema")
    endpoint: Optional[str] = None
    auth_required: bool = False

    model_config = {"populate_by_name": True}


class SkillFile(BaseModel):
    """A skill file for an agent (prompt + procedures)."""

    name: str
    content: str
    category: str = "generated"


class CoordinationConfig(BaseModel):
    """How agents coordinate with each other."""

    handoff_format: str = "json"
    shared_context: List[str] = Field(default_factory=list)
    escalation_path: Optional[List[str]] = None


class AgentDefinition(BaseModel):
    """A fully defined agent ready for deployment."""

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    role: str
    system_prompt: str
    tools: List[ToolConfig] = Field(default_factory=list)
    skills: List[SkillFile] = Field(default_factory=list)
    coordination_rules: CoordinationConfig = Field(
        default_factory=CoordinationConfig
    )
    config_yaml: str = ""  # Target-platform-specific YAML

    def model_dump_safe(self) -> Dict[str, Any]:
        """Dump without secrets in config_yaml."""
        data = self.model_dump()
        if "api_key" in data.get("config_yaml", "").lower():
            data["config_yaml"] = "[REDACTED]"
        return data


class DomainModel(BaseModel):
    """Output of ANALYZE stage — structured domain understanding."""

    domain: str
    actors: List[str] = Field(default_factory=list)
    intents: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    edge_cases: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)

    @property
    def intent_count(self) -> int:
        return len(self.intents)

    @property
    def actor_count(self) -> int:
        return len(self.actors)


class AgentArchitecture(BaseModel):
    """Output of ARCHITECT stage — agent topology design."""

    topology: str = Field(
        ..., description="Router, sequential, parallel, or swarm"
    )
    agents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of agent specs: {name, role, triggers, tools, escalates_to}"
    )
    routing: Dict[str, Any] = Field(
        default_factory=dict,
        description="Routing configuration: {strategy, confidence_threshold, ...}"
    )

    @property
    def agent_count(self) -> int:
        return len(self.agents)


# ---------------------------------------------------------------------------
# NEW: Tool Validation Models (Genesis Engine Upgrade)
# ---------------------------------------------------------------------------


class ToolValidationIssue(BaseModel):
    """A single validation issue found during tool validation."""

    agent_name: str
    tool_name: str
    severity: str = "error"  # error, warning, info
    message: str
    suggestion: str = ""


class ToolValidationResult(BaseModel):
    """Result of validating agent tools against the catalog."""

    passed: bool = True
    total_tools: int = 0
    valid_tools: int = 0
    invalid_tools: int = 0
    hallucinated_tools: int = 0
    issues: List[ToolValidationIssue] = Field(default_factory=list)
    agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_tools == 0:
            return 1.0
        return self.valid_tools / self.total_tools


# ---------------------------------------------------------------------------
# NEW: Runtime Execution Models (Genesis Engine Upgrade)
# ---------------------------------------------------------------------------


class RuntimeExecutionResult(BaseModel):
    """Result of executing an agent against a test input."""

    agent_name: str
    scenario: str
    input_text: str = ""
    actual_output: str = ""
    expected_outputs_found: int = 0
    expected_outputs_total: int = 0
    tools_called: List[str] = Field(default_factory=list)
    tools_expected: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    passed: bool = False
    error: Optional[str] = None

    @property
    def precision(self) -> float:
        if not self.expected_outputs_total:
            return 1.0
        return self.expected_outputs_found / self.expected_outputs_total

    @property
    def tool_accuracy(self) -> float:
        if not self.tools_expected:
            return 1.0
        if not self.tools_called:
            return 0.0
        expected_set = set(self.tools_expected)
        called_set = set(self.tools_called)
        return len(expected_set & called_set) / len(expected_set)


class RuntimeBatchResult(BaseModel):
    """Batch result of runtime execution tests."""

    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    total_latency_ms: float = 0.0

    @property
    def overall_score(self) -> float:
        if not self.total_tests:
            return 1.0
        return self.passed_tests / self.total_tests

    @property
    def avg_latency_ms(self) -> float:
        if not self.total_tests:
            return 0.0
        return self.total_latency_ms / self.total_tests


# ---------------------------------------------------------------------------
# NEW: Deployment Package Model (Genesis Engine Upgrade)
# ---------------------------------------------------------------------------


class DeploymentPackage(BaseModel):
    """A platform-specific deployment package with configs and instructions."""

    platform: str
    build_id: str
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Filename → file content mapping"
    )
    instructions: str = ""
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    health_check: str = ""
    agent_count: int = 0


# ---------------------------------------------------------------------------
# NEW: Feedback Metrics Model (Genesis Engine Upgrade)
# ---------------------------------------------------------------------------


class AgentMetrics(BaseModel):
    """Per-agent production metrics."""

    interaction_count: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    resolution_count: int = 0
    avg_satisfaction: float = 0.0
    tool_usage: Dict[str, int] = Field(default_factory=dict)

    @property
    def resolution_rate(self) -> float:
        if not self.interaction_count:
            return 0.0
        return self.resolution_count / self.interaction_count

    @property
    def error_rate(self) -> float:
        if not self.interaction_count:
            return 0.0
        return self.error_count / self.interaction_count


class FeedbackMetrics(BaseModel):
    """Production feedback metrics with improvement suggestions."""

    build_id: str = ""
    deployment_id: str = ""
    period_start: float = 0.0
    period_end: float = 0.0
    total_interactions: int = 0
    resolution_rate: float = 0.0
    user_satisfaction: float = 0.0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_per_interaction: float = 0.0
    agent_breakdown: Dict[str, AgentMetrics] = Field(default_factory=dict)
    common_issues: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
