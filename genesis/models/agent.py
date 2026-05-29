"""Agent models — definitions, tools, skills, coordination."""

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
        # Redact any API keys in config_yaml
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
