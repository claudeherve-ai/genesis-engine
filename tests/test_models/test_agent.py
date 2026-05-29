"""Tests for Agent models — ToolConfig, SkillFile, CoordinationConfig,
AgentDefinition, DomainModel, AgentArchitecture."""

import pytest
from pydantic import ValidationError

from genesis.models.agent import (
    ToolConfig,
    SkillFile,
    CoordinationConfig,
    AgentDefinition,
    DomainModel,
    AgentArchitecture,
)


# ── ToolConfig ──────────────────────────────────────────────────────────────

class TestToolConfig:
    def test_creation_minimal(self):
        tc = ToolConfig(name="web_search")
        assert tc.name == "web_search"
        assert tc.description == ""
        assert tc.tool_schema == {}
        assert tc.endpoint is None
        assert tc.auth_required is False

    def test_creation_full(self):
        tc = ToolConfig(
            name="calc",
            description="Runs calculations",
            tool_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
            endpoint="https://api.example.com/calc",
            auth_required=True,
        )
        assert tc.name == "calc"
        assert tc.description == "Runs calculations"
        assert tc.tool_schema == {"type": "object", "properties": {"expr": {"type": "string"}}}
        assert tc.endpoint == "https://api.example.com/calc"
        assert tc.auth_required is True

    def test_description_default(self):
        tc = ToolConfig(name="t")
        assert tc.description == ""

    def test_schema_default(self):
        tc = ToolConfig(name="t")
        assert tc.tool_schema == {}

    def test_endpoint_default(self):
        tc = ToolConfig(name="t")
        assert tc.endpoint is None

    def test_endpoint_is_optional(self):
        tc = ToolConfig(name="t")
        tc2 = ToolConfig(name="t", endpoint="http://x")
        assert tc.endpoint is None
        assert tc2.endpoint == "http://x"

    def test_model_dump(self):
        tc = ToolConfig(name="test", auth_required=True)
        d = tc.model_dump()
        assert d["name"] == "test"
        assert d["auth_required"] is True
        assert d["tool_schema"] == {}


# ── SkillFile ───────────────────────────────────────────────────────────────

class TestSkillFile:
    def test_creation(self):
        sf = SkillFile(name="customer_service", content="You are a helpful agent...")
        assert sf.name == "customer_service"
        assert sf.content == "You are a helpful agent..."
        assert sf.category == "generated"

    def test_category_default(self):
        sf = SkillFile(name="s", content="c")
        assert sf.category == "generated"

    def test_custom_category(self):
        sf = SkillFile(name="s", content="c", category="handwritten")
        assert sf.category == "handwritten"

    def test_empty_content(self):
        sf = SkillFile(name="empty", content="")
        assert sf.content == ""

    def test_model_dump(self):
        sf = SkillFile(name="skill1", content="prompt", category="testing")
        d = sf.model_dump()
        assert d == {"name": "skill1", "content": "prompt", "category": "testing"}


# ── CoordinationConfig ──────────────────────────────────────────────────────

class TestCoordinationConfig:
    def test_defaults(self):
        cc = CoordinationConfig()
        assert cc.handoff_format == "json"
        assert cc.shared_context == []
        assert cc.escalation_path is None

    def test_full_creation(self):
        cc = CoordinationConfig(
            handoff_format="yaml",
            shared_context=["conversation_id", "user_profile"],
            escalation_path=["human_agent", "supervisor"],
        )
        assert cc.handoff_format == "yaml"
        assert cc.shared_context == ["conversation_id", "user_profile"]
        assert cc.escalation_path == ["human_agent", "supervisor"]

    def test_shared_context_default_is_list(self):
        cc = CoordinationConfig()
        assert isinstance(cc.shared_context, list)
        assert cc.shared_context == []

    def test_escalation_path_none_by_default(self):
        cc = CoordinationConfig()
        assert cc.escalation_path is None

    def test_custom_handoff_format(self):
        cc = CoordinationConfig(handoff_format="protobuf")
        assert cc.handoff_format == "protobuf"

    def test_model_dump(self):
        cc = CoordinationConfig(shared_context=["a"])
        d = cc.model_dump()
        assert d["shared_context"] == ["a"]
        assert d["handoff_format"] == "json"


# ── AgentDefinition ─────────────────────────────────────────────────────────

class TestAgentDefinition:
    def test_creation_minimal(self):
        ad = AgentDefinition(
            name="triage_agent",
            role="Classify requests",
            system_prompt="You are a triage agent.",
        )
        assert ad.name == "triage_agent"
        assert ad.role == "Classify requests"
        assert ad.system_prompt == "You are a triage agent."
        assert ad.tools == []
        assert ad.skills == []
        assert isinstance(ad.coordination_rules, CoordinationConfig)
        assert ad.config_yaml == ""

    def test_full_creation(self):
        tools = [ToolConfig(name="web"), ToolConfig(name="db")]
        skills = [SkillFile(name="skill1", content="c1")]
        cc = CoordinationConfig(handoff_format="json", shared_context=["c1"])
        ad = AgentDefinition(
            name="full_agent",
            role="Full role",
            system_prompt="Full prompt",
            tools=tools,
            skills=skills,
            coordination_rules=cc,
            config_yaml="platform: custom",
        )
        assert len(ad.tools) == 2
        assert len(ad.skills) == 1
        assert ad.coordination_rules.handoff_format == "json"
        assert ad.config_yaml == "platform: custom"

    def test_name_pattern_valid(self):
        # Must match ^[a-z][a-z0-9_]*$
        AgentDefinition(name="a", role="r", system_prompt="p")
        AgentDefinition(name="agent_1", role="r", system_prompt="p")
        AgentDefinition(name="my_agent_2", role="r", system_prompt="p")

    def test_name_pattern_uppercase_raises(self):
        with pytest.raises(ValidationError):
            AgentDefinition(name="Agent", role="r", system_prompt="p")

    def test_name_pattern_leading_number_raises(self):
        with pytest.raises(ValidationError):
            AgentDefinition(name="1agent", role="r", system_prompt="p")

    def test_name_pattern_special_chars_raises(self):
        with pytest.raises(ValidationError):
            AgentDefinition(name="agent-name", role="r", system_prompt="p")

    def test_tools_default_empty_list(self):
        ad = AgentDefinition(name="a", role="r", system_prompt="p")
        assert ad.tools == []

    def test_skills_default_empty_list(self):
        ad = AgentDefinition(name="a", role="r", system_prompt="p")
        assert ad.skills == []

    def test_coordination_rules_default(self):
        ad = AgentDefinition(name="a", role="r", system_prompt="p")
        assert isinstance(ad.coordination_rules, CoordinationConfig)
        assert ad.coordination_rules.handoff_format == "json"

    def test_config_yaml_default_empty_string(self):
        ad = AgentDefinition(name="a", role="r", system_prompt="p")
        assert ad.config_yaml == ""

    def test_model_dump_safe_no_secrets(self):
        ad = AgentDefinition(
            name="agent",
            role="test",
            system_prompt="test",
            config_yaml="provider: openai\nmodel: gpt-4",
        )
        d = ad.model_dump_safe()
        assert d["config_yaml"] == "provider: openai\nmodel: gpt-4"

    def test_model_dump_safe_with_api_key_redacts(self):
        ad = AgentDefinition(
            name="agent",
            role="test",
            system_prompt="test",
            config_yaml="api_key: sk-secret-key-value",
        )
        d = ad.model_dump_safe()
        assert d["config_yaml"] == "[REDACTED]"

    def test_model_dump_safe_with_api_key_case_insensitive(self):
        ad = AgentDefinition(
            name="agent",
            role="test",
            system_prompt="test",
            config_yaml="API_KEY: UPPERCASE-SECRET",
        )
        d = ad.model_dump_safe()
        assert d["config_yaml"] == "[REDACTED]"

    def test_agent_definition_with_nested_tools(self):
        ad = AgentDefinition(
            name="agent",
            role="test",
            system_prompt="test",
            tools=[ToolConfig(name="tool1", endpoint="http://x")],
        )
        d = ad.model_dump()
        assert len(d["tools"]) == 1
        assert d["tools"][0]["name"] == "tool1"


# ── DomainModel ─────────────────────────────────────────────────────────────

class TestDomainModel:
    def test_creation_minimal(self):
        dm = DomainModel(domain="ecommerce")
        assert dm.domain == "ecommerce"
        assert dm.actors == []
        assert dm.intents == []
        assert dm.constraints == []
        assert dm.edge_cases == []
        assert dm.success_criteria == []

    def test_full_creation(self):
        dm = DomainModel(
            domain="customer_support",
            actors=["customer", "agent"],
            intents=[{"actor": "customer", "intent": "help"}],
            constraints=["must_log"],
            edge_cases=["offline"],
            success_criteria=["95%_resolution"],
        )
        assert dm.domain == "customer_support"
        assert dm.actors == ["customer", "agent"]
        assert dm.intents == [{"actor": "customer", "intent": "help"}]
        assert dm.constraints == ["must_log"]
        assert dm.edge_cases == ["offline"]
        assert dm.success_criteria == ["95%_resolution"]

    def test_intent_count_zero(self):
        dm = DomainModel(domain="test")
        assert dm.intent_count == 0

    def test_intent_count(self):
        dm = DomainModel(
            domain="test",
            intents=[{"a": 1}, {"b": 2}, {"c": 3}],
        )
        assert dm.intent_count == 3

    def test_actor_count_zero(self):
        dm = DomainModel(domain="test")
        assert dm.actor_count == 0

    def test_actor_count(self):
        dm = DomainModel(domain="test", actors=["a", "b", "c", "d"])
        assert dm.actor_count == 4

    def test_defaults_are_empty_lists(self):
        dm = DomainModel(domain="test")
        assert dm.actors == []
        assert dm.intents == []
        assert dm.constraints == []
        assert dm.edge_cases == []
        assert dm.success_criteria == []

    def test_model_dump(self):
        dm = DomainModel(domain="test", actors=["a"])
        d = dm.model_dump()
        assert d["domain"] == "test"
        assert d["actors"] == ["a"]


# ── AgentArchitecture ───────────────────────────────────────────────────────

class TestAgentArchitecture:
    def test_creation_minimal(self):
        aa = AgentArchitecture(topology="router")
        assert aa.topology == "router"
        assert aa.agents == []
        assert aa.routing == {}

    def test_full_creation(self):
        aa = AgentArchitecture(
            topology="swarm",
            agents=[
                {"name": "agent1", "role": "worker"},
                {"name": "agent2", "role": "worker"},
            ],
            routing={"strategy": "round_robin", "confidence_threshold": 0.9},
        )
        assert aa.topology == "swarm"
        assert len(aa.agents) == 2
        assert aa.routing["strategy"] == "round_robin"

    def test_agent_count_zero(self):
        aa = AgentArchitecture(topology="router")
        assert aa.agent_count == 0

    def test_agent_count(self):
        aa = AgentArchitecture(
            topology="parallel",
            agents=[{"name": f"a{i}", "role": "r"} for i in range(5)],
        )
        assert aa.agent_count == 5

    def test_routing_default_is_empty_dict(self):
        aa = AgentArchitecture(topology="sequential")
        assert aa.routing == {}

    def test_agents_default_is_empty_list(self):
        aa = AgentArchitecture(topology="router")
        assert aa.agents == []

    def test_topology_required(self):
        with pytest.raises(ValidationError):
            AgentArchitecture()

    def test_model_dump(self):
        aa = AgentArchitecture(
            topology="router",
            agents=[{"name": "a", "role": "r"}],
            routing={"strategy": "intent"},
        )
        d = aa.model_dump()
        assert d["topology"] == "router"
        assert len(d["agents"]) == 1
        assert d["routing"]["strategy"] == "intent"

    def test_topology_values_any_string_accepted(self):
        # topology field has no enum constraint; any string passes description
        aa = AgentArchitecture(topology="custom_topology")
        assert aa.topology == "custom_topology"
