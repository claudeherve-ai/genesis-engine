"""Tests for the Genesis Engine Orchestrator state machine.

Covers the full 5-stage pipeline lifecycle, TEST→BUILD retry loop,
error handling, state transitions, and storage persistence.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import List

from genesis.orchestrator.state_machine import (
    Orchestrator,
    DEFAULT_TEST_THRESHOLD,
    MAX_BUILD_RETRIES,
    _STAGE_TO_STATUS,
)
from genesis.models.build import Build, BuildStatus, PipelineStage
from genesis.models.agent import (
    DomainModel,
    AgentArchitecture,
    AgentDefinition,
    ToolConfig,
    SkillFile,
    CoordinationConfig,
)
from genesis.models.test_results import TestResults, TestFailure
from genesis.adapters.base import DeploymentResult


# ── Shared helpers ──────────────────────────────────────────────────────────


def make_build(**overrides) -> Build:
    """Create a Build with sensible defaults for orchestrator tests."""
    defaults = {
        "project_id": "test-project",
        "problem_description": "Build a customer support system for a B2B SaaS company.",
        "target": "agentsystem",
        "target_config": {"endpoint": "https://test.example.com"},
    }
    defaults.update(overrides)
    return Build(**defaults)


def make_domain_model() -> DomainModel:
    return DomainModel(
        domain="customer_support",
        actors=["customer", "support_agent", "billing_system"],
        intents=[
            {"actor": "customer", "intent": "report_technical_issue", "priority": "high"},
            {"actor": "customer", "intent": "billing_question", "priority": "medium"},
        ],
        constraints=["verify_identity"],
        edge_cases=["off_hours_contact"],
        success_criteria=["requests_triaged_within_5_seconds"],
    )


def make_architecture() -> AgentArchitecture:
    return AgentArchitecture(
        topology="router",
        agents=[
            {
                "name": "triage_agent",
                "role": "Classify and route incoming requests",
                "triggers": ["new_request"],
                "tools": ["intent_classifier"],
            },
            {
                "name": "technical_agent",
                "role": "Resolve technical issues",
                "tools": ["knowledge_base_search"],
                "escalates_to": "human_agent",
            },
        ],
        routing={"strategy": "intent_based", "confidence_threshold": 0.85},
    )


def make_agents() -> List[AgentDefinition]:
    return [
        AgentDefinition(
            name="triage_agent",
            role="Classify and route incoming requests",
            system_prompt="You are a triage agent.",
            tools=[ToolConfig(name="intent_classifier", description="Classify intent")],
            skills=[SkillFile(name="triage", content="Classify then route.")],
            coordination_rules=CoordinationConfig(
                handoff_format="json", escalation_path=["technical_agent"],
            ),
            config_yaml="name: triage_agent\n",
        ),
        AgentDefinition(
            name="technical_agent",
            role="Resolve technical issues",
            system_prompt="You are a technical support agent.",
            tools=[ToolConfig(name="kb_search", description="Search KB")],
            skills=[SkillFile(name="tech_proc", content="Diagnose and resolve.")],
            coordination_rules=CoordinationConfig(
                handoff_format="json", escalation_path=["human_agent"],
            ),
            config_yaml="name: technical_agent\n",
        ),
    ]


def make_test_results(
    overall_score: float = 0.95,
    passed: bool = True,
    failures: List[TestFailure] | None = None,
) -> TestResults:
    return TestResults(
        scenarios_run=12,
        scenarios_passed=11,
        overall_score=overall_score,
        metrics={"intent_classification": 0.95, "resolution_rate": 0.92},
        failures=failures or [],
        passed=passed,
    )


def make_deployment_result() -> DeploymentResult:
    return DeploymentResult(
        deployment_id="dep-001",
        endpoint_url="https://deploy.example.com/agents/support-system",
        status="deployed",
        agent_count=2,
    )


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """An async-capable mock LLMProvider."""
    llm = AsyncMock()
    return llm


@pytest.fixture
def mock_target():
    """A mock DeploymentTarget."""
    target = AsyncMock()
    target.provision.return_value = make_deployment_result()
    target.health_check.return_value = True
    return target


@pytest.fixture
def mock_repos(db_session):
    """Real ProjectRepository + BuildRepository backed by in-memory SQLite."""
    from genesis.storage.repository import ProjectRepository, BuildRepository
    return ProjectRepository(db_session), BuildRepository(db_session)


@pytest.fixture
def orchestrator(mock_llm, mock_repos, mock_target):
    """Fully-wired Orchestrator with real repos and mock LLM/target.

    All 5 stage .run() methods are mocked with AsyncMock so tests can
    selectively override individual stages without triggering the real LLM.
    """
    proj_repo, build_repo = mock_repos
    orch = Orchestrator(
        llm=mock_llm,
        project_repo=proj_repo,
        build_repo=build_repo,
        target=mock_target,
    )

    # Mock all stage .run() methods with sensible defaults
    orch._analyze_stage.run = AsyncMock(return_value=make_domain_model())
    orch._architect_stage.run = AsyncMock(return_value=make_architecture())
    orch._build_stage.run = AsyncMock(return_value=make_agents())
    orch._test_stage.run = AsyncMock(return_value=make_test_results())
    orch._deploy_stage.run = AsyncMock(return_value=make_deployment_result())

    return orch


@pytest.fixture
async def persisted_build(orchestrator):
    """A Build persisted via the repo."""
    b = make_build()
    await orchestrator.build_repo.create(b)
    return b


# ── State-transition helpers ────────────────────────────────────────────────


class TestStageStatusMapping:
    """Verify _STAGE_TO_STATUS maps every PipelineStage to a BuildStatus."""

    def test_all_stages_have_status(self):
        for stage in PipelineStage:
            assert stage in _STAGE_TO_STATUS
            assert isinstance(_STAGE_TO_STATUS[stage], BuildStatus)

    def test_analyze_maps_to_analyzing(self):
        assert _STAGE_TO_STATUS[PipelineStage.ANALYZE] == BuildStatus.ANALYZING

    def test_architect_maps_to_architecting(self):
        assert _STAGE_TO_STATUS[PipelineStage.ARCHITECT] == BuildStatus.ARCHITECTING

    def test_build_maps_to_building(self):
        assert _STAGE_TO_STATUS[PipelineStage.BUILD] == BuildStatus.BUILDING

    def test_test_maps_to_testing(self):
        assert _STAGE_TO_STATUS[PipelineStage.TEST] == BuildStatus.TESTING

    def test_deploy_maps_to_deploying(self):
        assert _STAGE_TO_STATUS[PipelineStage.DEPLOY] == BuildStatus.DEPLOYING


# ── Happy path: full pipeline ──────────────────────────────────────────────


class TestHappyPath:
    """End-to-end happy path through all 5 stages."""

    async def test_full_pipeline_completes(self, orchestrator, persisted_build):
        result = await orchestrator.run_pipeline(persisted_build)

        assert result is persisted_build
        assert result.status == BuildStatus.COMPLETED
        assert result.stage == PipelineStage.DEPLOY
        assert result.stage_progress == 1.0
        assert result.completed_at is not None
        assert result.error is None
        assert result.is_terminal is True

    async def test_artifacts_contain_all_stage_outputs(self, orchestrator, persisted_build):
        result = await orchestrator.run_pipeline(persisted_build)

        artifacts = result.artifacts
        assert isinstance(artifacts, dict)
        assert "domain_model" in artifacts
        assert "architecture" in artifacts
        assert "agents" in artifacts
        assert "deployment" in artifacts
        assert artifacts["deployment"]["status"] == "deployed"

    async def test_test_results_stored(self, orchestrator, persisted_build):
        result = await orchestrator.run_pipeline(persisted_build)

        assert result.test_results is not None
        assert "overall_score" in result.test_results

    async def test_retries_count_recorded(self, orchestrator, persisted_build):
        result = await orchestrator.run_pipeline(persisted_build)
        assert result.retries == 0  # happy path, no retries needed

    async def test_build_persisted_after_each_stage(self, orchestrator, persisted_build):
        await orchestrator.run_pipeline(persisted_build)

        fetched = await orchestrator.build_repo.get(persisted_build.id)
        assert fetched is not None
        assert fetched.status == BuildStatus.COMPLETED
        assert fetched.artifacts is not None
        assert fetched.test_results is not None

    async def test_default_target_config_passed_to_deploy(self, orchestrator, persisted_build):
        await orchestrator.run_pipeline(persisted_build)

        orchestrator._deploy_stage.run.assert_called_once()
        args, _ = orchestrator._deploy_stage.run.call_args
        assert len(args[0]) == 2  # 2 agents


# ── Stage-by-stage transition verification ──────────────────────────────────


class TestStageTransitions:
    """Each stage should set the correct status and progress on the build."""

    async def test_analyze_sets_status(self, orchestrator, persisted_build):
        await orchestrator.run_pipeline(persisted_build)
        assert persisted_build.status == BuildStatus.COMPLETED
        assert persisted_build.stage == PipelineStage.DEPLOY

    async def test_architect_sets_status(self, orchestrator, persisted_build):
        await orchestrator.run_pipeline(persisted_build)
        assert persisted_build.status == BuildStatus.COMPLETED
        assert "architecture" in persisted_build.artifacts

    async def test_build_sets_status(self, orchestrator, persisted_build):
        await orchestrator.run_pipeline(persisted_build)
        assert "agents" in persisted_build.artifacts
        assert len(persisted_build.artifacts["agents"]) == 2

    async def test_deploy_sets_status(self, orchestrator, persisted_build):
        await orchestrator.run_pipeline(persisted_build)
        assert "deployment" in persisted_build.artifacts
        assert persisted_build.artifacts["deployment"]["deployment_id"] == "dep-001"


# ── TEST → BUILD retry loop ────────────────────────────────────────────────


class TestRetryLoop:
    """Tests the TEST→BUILD retry loop."""

    async def test_single_retry_on_low_score(self, orchestrator, persisted_build):
        """When first TEST fails below threshold, BUILD is re-run with
        feedback, and pipeline succeeds on the second attempt."""
        test_results_low = make_test_results(overall_score=0.60, passed=False)
        test_results_high = make_test_results(overall_score=0.90, passed=True)

        orch = orchestrator
        call_count = [0]

        async def mock_test(agents):
            call_count[0] += 1
            if call_count[0] == 1:
                return test_results_low
            return test_results_high

        orch._test_stage.run = mock_test

        result = await orch.run_pipeline(persisted_build)

        assert result.status == BuildStatus.COMPLETED
        assert call_count[0] == 2
        assert result.retries == 1

    async def test_build_receives_feedback_on_retry(self, orchestrator, persisted_build):
        """On retry, the BUILD stage should receive test results as feedback."""
        test_results_low = make_test_results(
            overall_score=0.60,
            passed=False,
            failures=[
                TestFailure(
                    scenario="billing dispute",
                    agent="billing_agent",
                    expected="refund processed",
                    actual="no response",
                    metric="resolution_rate",
                )
            ],
        )
        test_results_high = make_test_results(overall_score=0.90, passed=True)

        orch = orchestrator
        feedbacks_received = []

        orig_build = orch._build_stage.run

        async def mock_build(architecture, feedback=None):
            feedbacks_received.append(feedback)
            return await orig_build(architecture, feedback=feedback)

        orch._build_stage.run = mock_build

        call_count = [0]

        async def mock_test(agents):
            call_count[0] += 1
            if call_count[0] == 1:
                return test_results_low
            return test_results_high

        orch._test_stage.run = mock_test

        await orch.run_pipeline(persisted_build)

        # First BUILD call: feedback is None (initial run)
        assert feedbacks_received[0] is None
        # Second BUILD call: feedback is the failing TestResults
        assert feedbacks_received[1] is not None
        assert feedbacks_received[1].overall_score == 0.60
        assert len(feedbacks_received[1].failures) == 1

    async def test_max_retries_exhausted(self, orchestrator, persisted_build):
        """When max retries are exhausted, pipeline still proceeds to DEPLOY
        with the last test results (graceful degradation)."""
        test_results_fail = make_test_results(overall_score=0.60, passed=False)

        orch = orchestrator
        orch.max_retries = 3

        async def mock_test(agents):
            return test_results_fail

        orch._test_stage.run = mock_test

        result = await orch.run_pipeline(persisted_build)

        # Pipeline completes (proceeds to deploy even after max retries)
        assert result.status == BuildStatus.COMPLETED
        assert result.test_results is not None
        assert result.retries == 3

    async def test_retry_respects_custom_threshold(self, orchestrator, persisted_build):
        """Custom threshold (0.90) should trigger retry when score is 0.85."""
        test_results_mid = make_test_results(overall_score=0.85, passed=False)

        orch = orchestrator
        orch.test_threshold = 0.90

        call_count = [0]

        async def mock_test(agents):
            call_count[0] += 1
            if call_count[0] == 1:
                return test_results_mid  # 0.85 < 0.90
            return make_test_results(overall_score=0.95, passed=True)

        orch._test_stage.run = mock_test

        result = await orch.run_pipeline(persisted_build)
        assert result.status == BuildStatus.COMPLETED
        assert call_count[0] == 2
        assert result.retries == 1

    async def test_no_retry_when_passing(self, orchestrator, persisted_build):
        """When test passes on first attempt, BUILD is not re-run."""
        test_results_pass = make_test_results(overall_score=0.95, passed=True)

        orch = orchestrator
        build_call_count = [0]

        orig_build = orch._build_stage.run

        async def mock_build(architecture, feedback=None):
            build_call_count[0] += 1
            return await orig_build(architecture, feedback=feedback)

        orch._build_stage.run = mock_build

        async def mock_test(agents):
            return test_results_pass

        orch._test_stage.run = mock_test

        result = await orch.run_pipeline(persisted_build)

        assert build_call_count[0] == 1  # BUILD called exactly once
        assert result.retries == 0


# ── Error handling ─────────────────────────────────────────────────────────


class TestErrorHandling:
    """The orchestrator catches exceptions and marks the build FAILED."""

    async def test_analyze_failure_sets_failed(self, orchestrator, persisted_build):
        orchestrator._analyze_stage.run = AsyncMock(
            side_effect=ValueError("ANALYZE failed: invalid JSON")
        )

        result = await orchestrator.run_pipeline(persisted_build)

        assert result.status == BuildStatus.FAILED
        assert "ANALYZE failed" in result.error
        assert result.is_terminal is True
        assert result.completed_at is not None

    async def test_architect_failure_sets_failed(self, orchestrator, persisted_build):
        orchestrator._architect_stage.run = AsyncMock(
            side_effect=ValueError("ARCHITECT failed")
        )

        result = await orchestrator.run_pipeline(persisted_build)

        assert result.status == BuildStatus.FAILED
        assert "ARCHITECT failed" in result.error

    async def test_build_failure_sets_failed(self, orchestrator, persisted_build):
        orchestrator._build_stage.run = AsyncMock(
            side_effect=ValueError("BUILD failed: invalid agent spec")
        )

        result = await orchestrator.run_pipeline(persisted_build)

        assert result.status == BuildStatus.FAILED
        assert "BUILD failed" in result.error

    async def test_test_failure_sets_failed(self, orchestrator, persisted_build):
        """Unexpected TEST exception (not a low score) should fail the build."""
        orchestrator._test_stage.run = AsyncMock(
            side_effect=ValueError("TEST failed: invalid JSON from LLM")
        )

        result = await orchestrator.run_pipeline(persisted_build)

        assert result.status == BuildStatus.FAILED
        assert "TEST failed" in result.error

    async def test_deploy_failure_sets_failed(self, orchestrator, persisted_build):
        orchestrator._deploy_stage.run = AsyncMock(
            side_effect=RuntimeError("Deployment target unreachable")
        )

        result = await orchestrator.run_pipeline(persisted_build)

        assert result.status == BuildStatus.FAILED
        assert "Deployment target unreachable" in result.error

    async def test_partial_artifacts_preserved_on_failure(self, orchestrator, persisted_build):
        """Artifacts from stages that completed before failure are preserved."""
        orchestrator._architect_stage.run = AsyncMock(
            side_effect=ValueError("ARCHITECT failed")
        )

        result = await orchestrator.run_pipeline(persisted_build)

        assert result.status == BuildStatus.FAILED
        assert result.artifacts is not None
        assert "domain_model" in result.artifacts
        assert "architecture" not in result.artifacts

    async def test_failed_build_persisted(self, orchestrator, persisted_build):
        orchestrator._analyze_stage.run = AsyncMock(
            side_effect=ValueError("Boom")
        )

        await orchestrator.run_pipeline(persisted_build)

        fetched = await orchestrator.build_repo.get(persisted_build.id)
        assert fetched is not None
        assert fetched.status == BuildStatus.FAILED
        assert fetched.error == "Boom"
        assert fetched.completed_at is not None


# ── Orchestrator construction and config ───────────────────────────────────


class TestOrchestratorConstruction:
    """Tests for Orchestrator instantiation and defaults."""

    def test_default_threshold(self, mock_llm, mock_repos, mock_target):
        proj_repo, build_repo = mock_repos
        orch = Orchestrator(mock_llm, proj_repo, build_repo, mock_target)
        assert orch.test_threshold == DEFAULT_TEST_THRESHOLD
        assert orch.test_threshold == 0.80

    def test_custom_threshold(self, mock_llm, mock_repos, mock_target):
        proj_repo, build_repo = mock_repos
        orch = Orchestrator(
            mock_llm, proj_repo, build_repo, mock_target, test_threshold=0.95
        )
        assert orch.test_threshold == 0.95

    def test_default_max_retries(self, mock_llm, mock_repos, mock_target):
        proj_repo, build_repo = mock_repos
        orch = Orchestrator(mock_llm, proj_repo, build_repo, mock_target)
        assert orch.max_retries == MAX_BUILD_RETRIES
        assert orch.max_retries == 3

    def test_custom_max_retries(self, mock_llm, mock_repos, mock_target):
        proj_repo, build_repo = mock_repos
        orch = Orchestrator(
            mock_llm, proj_repo, build_repo, mock_target, max_retries=5
        )
        assert orch.max_retries == 5

    def test_stages_wired_with_dependencies(self, mock_llm, mock_repos, mock_target):
        proj_repo, build_repo = mock_repos
        orch = Orchestrator(mock_llm, proj_repo, build_repo, mock_target)

        assert orch._analyze_stage.llm is mock_llm
        assert orch._architect_stage.llm is mock_llm
        assert orch._build_stage.llm is mock_llm
        assert orch._test_stage.llm is mock_llm
        assert orch._deploy_stage.target is mock_target


# ── Retry counter ──────────────────────────────────────────────────────────


class TestRetryCounter:
    """build.retries should be correctly incremented during retry loops."""

    async def test_initial_retries_zero(self, orchestrator, persisted_build):
        assert persisted_build.retries == 0

    async def test_retries_incremented_on_loop(self, orchestrator, persisted_build):
        orch = orchestrator
        orch.max_retries = 3

        call_count = [0]

        async def mock_test(agents):
            call_count[0] += 1
            if call_count[0] < 3:
                return make_test_results(overall_score=0.50, passed=False)
            return make_test_results(overall_score=0.95, passed=True)

        orch._test_stage.run = mock_test

        result = await orch.run_pipeline(persisted_build)
        assert result.retries == 2
        assert call_count[0] == 3  # 2 failures + 1 pass


# ── Transition method ───────────────────────────────────────────────────────


class TestTransition:
    """Unit tests for Orchestrator._transition()."""

    def test_transition_updates_stage_status_progress(self, orchestrator):
        b = make_build()
        orchestrator._transition(b, PipelineStage.BUILD, 0.45)

        assert b.stage == PipelineStage.BUILD
        assert b.status == BuildStatus.BUILDING
        assert b.stage_progress == 0.45

    def test_transition_all_stages(self, orchestrator):
        b = make_build()

        transitions = [
            (PipelineStage.ANALYZE, 0.0, BuildStatus.ANALYZING),
            (PipelineStage.ARCHITECT, 0.2, BuildStatus.ARCHITECTING),
            (PipelineStage.BUILD, 0.4, BuildStatus.BUILDING),
            (PipelineStage.TEST, 0.6, BuildStatus.TESTING),
            (PipelineStage.DEPLOY, 0.8, BuildStatus.DEPLOYING),
        ]

        for stage, progress, expected_status in transitions:
            orchestrator._transition(b, stage, progress)
            assert b.stage == stage
            assert b.stage_progress == progress
            assert b.status == expected_status


# ── End-to-end persistence ──────────────────────────────────────────────────


class TestEndToEndPersistence:
    """Full integration: pipeline run with real DB storage."""

    async def test_build_persisted_in_db_after_run(self, orchestrator, persisted_build):
        await orchestrator.run_pipeline(persisted_build)

        fetched = await orchestrator.build_repo.get(persisted_build.id)
        assert fetched is not None
        assert fetched.status == BuildStatus.COMPLETED
        assert fetched.artifacts is not None
        assert "domain_model" in fetched.artifacts

    async def test_pipeline_does_not_modify_different_build(
        self, orchestrator, persisted_build
    ):
        other = make_build()
        await orchestrator.build_repo.create(other)

        await orchestrator.run_pipeline(persisted_build)

        fetched_other = await orchestrator.build_repo.get(other.id)
        assert fetched_other.status == BuildStatus.QUEUED
        assert fetched_other.stage is None


# ── Deploy stage ───────────────────────────────────────────────────────────


class TestDeployStage:
    """Tests specific to the DEPLOY stage within the orchestrator."""

    async def test_deploy_called_with_agents_and_config(
        self, orchestrator, persisted_build
    ):
        await orchestrator.run_pipeline(persisted_build)

        deploy_mock = orchestrator._deploy_stage.run
        deploy_mock.assert_called_once()
        args, kwargs = deploy_mock.call_args
        # First positional arg: agents list
        assert len(args[0]) == 2
        # target_config passed as second positional
        assert args[1] == {"endpoint": "https://test.example.com"}
