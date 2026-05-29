"""State machine for the Genesis Engine 5-stage pipeline lifecycle.

Manages transitions through ANALYZE → ARCHITECT → BUILD → TEST → DEPLOY
with a TEST→BUILD retry loop (max 3 retries, default threshold 0.80).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from genesis.llm.provider import LLMProvider
from genesis.adapters.base import DeploymentTarget, DeploymentResult
from genesis.models.build import Build, BuildStatus, PipelineStage
from genesis.models.agent import DomainModel, AgentArchitecture, AgentDefinition
from genesis.models.test_results import TestResults
from genesis.pipeline.analyze import AnalyzeStage
from genesis.pipeline.architect import ArchitectStage
from genesis.pipeline.build import BuildStage as BuildPipelineStage
from genesis.pipeline.test import TestStage as TestPipelineStage
from genesis.pipeline.verify import verify_agents
from genesis.pipeline.deploy import DeployStage
from genesis.storage.repository import ProjectRepository, BuildRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEST_THRESHOLD = 0.80
MAX_BUILD_RETRIES = 3

# Ordered pipeline stages
_PIPELINE_SEQUENCE: List[PipelineStage] = [
    PipelineStage.ANALYZE,
    PipelineStage.ARCHITECT,
    PipelineStage.BUILD,
    PipelineStage.TEST,
    PipelineStage.DEPLOY,
]

# Map PipelineStage → BuildStatus (the "executing" status)
_STAGE_TO_STATUS: Dict[PipelineStage, BuildStatus] = {
    PipelineStage.ANALYZE: BuildStatus.ANALYZING,
    PipelineStage.ARCHITECT: BuildStatus.ARCHITECTING,
    PipelineStage.BUILD: BuildStatus.BUILDING,
    PipelineStage.TEST: BuildStatus.TESTING,
    PipelineStage.DEPLOY: BuildStatus.DEPLOYING,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Manages the 5-stage pipeline lifecycle for a Build.

    Coordinates pipeline stages (ANALYZE → ARCHITECT → BUILD → TEST →
    DEPLOY), persists state to storage after each transition, and handles
    the TEST→BUILD retry loop when test quality is below threshold.

    Args:
        llm: An LLMProvider instance used by ANALYZE, ARCHITECT, BUILD, and
            TEST stages.
        project_repo: Repository for updating project records.
        build_repo: Repository for persisting build state.
        target: A DeploymentTarget adapter (AgentSystem, Hermes, etc.).
        test_threshold: Minimum overall_score for tests to pass (default 0.80).
        max_retries: Maximum TEST→BUILD retry loop iterations (default 3).
    """

    def __init__(
        self,
        llm: LLMProvider,
        project_repo: ProjectRepository,
        build_repo: BuildRepository,
        target: DeploymentTarget,
        test_threshold: float = DEFAULT_TEST_THRESHOLD,
        max_retries: int = MAX_BUILD_RETRIES,
    ) -> None:
        self.llm = llm
        self.project_repo = project_repo
        self.build_repo = build_repo
        self.target = target
        self.test_threshold = test_threshold
        self.max_retries = max_retries

        # Stage instances created lazily with injected dependencies
        self._analyze_stage = AnalyzeStage(llm)
        self._architect_stage = ArchitectStage(llm)
        self._build_stage = BuildPipelineStage(llm)
        self._test_stage = TestPipelineStage(llm)
        self._deploy_stage = DeployStage(target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_pipeline(self, build: Build) -> Build:
        """Execute the full 5-stage pipeline for a build.

        Transitions the build through ANALYZE → ARCHITECT → BUILD → TEST →
        DEPLOY, with a TEST→BUILD retry loop when test score is below
        threshold. Updates storage after each stage. Catches exceptions
        and marks the build as FAILED on error.

        Args:
            build: The Build to execute. Must have status=QUEUED or be
                pre-initialised. The instance is mutated in-place and also
                returned.

        Returns:
            The completed (or failed) Build instance.
        """
        retries = build.retries
        artifacts: Dict[str, Any] = build.artifacts or {}

        try:
            # --- Stage 1: ANALYZE ---
            domain_model = await self._run_analyze(build)
            artifacts["domain_model"] = domain_model.model_dump()

            # --- Stage 2: ARCHITECT ---
            architecture = await self._run_architect(build, domain_model)
            artifacts["architecture"] = architecture.model_dump()

            # --- Stages 3+4: BUILD ↔ TEST retry loop ---
            agents, test_results, retries = await self._run_build_test_loop(
                build, architecture, retries
            )
            artifacts["agents"] = [a.model_dump() for a in agents]
            build.test_results = test_results.model_dump()
            build.retries = retries

            # --- Verification: double-check agent quality ---
            verification = await verify_agents(
                self.llm,
                artifacts["agents"],
                {"topology": architecture.topology},
            )
            artifacts["verification"] = verification
            logger.info("Verification complete — score=%.2f", verification.get("score", 0))

            # --- Stage 5: DEPLOY ---
            deployment = await self._run_deploy(build, agents)
            artifacts["deployment"] = {
                "deployment_id": deployment.deployment_id,
                "endpoint_url": deployment.endpoint_url,
                "status": deployment.status,
                "agent_count": deployment.agent_count,
                "metadata": deployment.metadata,
            }

            # --- Success ---
            build.artifacts = artifacts
            build.stage = PipelineStage.DEPLOY
            build.status = BuildStatus.COMPLETED
            build.stage_progress = 1.0
            build.completed_at = datetime.now(timezone.utc)
            await self._persist(build)
            logger.info("Pipeline complete — build=%s", build.id)

        except Exception as exc:
            logger.exception("Pipeline failed — build=%s: %s", build.id, exc)
            build.artifacts = artifacts or None
            build.status = BuildStatus.FAILED
            build.error = str(exc)
            build.retries = retries
            build.completed_at = datetime.now(timezone.utc)
            await self._persist(build)

        return build

    # ------------------------------------------------------------------
    # Stage runners (each transitions status, runs the stage, persists)
    # ------------------------------------------------------------------

    async def _run_analyze(self, build: Build) -> DomainModel:
        """Execute the ANALYZE stage."""
        self._transition(build, PipelineStage.ANALYZE, 0.0)
        await self._persist(build)

        domain_model = await self._analyze_stage.run(build.problem_description)

        self._transition(build, PipelineStage.ANALYZE, 0.2)
        await self._persist(build)

        logger.info("ANALYZE complete — build=%s", build.id)
        return domain_model

    async def _run_architect(
        self, build: Build, domain_model: DomainModel
    ) -> AgentArchitecture:
        """Execute the ARCHITECT stage."""
        self._transition(build, PipelineStage.ARCHITECT, 0.2)
        await self._persist(build)

        architecture = await self._architect_stage.run(
            domain_model, build.problem_description
        )

        self._transition(build, PipelineStage.ARCHITECT, 0.4)
        await self._persist(build)

        logger.info("ARCHITECT complete — build=%s", build.id)
        return architecture

    async def _run_build_test_loop(
        self,
        build: Build,
        architecture: AgentArchitecture,
        retries: int,
    ) -> tuple[List[AgentDefinition], TestResults, int]:
        """Execute the BUILD → TEST cycle with retry loop.

        If TEST score is below threshold and retries remain, feeds failure
        feedback back into BUILD and re-runs TEST. Returns the final agents,
        test results, and retry count.
        """
        feedback: Optional[TestResults] = None

        while True:
            # -- BUILD --
            self._transition(build, PipelineStage.BUILD, 0.4)
            build.retries = retries
            await self._persist(build)

            agents = await self._build_stage.run(architecture, feedback=feedback)

            self._transition(build, PipelineStage.BUILD, 0.6)
            await self._persist(build)

            # -- TEST --
            self._transition(build, PipelineStage.TEST, 0.6)
            await self._persist(build)

            test_results = await self._test_stage.run(agents)

            # Check threshold via the model's own method (sets .passed)
            test_results.check_threshold(self.test_threshold)

            self._transition(build, PipelineStage.TEST, 0.8)
            await self._persist(build)

            logger.info(
                "TEST complete — build=%s score=%.2f passed=%s retries=%d",
                build.id,
                test_results.overall_score,
                test_results.passed,
                retries,
            )

            if test_results.passed:
                return agents, test_results, retries

            # Not passed — can we retry?
            if retries < self.max_retries:
                retries += 1
                feedback = test_results
                logger.warning(
                    "BUILD retry %d/%d — build=%s score=%.2f threshold=%.2f",
                    retries,
                    self.max_retries,
                    build.id,
                    test_results.overall_score,
                    self.test_threshold,
                )
                continue

            # Max retries exhausted, but return what we have.
            # Caller can decide whether to fail the whole pipeline or deploy.
            logger.error(
                "BUILD retries exhausted — build=%s score=%.2f threshold=%.2f",
                build.id,
                test_results.overall_score,
                self.test_threshold,
            )
            return agents, test_results, retries

    async def _run_deploy(
        self,
        build: Build,
        agents: List[AgentDefinition],
    ) -> DeploymentResult:
        """Execute the DEPLOY stage."""
        self._transition(build, PipelineStage.DEPLOY, 0.8)
        await self._persist(build)

        deployment = await self._deploy_stage.run(agents, build.target_config)

        self._transition(build, PipelineStage.DEPLOY, 1.0)
        await self._persist(build)

        logger.info("DEPLOY complete — build=%s", build.id)
        return deployment

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transition(
        build: Build,
        stage: PipelineStage,
        progress: float,
    ) -> None:
        """Update the build's stage, status, and progress in-place."""
        build.stage = stage
        build.status = _STAGE_TO_STATUS[stage]
        build.stage_progress = progress

    async def _persist(self, build: Build) -> None:
        """Persist the build's current state to storage."""
        try:
            await self.build_repo.update(build)
        except Exception as exc:
            logger.error(
                "Failed to persist build %s at stage %s: %s",
                build.id,
                build.stage,
                exc,
            )