"""Shared deploy finalization.

Both the orchestrator's happy path and the human-in-the-loop *approve* endpoint
must drive a build through DEPLOY → COMPLETED identically — otherwise a build
that was paused for approval would finish with different artifacts/status than
one that sailed straight through. This module is the single source of truth for
that finalization so the two paths can never drift.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from genesis.adapters.base import DeploymentResult, DeploymentTarget
from genesis.models.agent import AgentDefinition
from genesis.models.build import Build, BuildStatus, PipelineStage
from genesis.pipeline.deploy import DeployStage

logger = logging.getLogger("genesis.orchestrator.deploy")


def deployment_artifact(deployment: DeploymentResult) -> Dict[str, Any]:
    """Canonical ``artifacts['deployment']`` shape — one definition, two callers."""
    return {
        "deployment_id": deployment.deployment_id,
        "endpoint_url": deployment.endpoint_url,
        "status": deployment.status,
        "agent_count": deployment.agent_count,
        "metadata": deployment.metadata,
    }


async def finalize_deploy(
    build: Build,
    agents: List[AgentDefinition],
    target: DeploymentTarget,
    build_repo,
    project_repo,
    *,
    decrypt: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Build:
    """Deploy ``agents`` for ``build`` and drive it to COMPLETED (or FAILED).

    Mirrors the orchestrator's deploy + success block exactly so an
    approved-out-of-band build finishes identically to one deployed inline.

    Args:
        build: The build to finalize. Mutated in place and persisted.
        agents: Reconstructed agent definitions to deploy.
        target: The deployment target adapter.
        build_repo: Build repository for persistence.
        project_repo: Project repository for status + active-pointer updates.
        decrypt: Optional secret-decryption callable applied to
            ``build.target_config`` before deploy (the DB stores it encrypted).

    Returns:
        The finalized build (status COMPLETED on success).

    Raises:
        Exception: Re-raises any deploy failure after marking the build FAILED.
    """
    cfg = build.target_config or {}
    if decrypt is not None:
        cfg = decrypt(cfg)
    artifacts: Dict[str, Any] = dict(build.artifacts or {})

    try:
        build.stage = PipelineStage.DEPLOY
        build.status = BuildStatus.DEPLOYING
        build.stage_progress = 0.8
        await build_repo.update(build)

        deployment = await DeployStage(target).run(agents, cfg)
        artifacts["deployment"] = deployment_artifact(deployment)

        build.artifacts = artifacts
        build.stage = PipelineStage.DEPLOY
        build.status = BuildStatus.COMPLETED
        build.stage_progress = 1.0
        build.completed_at = datetime.now(timezone.utc)
        await build_repo.update(build)

        project = await project_repo.get(build.project_id)
        if project:
            project.status = "deployed"
            project.active_build_id = build.id
            await project_repo.update(project)

        logger.info("Deploy finalized — build=%s", build.id)
        return build

    except Exception as exc:  # noqa: BLE001 - surfaced after marking FAILED
        logger.exception("Deploy finalization failed — build=%s: %s", build.id, exc)
        build.artifacts = artifacts or build.artifacts
        build.status = BuildStatus.FAILED
        build.error = f"Deployment failed: {exc}"
        build.completed_at = datetime.now(timezone.utc)
        await build_repo.update(build)
        project = await project_repo.get(build.project_id)
        if project:
            project.status = "failed"
            await project_repo.update(project)
        raise


def reconstruct_agents(
    agent_dicts: List[Dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> List[AgentDefinition]:
    """Rebuild AgentDefinition objects from persisted artifact dicts.

    Args:
        agent_dicts: ``artifacts['agents']`` — a list of ``model_dump()`` dicts.
        allow_partial: When True, silently skip dicts that fail validation;
            when False (default), raise ``ValueError`` on the first bad dict.

    Returns:
        The reconstructed agent definitions.

    Raises:
        ValueError: If a dict fails validation and ``allow_partial`` is False.
    """
    agents: List[AgentDefinition] = []
    for idx, raw in enumerate(agent_dicts or []):
        try:
            agents.append(AgentDefinition(**raw))
        except Exception as exc:  # noqa: BLE001
            if allow_partial:
                logger.warning("Skipping invalid agent at index %d: %s", idx, exc)
                continue
            raise ValueError(f"Invalid agent at index {idx}: {exc}") from exc
    return agents
