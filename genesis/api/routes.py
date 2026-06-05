"""FastAPI route definitions for Genesis Engine.

Projects, builds, one-shot generation, the verified-tool catalog, live
log streaming, deployment packaging, and feedback endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from genesis.models.project import Project, ProjectCreate, ProjectUpdate
from genesis.models.build import Build, BuildRequest, GenerateRequest, BuildStatus
from pydantic import BaseModel, Field
from genesis.storage.repository import ProjectRepository, BuildRepository
from genesis.orchestrator.state_machine import Orchestrator
from genesis.api.dependencies import (
    get_project_repo,
    get_build_repo,
    get_llm_provider,
    get_deployment_target,
    get_knowledge_store,
)
from genesis.security.auth import require_user, require_admin, Principal
from genesis.security.secrets import encrypt_config_secrets, decrypt_config_secrets
from genesis.observability.cost import get_cost_tracker
from genesis.feedback.collector import FeedbackCollector, AgentMetrics
from genesis.tools.catalog import (
    get_tool,
    list_tools,
    search_catalog,
    TOOL_CATALOG,
)

logger = logging.getLogger("genesis.api")
router = APIRouter(prefix="/v1")


# ────────────────────────────────────────────────────
# Projects
# ────────────────────────────────────────────────────


@router.post("/projects", status_code=201)
async def create_project(
    body: ProjectCreate,
    repo: ProjectRepository = Depends(get_project_repo),
):
    """Create a new Genesis project."""
    project = Project(name=body.name, description=body.description)
    await repo.create(project)
    return project.model_dump()


@router.get("/projects")
async def list_projects(
    repo: ProjectRepository = Depends(get_project_repo),
):
    """List all projects."""
    projects = await repo.list_all()
    return {"items": [p.model_dump() for p in projects], "total": len(projects)}


@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    repo: ProjectRepository = Depends(get_project_repo),
):
    """Get project details."""
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump()


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    repo: ProjectRepository = Depends(get_project_repo),
):
    """Update a project."""
    project = await repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if body.name is not None:
        project.name = body.name
    if body.description is not None:
        project.description = body.description
    project.updated_at = datetime.now(timezone.utc)

    await repo.update(project)
    return project.model_dump()


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    repo: ProjectRepository = Depends(get_project_repo),
    _: Principal = Depends(require_admin),
):
    """Delete a project. Requires admin when auth is enabled."""
    deleted = await repo.delete(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}


# ────────────────────────────────────────────────────
# Builds
# ────────────────────────────────────────────────────


@router.post("/projects/{project_id}/build", status_code=202)
async def trigger_build(
    project_id: str,
    body: BuildRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
    build_repo: BuildRepository = Depends(get_build_repo),
    _: Principal = Depends(require_user),
):
    """Trigger a build pipeline for a project."""
    project = await project_repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    build = Build(
        project_id=project_id,
        problem_description=body.problem_description,
        target=body.target,
        target_config=encrypt_config_secrets(body.target_config),
    )
    await build_repo.create(build)

    # Update project
    project.build_count += 1
    project.last_build_id = build.id
    project.status = "building"
    await project_repo.update(project)

    # Fire-and-forget the pipeline (client polls for status)
    asyncio.create_task(_run_pipeline(build))

    return build.model_dump()


@router.get("/projects/{project_id}/builds")
async def list_builds(
    project_id: str,
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """List builds for a project."""
    builds = await build_repo.list_by_project(project_id)
    return {"items": [b.model_dump() for b in builds], "total": len(builds)}


@router.get("/builds/{build_id}")
async def get_build(
    build_id: str,
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Get build status and details."""
    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    return build.model_dump()


@router.get("/builds/{build_id}/logs")
async def stream_build_logs(
    build_id: str,
    request: Request,
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Stream build progress as Server-Sent Events."""
    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    async def event_generator():
        last_status = None
        last_stage = None

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            build = await build_repo.get(build_id)
            if not build:
                break

            # Emit state changes
            if build.status != last_status or build.stage != last_stage:
                data = {
                    "build_id": build.id,
                    "status": build.status.value,
                    "stage": build.stage.value if build.stage else None,
                    "stage_progress": build.stage_progress,
                    "error": build.error,
                }
                yield {
                    "event": "status",
                    "data": json.dumps(data),
                }
                last_status = build.status
                last_stage = build.stage

            # Terminal state — send final event and exit
            if build.is_terminal:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "build_id": build.id,
                        "status": build.status.value,
                        "test_results": build.test_results,
                        "error": build.error,
                    }),
                }
                break

            # Paused for human approval — emit a distinct event and exit so
            # polling clients don't hang forever waiting for a terminal state.
            if build.is_paused:
                yield {
                    "event": "awaiting_approval",
                    "data": json.dumps({
                        "build_id": build.id,
                        "status": build.status.value,
                        "approve_url": f"/v1/builds/{build.id}/approve",
                        "reject_url": f"/v1/builds/{build.id}/reject",
                    }),
                }
                break

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@router.get("/builds/{build_id}/artifacts")
async def get_build_artifacts(
    build_id: str,
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Download generated agent artifacts."""
    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    if not build.artifacts:
        raise HTTPException(status_code=404, detail="No artifacts available")
    return build.artifacts


@router.get("/builds/{build_id}/export")
async def export_agents(
    build_id: str,
    format: str = "json",
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Export generated agents as downloadable files.

    Formats: json (single file with all agents), individual (one file per agent)
    """
    build = await build_repo.get(build_id)
    if not build or not build.artifacts:
        raise HTTPException(status_code=404, detail="No artifacts available")

    agents = build.artifacts.get("agents", [])
    if not agents:
        raise HTTPException(status_code=404, detail="No agents in this build")

    if format == "individual":
        files = {}
        for agent in agents:
            name = agent.get("name", "agent")
            files[f"{name}.json"] = json.dumps(agent, indent=2)
            files[f"{name}.md"] = (
                f"# {agent.get('name', 'Agent')}\n\n"
                f"**Role:** {agent.get('role', '')}\n\n"
                f"## System Prompt\n\n{agent.get('system_prompt', '')}\n\n"
                f"## Tools\n\n"
                + "\n".join(f"- **{t.get('name', 'tool')}**: {t.get('description', '')}"
                          for t in agent.get("tools", []))
            )
        return {"agents": [a.get("name") for a in agents], "files": files}

    return {
        "build_id": build_id,
        "agent_count": len(agents),
        "agents": agents,
    }


# ────────────────────────────────────────────────────
# Tool Catalog
# ────────────────────────────────────────────────────


def _serialize_tool(tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "category": tool.category,
        "endpoint": tool.endpoint,
        "auth_required": tool.auth_required,
        "auth_method": tool.auth_method,
        "env_var": tool.env_var,
        "rate_limit": tool.rate_limit,
        "is_available": tool.is_available,
        "requires_configuration": tool.requires_configuration,
        "docs_url": tool.docs_url,
        "schema": tool.json_schema,
    }


@router.get("/tools")
async def list_catalog_tools(
    category: Optional[str] = Query(None, description="Filter by category"),
    q: Optional[str] = Query(None, description="Free-text search"),
):
    """List the verified tool catalog. Generated agents may ONLY use these tools."""
    if q:
        tools = search_catalog(q)
    else:
        tools = list_tools(category or "")
    categories = sorted({t.category for t in TOOL_CATALOG.values()})
    return {
        "items": [_serialize_tool(t) for t in tools],
        "total": len(tools),
        "categories": categories,
    }


@router.get("/tools/{tool_name}")
async def get_catalog_tool(tool_name: str):
    """Get a single verified tool by name."""
    tool = get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found in catalog")
    return _serialize_tool(tool)


# ────────────────────────────────────────────────────
# Observability / Metrics
# ────────────────────────────────────────────────────


@router.get("/metrics")
async def get_metrics():
    """Token/cost telemetry: totals, per-model, per-stage, per-build."""
    return get_cost_tracker().snapshot()


@router.get("/builds/{build_id}/metrics")
async def get_build_metrics(build_id: str):
    """Cost/token telemetry scoped to a single build."""
    return get_cost_tracker().snapshot(build_id=build_id)


# ────────────────────────────────────────────────────
# One-shot Generate
# ────────────────────────────────────────────────────


@router.post("/generate", status_code=202)
async def generate(
    body: GenerateRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
    build_repo: BuildRepository = Depends(get_build_repo),
    _: Principal = Depends(require_user),
):
    """One-shot: create project + build + deploy in one call.

    The "holy shit" endpoint — describe a problem, get a deployed
    multi-agent system. Returns immediately with a build ID;
    poll /v1/builds/{id} or stream /v1/builds/{id}/logs for progress.
    """
    # Create project
    project = Project(
        name=body.project_name or f"genesis-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        description=body.problem[:200],
    )
    await project_repo.create(project)

    # Create build
    build = Build(
        project_id=project.id,
        problem_description=body.problem,
        target=body.target,
        target_config=encrypt_config_secrets(body.target_config),
    )
    await build_repo.create(build)

    # Update project
    project.build_count = 1
    project.last_build_id = build.id
    project.status = "building"
    await project_repo.update(project)

    # Fire-and-forget pipeline
    asyncio.create_task(_run_pipeline(build))

    return {
        "build_id": build.id,
        "project_id": project.id,
        "status": "queued",
        "status_url": f"/v1/builds/{build.id}",
        "logs_url": f"/v1/builds/{build.id}/logs",
    }


# ────────────────────────────────────────────────────
# Internal: Pipeline runner
# ────────────────────────────────────────────────────


async def _run_pipeline(build: Build) -> None:
    """Run the pipeline in the background, updating storage as we go."""
    session = None
    # Decrypt any secrets that were encrypted at rest before the pipeline
    # (deploy stage) needs them in plaintext.
    build.target_config = decrypt_config_secrets(build.target_config)
    try:
        # Fresh session for the background task
        from genesis.storage.database import get_session as _get_session
        session = _get_session()
        build_repo = BuildRepository(session)
        project_repo = ProjectRepository(session)

        llm = get_llm_provider()
        target = get_deployment_target()
        orchestrator = Orchestrator(llm, project_repo, build_repo, target)

        result = await orchestrator.run_pipeline(build)

        # Update project status
        project = await project_repo.get(result.project_id)
        if project:
            if result.status == BuildStatus.COMPLETED:
                project.status = "deployed"
                project.active_build_id = result.id
            elif result.status == BuildStatus.AWAITING_APPROVAL:
                # Paused for human approval — keep it "building", not failed.
                project.status = "building"
            else:
                project.status = "failed"
            await project_repo.update(project)

    except Exception as e:
        logger.exception(f"Pipeline failed for build {build.id}")
        if session:
            try:
                build_repo = BuildRepository(session)
                build.status = BuildStatus.FAILED
                build.error = str(e)
                build.completed_at = datetime.now(timezone.utc)
                await build_repo.update(build)
            except Exception:
                pass
    finally:
        if session:
            session.close()


# ────────────────────────────────────────────────────
# Knowledge base (persistent memory / grounding)
# ────────────────────────────────────────────────────


class KnowledgeIngestRequest(BaseModel):
    """Ingest a document into the persistent knowledge store."""

    title: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1, max_length=200_000)
    source: Optional[str] = Field(None, max_length=1000)
    metadata: Optional[dict] = None


@router.post("/knowledge", status_code=201)
async def ingest_knowledge(
    body: KnowledgeIngestRequest,
    store=Depends(get_knowledge_store),
    _: Principal = Depends(require_user),
):
    """Ingest a document into the persistent knowledge store.

    Ingested documents become available for retrieval-augmented grounding
    so the pipeline can cite real organizational knowledge instead of
    guessing.
    """
    try:
        doc_id = store.ingest(
            title=body.title,
            text=body.text,
            source=body.source,
            metadata=body.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": doc_id, "title": body.title, "total": store.count()}


@router.get("/knowledge/search")
async def search_knowledge(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(5, ge=1, le=50, description="Max results"),
    store=Depends(get_knowledge_store),
):
    """Search the persistent knowledge store (deterministic TF-IDF offline)."""
    results = store.search(q, k=k)
    return {
        "query": q,
        "items": [
            {
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "score": round(score, 6),
                "snippet": doc.text[:280],
            }
            for doc, score in results
        ],
        "total": len(results),
    }


# ────────────────────────────────────────────────────
# Feedback loop → auto-rebuild lineage
# ────────────────────────────────────────────────────

MAX_REBUILD_DEPTH = 3
_feedback_collector = FeedbackCollector()


class FeedbackSubmitRequest(BaseModel):
    """Production metrics reported for one agent in a build."""

    agent_name: str = Field(..., min_length=1, max_length=200)
    agent_version: str = Field("", max_length=200)
    total_requests: int = Field(0, ge=0)
    successful_responses: int = Field(0, ge=0)
    avg_latency_ms: float = Field(0.0, ge=0.0)
    avg_confidence: float = Field(0.0, ge=0.0, le=1.0)
    error_rate: float = Field(0.0, ge=0.0, le=1.0)
    user_satisfaction: float = Field(0.0, ge=0.0, le=1.0)
    tool_usage: dict = Field(default_factory=dict)
    common_escalations: list = Field(default_factory=list)


class RebuildRequest(BaseModel):
    """Optional overrides for a feedback-seeded rebuild."""

    feedback_seed: Optional[str] = Field(
        None,
        max_length=4000,
        description="Compact guidance injected into the rebuild. "
        "Defaults to the generated feedback summary.",
    )


def _build_agent_names(build: Build) -> list:
    """Derive agent names from a build's BUILD-stage artifacts."""
    if not build.artifacts:
        return []
    return [
        a.get("name")
        for a in build.artifacts.get("agents", [])
        if isinstance(a, dict) and a.get("name")
    ]


@router.post("/builds/{build_id}/feedback", status_code=201)
async def submit_feedback(
    build_id: str,
    body: FeedbackSubmitRequest,
    build_repo: BuildRepository = Depends(get_build_repo),
    _: Principal = Depends(require_user),
):
    """Record production metrics for one of a build's agents."""
    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    metrics = AgentMetrics(
        agent_name=body.agent_name,
        agent_version=body.agent_version,
        total_requests=body.total_requests,
        successful_responses=body.successful_responses,
        avg_latency_ms=body.avg_latency_ms,
        avg_confidence=body.avg_confidence,
        error_rate=body.error_rate,
        user_satisfaction=body.user_satisfaction,
        tool_usage=body.tool_usage,
        common_escalations=body.common_escalations,
    )
    _feedback_collector.record_metrics(metrics)
    return {"recorded": True, "build_id": build_id, "agent_name": body.agent_name}


@router.get("/builds/{build_id}/feedback")
async def get_feedback(
    build_id: str,
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Generate a feedback report from recorded metrics for this build."""
    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    agent_names = _build_agent_names(build)
    report = _feedback_collector.generate_report(build_id, agent_names)
    seed = _feedback_collector.feed_back_to_generator(report)
    return {
        "build_id": build_id,
        "agent_names": agent_names,
        "overall_score": report.overall_score,
        "insights": [
            {
                "agent_name": i.agent_name,
                "category": i.category,
                "severity": i.severity,
                "insight": i.insight,
                "suggested_fix": i.suggested_fix,
            }
            for i in report.insights
        ],
        "recommendations": report.recommendations,
        "rebuild_seed": seed,
    }


async def _rebuild_depth(build: Build, build_repo: BuildRepository) -> int:
    """Count how many ancestors this build chain already has."""
    depth = 0
    parent_id = build.parent_build_id
    while parent_id and depth < MAX_REBUILD_DEPTH + 1:
        parent = await build_repo.get(parent_id)
        if not parent:
            break
        depth += 1
        parent_id = parent.parent_build_id
    return depth


@router.post("/builds/{build_id}/rebuild", status_code=202)
async def rebuild(
    build_id: str,
    body: RebuildRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
    build_repo: BuildRepository = Depends(get_build_repo),
    _: Principal = Depends(require_user),
):
    """Spawn a feedback-seeded child build from a finished build.

    Lineage only — the child records its `parent_build_id` and the
    `feedback_seed` that informed the rebuild. Guards against runaway
    self-rebuilding via MAX_REBUILD_DEPTH and refuses to rebuild a build
    that is still running.
    """
    parent = await build_repo.get(build_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Build not found")
    if not parent.is_terminal:
        raise HTTPException(
            status_code=409,
            detail="Cannot rebuild a build that is still running.",
        )

    depth = await _rebuild_depth(parent, build_repo)
    if depth >= MAX_REBUILD_DEPTH:
        raise HTTPException(
            status_code=409,
            detail=f"Rebuild depth limit reached (max {MAX_REBUILD_DEPTH}).",
        )

    # Derive a feedback seed if the caller did not supply one.
    seed = body.feedback_seed
    if not seed:
        report = _feedback_collector.generate_report(
            build_id, _build_agent_names(parent)
        )
        if report.recommendations:
            seed = " | ".join(report.recommendations[:5])

    child = Build(
        project_id=parent.project_id,
        problem_description=parent.problem_description,
        target=parent.target,
        target_config=parent.target_config,
        parent_build_id=parent.id,
        feedback_seed=seed,
    )
    await build_repo.create(child)

    project = await project_repo.get(parent.project_id)
    if project:
        project.build_count += 1
        project.last_build_id = child.id
        project.status = "building"
        await project_repo.update(project)

    asyncio.create_task(_run_pipeline(child))

    return {
        "build_id": child.id,
        "parent_build_id": parent.id,
        "feedback_seed": seed,
        "rebuild_depth": depth + 1,
        "status": "queued",
        "status_url": f"/v1/builds/{child.id}",
        "logs_url": f"/v1/builds/{child.id}/logs",
    }


# ────────────────────────────────────────────────────
# P5: Template / recipe gallery
# ────────────────────────────────────────────────────


@router.get("/templates")
async def list_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    q: Optional[str] = Query(None, description="Free-text search"),
):
    """List curated, one-click system recipes."""
    from genesis.templates import list_recipes, list_categories

    recipes = list_recipes(category=category, q=q)
    return {
        "items": [r.to_dict() for r in recipes],
        "total": len(recipes),
        "categories": list_categories(),
    }


@router.get("/templates/categories")
async def template_categories():
    """List distinct template categories."""
    from genesis.templates import list_categories

    return {"categories": list_categories()}


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Fetch a single recipe by id."""
    from genesis.templates import get_recipe

    recipe = get_recipe(template_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Template not found")
    return recipe.to_dict()


# ────────────────────────────────────────────────────
# P5: One-click deployment package
# ────────────────────────────────────────────────────


@router.get("/builds/{build_id}/deploy-package")
async def get_deploy_package(
    build_id: str,
    topology: str = Query("router", description="Topology label for the guide"),
    project_name: Optional[str] = Query(None, description="Override project name"),
    allow_partial: bool = Query(
        False, description="Skip invalid agents instead of erroring"
    ),
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Generate a ready-to-run deployment package (Docker, Azure, K8s, local).

    Reconstructs the agents from build artifacts and produces step-by-step
    guides the user can follow to run the system in their own environment.
    """
    from genesis.deployment.guides import DeploymentGuideGenerator
    from genesis.orchestrator.deploy_finalize import reconstruct_agents

    build = await build_repo.get(build_id)
    if not build or not build.artifacts:
        raise HTTPException(status_code=404, detail="No artifacts available")

    agent_dicts = build.artifacts.get("agents", [])
    if not agent_dicts:
        raise HTTPException(status_code=404, detail="No agents in this build")

    try:
        agents = reconstruct_agents(agent_dicts, allow_partial=allow_partial)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not agents:
        raise HTTPException(status_code=422, detail="No valid agents to package")

    name = project_name or f"genesis-{build_id[:8]}"
    package = DeploymentGuideGenerator().generate(
        agents, project_name=name, topology=topology
    )
    return package.to_dict()


# ────────────────────────────────────────────────────
# P5: Versioning + rollback
# ────────────────────────────────────────────────────


@router.get("/projects/{project_id}/versions")
async def list_project_versions(
    project_id: str,
    project_repo: ProjectRepository = Depends(get_project_repo),
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """List a project's builds as versions (v1..vN), marking the active one."""
    project = await project_repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    builds = await build_repo.list_by_project(project_id)
    # list_by_project returns newest-first; number oldest→newest for stable vN.
    ordered = sorted(builds, key=lambda b: b.created_at)
    active_id = project.active_build_id

    versions = []
    for idx, b in enumerate(ordered, start=1):
        deployment = (b.artifacts or {}).get("deployment") if b.artifacts else None
        versions.append({
            "version": f"v{idx}",
            "build_id": b.id,
            "status": b.status.value,
            "is_active": b.id == active_id,
            "is_deployable": bool((b.artifacts or {}).get("agents")),
            "created_at": b.created_at.isoformat() if b.created_at else None,
            "endpoint_url": deployment.get("endpoint_url") if deployment else None,
            "parent_build_id": b.parent_build_id,
        })

    return {
        "project_id": project_id,
        "active_build_id": active_id,
        "total": len(versions),
        "versions": versions,
    }


class RollbackRequest(BaseModel):
    """Re-deploy a prior build and point live traffic at it."""

    build_id: str = Field(..., min_length=1)


@router.post("/projects/{project_id}/rollback", status_code=202)
async def rollback_project(
    project_id: str,
    body: RollbackRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
    build_repo: BuildRepository = Depends(get_build_repo),
    _: Principal = Depends(require_admin),
):
    """Roll a project back to a previous deployable build.

    Re-runs DEPLOY for the target build (admin-only) and flips the project's
    active build pointer. The target build must belong to the project and have
    deployable agent artifacts.
    """
    from genesis.orchestrator.deploy_finalize import finalize_deploy, reconstruct_agents

    project = await project_repo.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    target_build = await build_repo.get(body.build_id)
    if not target_build or target_build.project_id != project_id:
        raise HTTPException(status_code=404, detail="Build not found in project")

    agent_dicts = (target_build.artifacts or {}).get("agents", [])
    if not agent_dicts:
        raise HTTPException(
            status_code=409, detail="Target build has no deployable agents"
        )

    try:
        agents = reconstruct_agents(agent_dicts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target = get_deployment_target()
    try:
        await finalize_deploy(
            target_build,
            agents,
            target,
            build_repo,
            project_repo,
            decrypt=decrypt_config_secrets,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Rollback deploy failed for project %s", project_id)
        raise HTTPException(status_code=502, detail=f"Rollback failed: {exc}") from exc

    return {
        "project_id": project_id,
        "active_build_id": target_build.id,
        "status": "deployed",
        "endpoint_url": (target_build.artifacts or {})
        .get("deployment", {})
        .get("endpoint_url"),
    }


# ────────────────────────────────────────────────────
# P5: Human-in-the-loop approval gate
# ────────────────────────────────────────────────────


class ApprovalRequest(BaseModel):
    """Approve or reject a paused build."""

    approver: Optional[str] = Field(None, max_length=200)
    reason: Optional[str] = Field(None, max_length=2000)


@router.get("/builds/{build_id}/approval")
async def get_build_approval(
    build_id: str,
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Inspect the approval state of a build."""
    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    decision = (build.artifacts or {}).get("approval")
    return {
        "build_id": build.id,
        "status": build.status.value,
        "is_paused": build.is_paused,
        "requires_approval": bool((build.target_config or {}).get("require_approval")),
        "decision": decision,
    }


@router.post("/builds/{build_id}/approve", status_code=202)
async def approve_build(
    build_id: str,
    body: ApprovalRequest,
    principal: Principal = Depends(require_admin),
    project_repo: ProjectRepository = Depends(get_project_repo),
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Approve a paused build and finish deploying it (admin-only).

    Idempotency derives from build status: approval is only valid while the
    build is AWAITING_APPROVAL; any other state returns 409.
    """
    from genesis.orchestrator.deploy_finalize import finalize_deploy, reconstruct_agents

    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    if not build.is_paused:
        raise HTTPException(
            status_code=409,
            detail=f"Build is not awaiting approval (status={build.status.value}).",
        )

    agent_dicts = (build.artifacts or {}).get("agents", [])
    if not agent_dicts:
        raise HTTPException(status_code=409, detail="Paused build has no agents")
    try:
        agents = reconstruct_agents(agent_dicts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Record the decision in artifacts before deploying.
    artifacts = dict(build.artifacts or {})
    artifacts["approval"] = {
        "approved": True,
        "approver": body.approver or principal.key_id,
        "reason": body.reason,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    build.artifacts = artifacts

    target = get_deployment_target()
    try:
        await finalize_deploy(
            build,
            agents,
            target,
            build_repo,
            project_repo,
            decrypt=decrypt_config_secrets,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Approval deploy failed for build %s", build_id)
        raise HTTPException(status_code=502, detail=f"Deploy failed: {exc}") from exc

    return {
        "build_id": build.id,
        "status": build.status.value,
        "endpoint_url": (build.artifacts or {})
        .get("deployment", {})
        .get("endpoint_url"),
    }


@router.post("/builds/{build_id}/reject", status_code=202)
async def reject_build(
    build_id: str,
    body: ApprovalRequest,
    principal: Principal = Depends(require_admin),
    project_repo: ProjectRepository = Depends(get_project_repo),
    build_repo: BuildRepository = Depends(get_build_repo),
):
    """Reject a paused build, marking it FAILED (admin-only)."""
    build = await build_repo.get(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    if not build.is_paused:
        raise HTTPException(
            status_code=409,
            detail=f"Build is not awaiting approval (status={build.status.value}).",
        )

    artifacts = dict(build.artifacts or {})
    artifacts["approval"] = {
        "approved": False,
        "approver": body.approver or principal.key_id,
        "reason": body.reason,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    build.artifacts = artifacts
    build.status = BuildStatus.FAILED
    build.error = "Rejected by approver"
    build.completed_at = datetime.now(timezone.utc)
    await build_repo.update(build)

    project = await project_repo.get(build.project_id)
    if project:
        project.status = "failed"
        await project_repo.update(project)

    return {"build_id": build.id, "status": build.status.value}
