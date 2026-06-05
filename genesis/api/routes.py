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
from genesis.storage.repository import ProjectRepository, BuildRepository
from genesis.orchestrator.state_machine import Orchestrator
from genesis.api.dependencies import (
    get_project_repo,
    get_build_repo,
    get_llm_provider,
    get_deployment_target,
)
from genesis.security.auth import require_user, require_admin, Principal
from genesis.security.secrets import encrypt_config_secrets, decrypt_config_secrets
from genesis.observability.cost import get_cost_tracker
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
            project.status = (
                "deployed" if result.status == BuildStatus.COMPLETED else "failed"
            )
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
