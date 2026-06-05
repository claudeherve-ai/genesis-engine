"""Offline tests for the Phase 5 platform API surface.

Covers the template/recipe gallery, one-click deployment packaging, project
versioning, rollback, and the human-in-the-loop approval gate — all reachable
with no API keys (auth is open when ``GENESIS_API_KEYS`` is unset, and the
deployment target runs in dry-run mode without ``AGENTSYSTEM_API_KEY``).
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from genesis.api.app import app
from genesis.models.build import Build, BuildStatus
from genesis.models.project import Project
from genesis.storage.database import get_session
from genesis.storage.repository import BuildRepository, ProjectRepository


def _client(monkeypatch):
    monkeypatch.delenv("GENESIS_API_KEYS", raising=False)
    monkeypatch.delenv("GENESIS_RATE_LIMIT", raising=False)
    monkeypatch.delenv("AGENTSYSTEM_API_KEY", raising=False)
    monkeypatch.delenv("GENESIS_SECRET_KEY", raising=False)
    return TestClient(app)


def _persist(*objs):
    async def _go():
        session = get_session()
        try:
            proj_repo = ProjectRepository(session)
            build_repo = BuildRepository(session)
            for obj in objs:
                if isinstance(obj, Project):
                    await proj_repo.create(obj)
                else:
                    await build_repo.create(obj)
        finally:
            session.close()

    asyncio.run(_go())


def _fetch_project(project_id):
    async def _go():
        session = get_session()
        try:
            return await ProjectRepository(session).get(project_id)
        finally:
            session.close()

    return asyncio.run(_go())


VALID_AGENT = {
    "name": "triage_router",
    "role": "Router",
    "system_prompt": "Route incoming tickets to the right specialist.",
}
INVALID_AGENT = {"name": "Bad-Name", "role": "x"}  # bad regex + missing prompt


# ── Templates ────────────────────────────────────────────────


def test_templates_list_all(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.get("/v1/templates")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 8
        assert len(body["items"]) == 8
        assert body["categories"]
        first = body["items"][0]
        for key in ("id", "title", "category", "summary", "prompt", "target"):
            assert key in first


def test_templates_category_filter(monkeypatch):
    with _client(monkeypatch) as client:
        cats = client.get("/v1/templates/categories").json()["categories"]
        assert cats
        target = cats[0]
        r = client.get("/v1/templates", params={"category": target})
        assert r.status_code == 200
        items = r.json()["items"]
        assert items
        assert all(i["category"] == target for i in items)
        assert len(items) < 8 or len(cats) == 1


def test_templates_text_search(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.get("/v1/templates", params={"q": "support"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        assert body["total"] <= 8


def test_template_get_and_404(monkeypatch):
    with _client(monkeypatch) as client:
        listed = client.get("/v1/templates").json()["items"][0]
        got = client.get(f"/v1/templates/{listed['id']}")
        assert got.status_code == 200
        assert got.json()["id"] == listed["id"]

        missing = client.get("/v1/templates/does-not-exist")
        assert missing.status_code == 404


# ── Deploy package ───────────────────────────────────────────


def _build_with_agents(agents, status=BuildStatus.COMPLETED, project=None):
    project = project or Project(name=f"p-{uuid.uuid4().hex[:8]}", description="x")
    build = Build(
        project_id=project.id,
        problem_description="route support tickets to the right team",
        status=status,
        artifacts={"agents": agents} if agents is not None else None,
    )
    return project, build


def test_deploy_package_success(monkeypatch):
    project, build = _build_with_agents([VALID_AGENT])
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.get(f"/v1/builds/{build.id}/deploy-package")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["agent_count"] == 1
        names = {o["name"] for o in body["options"]}
        assert {"docker", "local"}.issubset(names)


def test_deploy_package_invalid_agents_422(monkeypatch):
    project, build = _build_with_agents([INVALID_AGENT])
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.get(f"/v1/builds/{build.id}/deploy-package")
        assert r.status_code == 422, r.text


def test_deploy_package_allow_partial_skips_invalid(monkeypatch):
    project, build = _build_with_agents([VALID_AGENT, INVALID_AGENT])
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.get(
            f"/v1/builds/{build.id}/deploy-package",
            params={"allow_partial": True},
        )
        assert r.status_code == 200, r.text
        assert r.json()["agent_count"] == 1


def test_deploy_package_no_artifacts_404(monkeypatch):
    project, build = _build_with_agents(None)
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.get(f"/v1/builds/{build.id}/deploy-package")
        assert r.status_code == 404


def test_deploy_package_no_agents_404(monkeypatch):
    project, build = _build_with_agents([])
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.get(f"/v1/builds/{build.id}/deploy-package")
        assert r.status_code == 404


# ── Versions ─────────────────────────────────────────────────


def test_versions_ordering_and_active(monkeypatch):
    project = Project(name=f"p-{uuid.uuid4().hex[:8]}", description="x")
    t0 = datetime.now(timezone.utc)
    b1 = Build(
        project_id=project.id,
        problem_description="version one of the system",
        status=BuildStatus.COMPLETED,
        artifacts={"agents": [VALID_AGENT]},
        created_at=t0,
    )
    b2 = Build(
        project_id=project.id,
        problem_description="version two of the system",
        status=BuildStatus.COMPLETED,
        artifacts={"agents": [VALID_AGENT]},
        created_at=t0 + timedelta(seconds=5),
    )
    project.active_build_id = b2.id
    _persist(project, b1, b2)

    with _client(monkeypatch) as client:
        r = client.get(f"/v1/projects/{project.id}/versions")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 2
        assert body["active_build_id"] == b2.id
        v1, v2 = body["versions"]
        assert v1["version"] == "v1" and v1["build_id"] == b1.id
        assert v2["version"] == "v2" and v2["build_id"] == b2.id
        assert v1["is_active"] is False
        assert v2["is_active"] is True
        assert v1["is_deployable"] is True


def test_versions_unknown_project_404(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.get(f"/v1/projects/{uuid.uuid4()}/versions")
        assert r.status_code == 404


# ── Rollback ─────────────────────────────────────────────────


def test_rollback_flips_active_pointer(monkeypatch):
    project = Project(name=f"p-{uuid.uuid4().hex[:8]}", description="x")
    t0 = datetime.now(timezone.utc)
    b1 = Build(
        project_id=project.id,
        problem_description="stable previous version of the system",
        status=BuildStatus.COMPLETED,
        artifacts={"agents": [VALID_AGENT]},
        created_at=t0,
    )
    b2 = Build(
        project_id=project.id,
        problem_description="regressed latest version of the system",
        status=BuildStatus.COMPLETED,
        artifacts={"agents": [VALID_AGENT]},
        created_at=t0 + timedelta(seconds=5),
    )
    project.active_build_id = b2.id
    project.status = "deployed"
    _persist(project, b1, b2)

    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/projects/{project.id}/rollback",
            json={"build_id": b1.id},
        )
        assert r.status_code == 202, r.text
        assert r.json()["active_build_id"] == b1.id

    refreshed = _fetch_project(project.id)
    assert refreshed.active_build_id == b1.id
    assert refreshed.status == "deployed"


def test_rollback_build_not_in_project_404(monkeypatch):
    project, build = _build_with_agents([VALID_AGENT])
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/projects/{project.id}/rollback",
            json={"build_id": str(uuid.uuid4())},
        )
        assert r.status_code == 404


def test_rollback_no_agents_409(monkeypatch):
    project, build = _build_with_agents([])
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/projects/{project.id}/rollback",
            json={"build_id": build.id},
        )
        assert r.status_code == 409


# ── Approval gate ────────────────────────────────────────────


def _paused_build():
    project = Project(name=f"p-{uuid.uuid4().hex[:8]}", description="x")
    build = Build(
        project_id=project.id,
        problem_description="needs a human to approve before going live",
        status=BuildStatus.AWAITING_APPROVAL,
        artifacts={"agents": [VALID_AGENT]},
        target_config={"require_approval": True},
    )
    return project, build


def test_approval_state_reflects_pause(monkeypatch):
    project, build = _paused_build()
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.get(f"/v1/builds/{build.id}/approval")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["is_paused"] is True
        assert body["status"] == "awaiting_approval"
        assert body["decision"] is None


def test_approve_paused_build_deploys(monkeypatch):
    project, build = _paused_build()
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/builds/{build.id}/approve",
            json={"approver": "alice", "reason": "looks good"},
        )
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "completed"

        approval = client.get(f"/v1/builds/{build.id}/approval").json()
        assert approval["status"] == "completed"
        assert approval["decision"]["approved"] is True
        assert approval["decision"]["approver"] == "alice"

    refreshed = _fetch_project(project.id)
    assert refreshed.active_build_id == build.id
    assert refreshed.status == "deployed"


def test_approve_non_paused_build_409(monkeypatch):
    project, build = _build_with_agents([VALID_AGENT])  # COMPLETED, not paused
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.post(f"/v1/builds/{build.id}/approve", json={})
        assert r.status_code == 409


def test_reject_paused_build_fails_it(monkeypatch):
    project, build = _paused_build()
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/builds/{build.id}/reject",
            json={"approver": "bob", "reason": "scope creep"},
        )
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "failed"

        approval = client.get(f"/v1/builds/{build.id}/approval").json()
        assert approval["decision"]["approved"] is False


def test_reject_non_paused_build_409(monkeypatch):
    project, build = _build_with_agents([VALID_AGENT])
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.post(f"/v1/builds/{build.id}/reject", json={})
        assert r.status_code == 409


def test_approve_unknown_build_404(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.post(f"/v1/builds/{uuid.uuid4()}/approve", json={})
        assert r.status_code == 404
