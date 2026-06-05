"""Offline tests for the Phase 4 API surface.

Covers the persistent knowledge store, the feedback loop, and the
feedback-seeded rebuild lineage — all reachable with no API keys.
"""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import app
from genesis.models.build import Build, BuildStatus
from genesis.models.project import Project
from genesis.storage.database import get_session
from genesis.storage.repository import BuildRepository, ProjectRepository


def _client(monkeypatch):
    monkeypatch.delenv("GENESIS_API_KEYS", raising=False)
    monkeypatch.delenv("GENESIS_RATE_LIMIT", raising=False)
    return TestClient(app)


def _persist(*objs):
    """Persist project/build objects through their repositories."""
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


def _terminal_build(agents=None):
    project = Project(name=f"p-{uuid.uuid4().hex[:8]}", description="x")
    build = Build(
        project_id=project.id,
        problem_description="route support tickets to the right team",
        status=BuildStatus.COMPLETED,
        artifacts={"agents": agents or [{"name": "triage_router"}]},
    )
    _persist(project, build)
    return project, build


# ── Knowledge store ──────────────────────────────────────────


def test_knowledge_ingest_and_search(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.post(
            "/v1/knowledge",
            json={
                "title": "Refund policy",
                "text": "Customers may request a refund within 30 days of purchase.",
                "source": "handbook",
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["id"]

        s = client.get("/v1/knowledge/search", params={"q": "refund", "k": 5})
        assert s.status_code == 200
        body = s.json()
        assert body["total"] >= 1
        assert any("refund" in i["title"].lower() or "refund" in i["snippet"].lower()
                   for i in body["items"])


def test_knowledge_ingest_rejects_empty_text(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.post(
            "/v1/knowledge",
            json={"title": "Empty", "text": "   "},
        )
        # Pydantic min_length=1 rejects whitespace-only? No — it has length.
        # The store raises ValueError on tokenization → 400.
        assert r.status_code in (400, 422)


# ── Feedback loop ────────────────────────────────────────────


def test_submit_and_get_feedback(monkeypatch):
    _, build = _terminal_build(agents=[{"name": f"agent_{uuid.uuid4().hex[:6]}"}])
    agent_name = build.artifacts["agents"][0]["name"]

    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/builds/{build.id}/feedback",
            json={
                "agent_name": agent_name,
                "total_requests": 100,
                "successful_responses": 60,
                "error_rate": 0.4,
                "user_satisfaction": 0.5,
            },
        )
        assert r.status_code == 201, r.text

        g = client.get(f"/v1/builds/{build.id}/feedback")
        assert g.status_code == 200
        body = g.json()
        assert agent_name in body["agent_names"]
        # 60% success is below the 80% threshold → a high-severity insight.
        assert any(i["severity"] in ("high", "critical") for i in body["insights"])
        assert body["recommendations"]


def test_feedback_rejects_out_of_range(monkeypatch):
    _, build = _terminal_build()
    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/builds/{build.id}/feedback",
            json={"agent_name": "x", "error_rate": 1.5},
        )
        assert r.status_code == 422


def test_feedback_unknown_build_404(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.get("/v1/builds/does-not-exist/feedback")
        assert r.status_code == 404


# ── Rebuild lineage ──────────────────────────────────────────


def test_rebuild_rejects_running_build(monkeypatch):
    project = Project(name=f"p-{uuid.uuid4().hex[:8]}", description="x")
    build = Build(
        project_id=project.id,
        problem_description="still going",
        status=BuildStatus.BUILDING,
    )
    _persist(project, build)
    with _client(monkeypatch) as client:
        r = client.post(f"/v1/builds/{build.id}/rebuild", json={})
        assert r.status_code == 409


def test_rebuild_creates_child_with_lineage(monkeypatch):
    _, build = _terminal_build()
    with _client(monkeypatch) as client:
        r = client.post(
            f"/v1/builds/{build.id}/rebuild",
            json={"feedback_seed": "tighten the triage prompt"},
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["parent_build_id"] == build.id
        assert body["feedback_seed"] == "tighten the triage prompt"
        assert body["rebuild_depth"] == 1


def test_rebuild_unknown_build_404(monkeypatch):
    with _client(monkeypatch) as client:
        r = client.post("/v1/builds/nope/rebuild", json={})
        assert r.status_code == 404
