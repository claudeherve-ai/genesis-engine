"""Tests for BuildRepository CRUD operations."""

import pytest
from datetime import datetime, timezone

from genesis.models.build import Build, BuildStatus, PipelineStage
from genesis.storage.repository import BuildRepository


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_build(
    project_id="p-default",
    problem_description="Default problem description long enough",
    **kwargs,
) -> Build:
    return Build(project_id=project_id, problem_description=problem_description, **kwargs)


# ── Create ──────────────────────────────────────────────────────────────────

class TestBuildRepositoryCreate:
    """Tests for BuildRepository.create()."""

    async def test_create_build(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build()
        result = await repo.create(b)
        assert result is b

    async def test_create_persists_to_db(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(project_id="pid-1")
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched is not None
        assert fetched.project_id == "pid-1"

    async def test_create_with_status(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(status=BuildStatus.BUILDING, stage=PipelineStage.BUILD)
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.status == BuildStatus.BUILDING
        assert fetched.stage == PipelineStage.BUILD

    async def test_create_with_stage_progress(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(stage_progress=0.75)
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.stage_progress == 0.75

    async def test_create_with_target_and_config(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(
            target="custom-target",
            target_config={"provider": "anthropic", "model": "claude-3"},
        )
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.target == "custom-target"
        assert fetched.target_config == {"provider": "anthropic", "model": "claude-3"}

    async def test_create_with_artifacts(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(artifacts={"agents": [{"name": "a1"}]})
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.artifacts == {"agents": [{"name": "a1"}]}

    async def test_create_with_test_results(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(test_results={"passed": 10, "failed": 2})
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.test_results == {"passed": 10, "failed": 2}

    async def test_create_with_error(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(error="Something went wrong")
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.error == "Something went wrong"

    async def test_create_with_retries(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(retries=3)
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.retries == 3

    async def test_create_with_completed_at(self, db_session):
        repo = BuildRepository(db_session)
        now = datetime.now(timezone.utc)
        b = make_build(status=BuildStatus.COMPLETED, completed_at=now)
        await repo.create(b)
        fetched = await repo.get(b.id)
        # SQLite drops tzinfo, so compare naive
        assert fetched.completed_at.replace(tzinfo=timezone.utc) == now

    async def test_create_with_none_stage(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(stage=None)
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.stage is None

    async def test_create_with_empty_target_config(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(target_config={})
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.target_config == {}


# ── Get ─────────────────────────────────────────────────────────────────────

class TestBuildRepositoryGet:
    """Tests for BuildRepository.get()."""

    async def test_get_existing(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build()
        await repo.create(b)
        result = await repo.get(b.id)
        assert result is not None
        assert result.id == b.id

    async def test_get_nonexistent(self, db_session):
        repo = BuildRepository(db_session)
        result = await repo.get("nonexistent-build-id")
        assert result is None

    async def test_get_returns_correct_type(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build()
        await repo.create(b)
        result = await repo.get(b.id)
        assert isinstance(result, Build)


# ── List by Project ─────────────────────────────────────────────────────────

class TestBuildRepositoryListByProject:
    """Tests for BuildRepository.list_by_project()."""

    async def test_list_empty_project(self, db_session):
        repo = BuildRepository(db_session)
        results = await repo.list_by_project("nonexistent")
        assert results == []

    async def test_list_by_project_returns_only_matching(self, db_session):
        repo = BuildRepository(db_session)
        b1 = make_build(project_id="proj-a")
        b2 = make_build(project_id="proj-b")
        await repo.create(b1)
        await repo.create(b2)
        results_a = await repo.list_by_project("proj-a")
        assert len(results_a) == 1
        assert results_a[0].project_id == "proj-a"
        results_b = await repo.list_by_project("proj-b")
        assert len(results_b) == 1
        assert results_b[0].project_id == "proj-b"

    async def test_list_multiple_builds_same_project(self, db_session):
        repo = BuildRepository(db_session)
        for i in range(5):
            b = make_build(project_id="multi-proj")
            await repo.create(b)
        results = await repo.list_by_project("multi-proj")
        assert len(results) == 5
        for r in results:
            assert r.project_id == "multi-proj"

    async def test_list_returns_models(self, db_session):
        repo = BuildRepository(db_session)
        await repo.create(make_build(project_id="p"))
        results = await repo.list_by_project("p")
        for r in results:
            assert isinstance(r, Build)

    async def test_list_ordered_by_created_at_desc(self, db_session):
        repo = BuildRepository(db_session)
        b1 = make_build(project_id="order-proj")
        b2 = make_build(project_id="order-proj")
        await repo.create(b1)
        import asyncio
        await asyncio.sleep(0.01)
        await repo.create(b2)
        results = await repo.list_by_project("order-proj")
        assert results[0].id == b2.id
        assert results[1].id == b1.id


# ── Update ──────────────────────────────────────────────────────────────────

class TestBuildRepositoryUpdate:
    """Tests for BuildRepository.update()."""

    async def test_update_existing(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(status=BuildStatus.QUEUED)
        await repo.create(b)
        b.status = BuildStatus.BUILDING
        b.stage = PipelineStage.BUILD
        b.stage_progress = 0.42
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.status == BuildStatus.BUILDING
        assert fetched.stage == PipelineStage.BUILD
        assert fetched.stage_progress == 0.42

    async def test_update_nonexistent_raises(self, db_session):
        repo = BuildRepository(db_session)
        b = Build(id="ghost-build", project_id="p", problem_description="x" * 10)
        with pytest.raises(ValueError, match="not found"):
            await repo.update(b)

    async def test_update_artifacts(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build()
        await repo.create(b)
        b.artifacts = {"deployed": ["agent-1", "agent-2"]}
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.artifacts == {"deployed": ["agent-1", "agent-2"]}

    async def test_update_test_results(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build()
        await repo.create(b)
        b.test_results = {"score": 0.95}
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.test_results == {"score": 0.95}

    async def test_update_error(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build()
        await repo.create(b)
        b.error = "Timeout during testing"
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.error == "Timeout during testing"

    async def test_update_to_none_error(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(error="old error")
        await repo.create(b)
        b.error = None
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.error is None

    async def test_update_completed_at(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(status=BuildStatus.COMPLETED,
                       completed_at=None)
        await repo.create(b)
        now = datetime.now(timezone.utc)
        b.completed_at = now
        await repo.update(b)
        fetched = await repo.get(b.id)
        # SQLite drops tzinfo, so compare naive
        assert fetched.completed_at.replace(tzinfo=timezone.utc) == now

    async def test_update_retries(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(retries=0)
        await repo.create(b)
        b.retries = 2
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.retries == 2

    async def test_update_preserves_immutable_fields(self, db_session):
        """Fields like project_id, problem_description are not updated by repo."""
        repo = BuildRepository(db_session)
        b = make_build(project_id="original-pid",
                       problem_description="original problem description")
        await repo.create(b)
        b.project_id = "new-pid"  # mutation that won't persist
        await repo.update(b)
        fetched = await repo.get(b.id)
        # project_id is NOT updated — the repo doesn't write it back
        assert fetched.project_id == "original-pid"
        # problem_description also not updated by repo
        assert fetched.problem_description == "original problem description"


# ── Integration Scenarios ───────────────────────────────────────────────────

class TestBuildRepositoryIntegration:
    """Full lifecycle and cross-operation scenarios."""

    async def test_full_lifecycle(self, db_session):
        repo = BuildRepository(db_session)
        # Create
        b = make_build(project_id="lifecycle-proj")
        await repo.create(b)
        assert await repo.get(b.id) is not None

        # Update status through pipeline stages
        stages = [
            (BuildStatus.ANALYZING, PipelineStage.ANALYZE),
            (BuildStatus.ARCHITECTING, PipelineStage.ARCHITECT),
            (BuildStatus.BUILDING, PipelineStage.BUILD),
            (BuildStatus.TESTING, PipelineStage.TEST),
            (BuildStatus.DEPLOYING, PipelineStage.DEPLOY),
            (BuildStatus.COMPLETED, None),
        ]
        for status, stage in stages:
            b.status = status
            b.stage = stage
            b.stage_progress = 1.0
            await repo.update(b)
            fetched = await repo.get(b.id)
            assert fetched.status == status
            assert fetched.stage == stage

        assert b.is_terminal is True

    async def test_multiple_builds_across_projects(self, db_session):
        repo = BuildRepository(db_session)
        for pid in ["proj-x", "proj-y", "proj-z"]:
            for i in range(2):
                await repo.create(make_build(project_id=pid))
        assert len(await repo.list_by_project("proj-x")) == 2
        assert len(await repo.list_by_project("proj-y")) == 2
        assert len(await repo.list_by_project("proj-z")) == 2

    async def test_terminal_build_lifecycle(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build(status=BuildStatus.QUEUED)
        await repo.create(b)
        assert b.is_terminal is False
        b.status = BuildStatus.COMPLETED
        b.completed_at = datetime.now(timezone.utc)
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.is_terminal is True
        assert fetched.completed_at is not None

    async def test_failed_build_with_error(self, db_session):
        repo = BuildRepository(db_session)
        b = make_build()
        await repo.create(b)
        b.status = BuildStatus.FAILED
        b.error = "Pipeline failed at TEST stage"
        b.completed_at = datetime.now(timezone.utc)
        await repo.update(b)
        fetched = await repo.get(b.id)
        assert fetched.status == BuildStatus.FAILED
        assert fetched.error == "Pipeline failed at TEST stage"
        assert fetched.is_terminal is True

    async def test_build_with_none_fields_roundtrip(self, db_session):
        """Verify None fields come back as None after create+get roundtrip."""
        repo = BuildRepository(db_session)
        b = make_build(
            stage=None,
            artifacts=None,
            test_results=None,
            error=None,
            completed_at=None,
        )
        await repo.create(b)
        fetched = await repo.get(b.id)
        assert fetched.stage is None
        assert fetched.artifacts is None
        assert fetched.test_results is None
        assert fetched.error is None
        assert fetched.completed_at is None
