"""Tests for ProjectRepository CRUD operations."""

import pytest
from datetime import datetime, timezone

from genesis.models.project import Project, ProjectStatus
from genesis.storage.repository import ProjectRepository


class TestProjectRepositoryCreate:
    """Tests for ProjectRepository.create()."""

    async def test_create_project(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="test-project", description="A test project")
        result = await repo.create(p)
        assert result is p
        assert result.id is not None

    async def test_create_persists_to_db(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="persisted", description="Should persist")
        await repo.create(p)
        fetched = await repo.get(p.id)
        assert fetched is not None
        assert fetched.name == "persisted"
        assert fetched.description == "Should persist"

    async def test_create_with_explicit_status(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="building-proj", description="x",
                    status=ProjectStatus.BUILDING)
        await repo.create(p)
        fetched = await repo.get(p.id)
        assert fetched.status == ProjectStatus.BUILDING

    async def test_create_with_build_count(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="with-builds", description="x", build_count=5,
                    last_build_id="b-1")
        await repo.create(p)
        fetched = await repo.get(p.id)
        assert fetched.build_count == 5
        assert fetched.last_build_id == "b-1"

    async def test_create_with_timestamps(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="timestamps", description="x")
        await repo.create(p)
        fetched = await repo.get(p.id)
        assert fetched.created_at is not None
        assert fetched.updated_at is not None


class TestProjectRepositoryGet:
    """Tests for ProjectRepository.get()."""

    async def test_get_existing(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="get-test", description="desc")
        await repo.create(p)
        result = await repo.get(p.id)
        assert result is not None
        assert result.id == p.id
        assert result.name == "get-test"

    async def test_get_nonexistent(self, db_session):
        repo = ProjectRepository(db_session)
        result = await repo.get("nonexistent-id")
        assert result is None

    async def test_get_returns_correct_type(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="type-test", description="x")
        await repo.create(p)
        result = await repo.get(p.id)
        assert isinstance(result, Project)


class TestProjectRepositoryListAll:
    """Tests for ProjectRepository.list_all()."""

    async def test_list_all_empty(self, db_session):
        repo = ProjectRepository(db_session)
        results = await repo.list_all()
        assert results == []

    async def test_list_all_with_items(self, db_session):
        repo = ProjectRepository(db_session)
        p1 = Project(name="p1", description="first")
        p2 = Project(name="p2", description="second")
        await repo.create(p1)
        await repo.create(p2)
        results = await repo.list_all()
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"p1", "p2"}

    async def test_list_all_returns_models(self, db_session):
        repo = ProjectRepository(db_session)
        await repo.create(Project(name="p", description="d"))
        results = await repo.list_all()
        for r in results:
            assert isinstance(r, Project)

    async def test_list_all_ordering_by_updated_at_desc(self, db_session):
        repo = ProjectRepository(db_session)
        p1 = Project(name="first", description="1")
        p2 = Project(name="second", description="2")
        await repo.create(p1)
        # Small delay to ensure updated_at differs
        import asyncio
        await asyncio.sleep(0.01)
        await repo.create(p2)
        results = await repo.list_all()
        assert results[0].name == "second"
        assert results[1].name == "first"


class TestProjectRepositoryUpdate:
    """Tests for ProjectRepository.update()."""

    async def test_update_existing(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="update-me", description="original")
        await repo.create(p)
        p.name = "updated-name"
        p.description = "updated desc"
        p.status = ProjectStatus.DEPLOYED
        result = await repo.update(p)
        assert result.name == "updated-name"
        assert result.description == "updated desc"
        assert result.status == ProjectStatus.DEPLOYED
        fetched = await repo.get(p.id)
        assert fetched.name == "updated-name"

    async def test_update_nonexistent_raises(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(id="nonexistent", name="ghost", description="boo")
        with pytest.raises(ValueError, match="not found"):
            await repo.update(p)

    async def test_update_updates_timestamp(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="ts-test", description="x")
        await repo.create(p)
        original_updated = (await repo.get(p.id)).updated_at
        import asyncio
        await asyncio.sleep(0.01)
        p.name = "ts-updated"
        await repo.update(p)
        fetched = await repo.get(p.id)
        assert fetched.updated_at > original_updated

    async def test_update_partial_fields(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="partial", description="original desc",
                    status=ProjectStatus.ACTIVE)
        await repo.create(p)
        p.description = "new desc"
        # name unchanged
        result = await repo.update(p)
        assert result.name == "partial"
        assert result.description == "new desc"


class TestProjectRepositoryDelete:
    """Tests for ProjectRepository.delete()."""

    async def test_delete_existing(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="delete-me", description="x")
        await repo.create(p)
        result = await repo.delete(p.id)
        assert result is True
        assert await repo.get(p.id) is None

    async def test_delete_nonexistent(self, db_session):
        repo = ProjectRepository(db_session)
        result = await repo.delete("nonexistent")
        assert result is False

    async def test_delete_then_get_returns_none(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(name="gone", description="x")
        await repo.create(p)
        await repo.delete(p.id)
        assert await repo.get(p.id) is None


class TestProjectRepositoryIntegration:
    """Integration scenarios combining multiple operations."""

    async def test_create_update_delete_cycle(self, db_session):
        repo = ProjectRepository(db_session)
        # Create
        p = Project(name="lifecycle", description="test")
        await repo.create(p)
        assert await repo.get(p.id) is not None

        # Update
        p.name = "lifecycle-updated"
        await repo.update(p)
        assert (await repo.get(p.id)).name == "lifecycle-updated"

        # Delete
        await repo.delete(p.id)
        assert await repo.get(p.id) is None

    async def test_list_all_after_delete_reflects_change(self, db_session):
        repo = ProjectRepository(db_session)
        p1 = Project(name="keep", description="x")
        p2 = Project(name="remove", description="x")
        await repo.create(p1)
        await repo.create(p2)
        assert len(await repo.list_all()) == 2
        await repo.delete(p2.id)
        assert len(await repo.list_all()) == 1

    async def test_multiple_creates_and_gets(self, db_session):
        repo = ProjectRepository(db_session)
        projects = []
        for i in range(5):
            p = Project(name=f"project-{i}", description=f"desc-{i}")
            await repo.create(p)
            projects.append(p)
        for p in projects:
            fetched = await repo.get(p.id)
            assert fetched is not None
            assert fetched.name.startswith("project-")

    async def test_update_roundtrip_fidelity(self, db_session):
        repo = ProjectRepository(db_session)
        p = Project(
            name="roundtrip",
            description="original",
            status=ProjectStatus.BUILDING,
            build_count=3,
            last_build_id="lb-42",
        )
        await repo.create(p)
        p.name = "roundtrip-new"
        p.description = "updated"
        p.status = ProjectStatus.DEPLOYED
        p.build_count = 4
        p.last_build_id = "lb-43"
        await repo.update(p)
        fetched = await repo.get(p.id)
        assert fetched.name == "roundtrip-new"
        assert fetched.description == "updated"
        assert fetched.status == ProjectStatus.DEPLOYED
        assert fetched.build_count == 4
        assert fetched.last_build_id == "lb-43"
