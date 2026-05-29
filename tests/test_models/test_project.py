"""Tests for Project model, enums, and request schemas."""

import pytest
import uuid
from datetime import datetime, timezone
from pydantic import ValidationError

from genesis.models.project import Project, ProjectStatus, ProjectCreate, ProjectUpdate


# ── ProjectStatus enum ──────────────────────────────────────────────────────

class TestProjectStatus:
    def test_enum_values(self):
        assert ProjectStatus.ACTIVE.value == "active"
        assert ProjectStatus.BUILDING.value == "building"
        assert ProjectStatus.DEPLOYED.value == "deployed"
        assert ProjectStatus.FAILED.value == "failed"

    def test_enum_membership(self):
        assert len(ProjectStatus) == 4
        values = {s.value for s in ProjectStatus}
        assert values == {"active", "building", "deployed", "failed"}

    def test_enum_from_string(self):
        assert ProjectStatus("active") == ProjectStatus.ACTIVE
        assert ProjectStatus("building") == ProjectStatus.BUILDING
        assert ProjectStatus("deployed") == ProjectStatus.DEPLOYED
        assert ProjectStatus("failed") == ProjectStatus.FAILED

    def test_enum_from_string_invalid_raises(self):
        with pytest.raises(ValueError):
            ProjectStatus("invalid")


# ── Project model ───────────────────────────────────────────────────────────

class TestProject:
    def test_creation_with_minimal_fields(self):
        p = Project(name="minimal", description="a minimal project")
        assert isinstance(p.id, str)
        assert uuid.UUID(p.id)  # valid UUID4
        assert p.name == "minimal"
        assert p.description == "a minimal project"

    def test_defaults(self):
        p = Project(name="test", description="desc")
        assert p.status == ProjectStatus.ACTIVE
        assert p.build_count == 0
        assert p.last_build_id is None
        assert isinstance(p.created_at, datetime)
        assert isinstance(p.updated_at, datetime)
        assert p.created_at.tzinfo is not None
        assert p.updated_at.tzinfo is not None

    def test_full_creation(self):
        p = Project(
            name="full",
            description="fully specified",
            status=ProjectStatus.BUILDING,
            build_count=5,
            last_build_id="build-123",
        )
        assert p.name == "full"
        assert p.status == ProjectStatus.BUILDING
        assert p.build_count == 5
        assert p.last_build_id == "build-123"

    def test_explicit_id(self):
        p = Project(id="my-custom-id", name="x", description="y")
        assert p.id == "my-custom-id"

    def test_model_dump_includes_all_fields(self):
        p = Project(name="serial", description="test serialization")
        d = p.model_dump()
        assert d["name"] == "serial"
        assert d["description"] == "test serialization"
        assert d["status"] == "active"
        assert d["build_count"] == 0
        assert d["last_build_id"] is None
        assert "id" in d
        assert "created_at" in d
        assert "updated_at" in d

    def test_unique_ids(self):
        p1 = Project(name="a", description="a")
        p2 = Project(name="b", description="b")
        assert p1.id != p2.id

    def test_status_default_is_active(self):
        p = Project(name="default-status", description="")
        assert p.status == ProjectStatus.ACTIVE

    def test_accepts_string_status(self):
        p = Project(name="s", description="d", status="deployed")
        assert p.status == ProjectStatus.DEPLOYED


# ── ProjectCreate validation ────────────────────────────────────────────────

class TestProjectCreate:
    def test_minimal(self):
        pc = ProjectCreate(name="my-project")
        assert pc.name == "my-project"
        assert pc.description == ""

    def test_with_description(self):
        pc = ProjectCreate(name="test", description="some description")
        assert pc.name == "test"
        assert pc.description == "some description"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            ProjectCreate(name="")

    def test_name_too_long_raises(self):
        with pytest.raises(ValidationError):
            ProjectCreate(name="a" * 201)

    def test_name_max_length(self):
        pc = ProjectCreate(name="a" * 200)
        assert len(pc.name) == 200

    def test_description_default_is_empty_string(self):
        pc = ProjectCreate(name="test")
        assert pc.description == ""

    def test_description_too_long_raises(self):
        with pytest.raises(ValidationError):
            ProjectCreate(name="test", description="a" * 2001)

    def test_description_max_length(self):
        pc = ProjectCreate(name="test", description="a" * 2000)
        assert len(pc.description) == 2000


# ── ProjectUpdate validation ────────────────────────────────────────────────

class TestProjectUpdate:
    def test_empty_update(self):
        pu = ProjectUpdate()
        assert pu.name is None
        assert pu.description is None

    def test_partial_name_only(self):
        pu = ProjectUpdate(name="new-name")
        assert pu.name == "new-name"
        assert pu.description is None

    def test_partial_description_only(self):
        pu = ProjectUpdate(description="new desc")
        assert pu.description == "new desc"
        assert pu.name is None

    def test_full_update(self):
        pu = ProjectUpdate(name="updated", description="updated desc")
        assert pu.name == "updated"
        assert pu.description == "updated desc"

    def test_name_empty_raises(self):
        with pytest.raises(ValidationError):
            ProjectUpdate(name="")

    def test_name_too_long_raises(self):
        with pytest.raises(ValidationError):
            ProjectUpdate(name="a" * 201)

    def test_description_too_long_raises(self):
        with pytest.raises(ValidationError):
            ProjectUpdate(description="a" * 2001)
