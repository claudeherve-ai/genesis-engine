"""Tests for Build model, enums, and request schemas."""

import pytest
import uuid
from datetime import datetime, timezone
from pydantic import ValidationError

from genesis.models.build import (
    Build,
    BuildStatus,
    PipelineStage,
    BuildRequest,
    GenerateRequest,
)


# ── BuildStatus enum ────────────────────────────────────────────────────────

class TestBuildStatus:
    def test_enum_values(self):
        assert BuildStatus.QUEUED.value == "queued"
        assert BuildStatus.ANALYZING.value == "analyzing"
        assert BuildStatus.ARCHITECTING.value == "architecting"
        assert BuildStatus.BUILDING.value == "building"
        assert BuildStatus.TESTING.value == "testing"
        assert BuildStatus.DEPLOYING.value == "deploying"
        assert BuildStatus.COMPLETED.value == "completed"
        assert BuildStatus.FAILED.value == "failed"

    def test_enum_membership(self):
        assert len(BuildStatus) == 8
        values = {s.value for s in BuildStatus}
        assert values == {
            "queued", "analyzing", "architecting", "building",
            "testing", "deploying", "completed", "failed",
        }

    def test_enum_from_string(self):
        for val in ["queued", "analyzing", "architecting", "building",
                     "testing", "deploying", "completed", "failed"]:
            assert BuildStatus(val).value == val

    def test_enum_from_string_invalid_raises(self):
        with pytest.raises(ValueError):
            BuildStatus("nope")


# ── PipelineStage enum ──────────────────────────────────────────────────────

class TestPipelineStage:
    def test_enum_values(self):
        assert PipelineStage.ANALYZE.value == "analyze"
        assert PipelineStage.ARCHITECT.value == "architect"
        assert PipelineStage.BUILD.value == "build"
        assert PipelineStage.TEST.value == "test"
        assert PipelineStage.DEPLOY.value == "deploy"

    def test_enum_membership(self):
        assert len(PipelineStage) == 5
        values = {s.value for s in PipelineStage}
        assert values == {"analyze", "architect", "build", "test", "deploy"}

    def test_enum_from_string(self):
        assert PipelineStage("analyze") == PipelineStage.ANALYZE
        assert PipelineStage("architect") == PipelineStage.ARCHITECT
        assert PipelineStage("build") == PipelineStage.BUILD
        assert PipelineStage("test") == PipelineStage.TEST
        assert PipelineStage("deploy") == PipelineStage.DEPLOY


# ── Build model ─────────────────────────────────────────────────────────────

class TestBuild:
    def test_creation_with_minimal_fields(self):
        b = Build(project_id="p-1", problem_description="Build something useful")
        assert isinstance(b.id, str)
        assert uuid.UUID(b.id)
        assert b.project_id == "p-1"
        assert b.problem_description == "Build something useful"

    def test_defaults(self):
        b = Build(project_id="p-1", problem_description="desc")
        assert b.status == BuildStatus.QUEUED
        assert b.stage is None
        assert b.stage_progress == 0.0
        assert b.target == "agentsystem"
        assert b.target_config == {}
        assert b.artifacts is None
        assert b.test_results is None
        assert b.error is None
        assert b.retries == 0
        assert isinstance(b.created_at, datetime)
        assert b.created_at.tzinfo is not None
        assert b.completed_at is None

    def test_full_creation(self):
        now = datetime.now(timezone.utc)
        b = Build(
            project_id="p-2",
            status=BuildStatus.BUILDING,
            stage=PipelineStage.BUILD,
            stage_progress=0.5,
            problem_description="Full build test",
            target="custom-target",
            target_config={"key": "value"},
            artifacts={"output": "data"},
            test_results={"passed": 10, "failed": 2},
            error=None,
            retries=2,
            completed_at=now,
        )
        assert b.status == BuildStatus.BUILDING
        assert b.stage == PipelineStage.BUILD
        assert b.stage_progress == 0.5
        assert b.target == "custom-target"
        assert b.target_config == {"key": "value"}
        assert b.artifacts == {"output": "data"}
        assert b.test_results == {"passed": 10, "failed": 2}
        assert b.retries == 2
        assert b.completed_at == now

    def test_is_terminal_completed(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.COMPLETED)
        assert b.is_terminal is True

    def test_is_terminal_failed(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.FAILED)
        assert b.is_terminal is True

    def test_is_terminal_queued(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.QUEUED)
        assert b.is_terminal is False

    def test_is_terminal_building(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.BUILDING)
        assert b.is_terminal is False

    def test_is_running_queued_false(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.QUEUED)
        assert b.is_running is False

    def test_is_running_completed_false(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.COMPLETED)
        assert b.is_running is False

    def test_is_running_failed_false(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.FAILED)
        assert b.is_running is False

    def test_is_running_analyzing_true(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.ANALYZING)
        assert b.is_running is True

    def test_is_running_architecting_true(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.ARCHITECTING)
        assert b.is_running is True

    def test_is_running_building_true(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.BUILDING)
        assert b.is_running is True

    def test_is_running_testing_true(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.TESTING)
        assert b.is_running is True

    def test_is_running_deploying_true(self):
        b = Build(project_id="p", problem_description="x",
                  status=BuildStatus.DEPLOYING)
        assert b.is_running is True

    def test_stage_progress_boundaries(self):
        # ge=0, le=1 enforced by Pydantic
        Build(project_id="p", problem_description="x", stage_progress=0.0)
        Build(project_id="p", problem_description="x", stage_progress=1.0)
        with pytest.raises(ValidationError):
            Build(project_id="p", problem_description="x", stage_progress=-0.1)
        with pytest.raises(ValidationError):
            Build(project_id="p", problem_description="x", stage_progress=1.1)

    def test_retries_non_negative(self):
        with pytest.raises(ValidationError):
            Build(project_id="p", problem_description="x", retries=-1)

    def test_target_config_default_is_dict(self):
        b = Build(project_id="p", problem_description="x")
        assert isinstance(b.target_config, dict)
        assert b.target_config == {}

    def test_model_dump(self):
        b = Build(project_id="p-3", problem_description="dump test")
        d = b.model_dump()
        assert d["project_id"] == "p-3"
        assert d["status"] == "queued"
        assert d["stage"] is None
        assert d["stage_progress"] == 0.0
        assert d["target_config"] == {}
        assert "id" in d
        assert "created_at" in d

    def test_unique_ids(self):
        b1 = Build(project_id="p", problem_description="a")
        b2 = Build(project_id="p", problem_description="b")
        assert b1.id != b2.id

    def test_stage_none_by_default(self):
        b = Build(project_id="p", problem_description="x")
        assert b.stage is None

    def test_artifacts_none_by_default(self):
        b = Build(project_id="p", problem_description="x")
        assert b.artifacts is None

    def test_test_results_none_by_default(self):
        b = Build(project_id="p", problem_description="x")
        assert b.test_results is None


# ── BuildRequest validation ─────────────────────────────────────────────────

class TestBuildRequest:
    def test_minimal(self):
        br = BuildRequest(problem_description="Build a chatbot for customer service")
        assert br.problem_description == "Build a chatbot for customer service"
        assert br.target == "agentsystem"
        assert br.target_config == {}

    def test_full(self):
        br = BuildRequest(
            problem_description="x" * 100,
            target="custom-backend",
            target_config={"provider": "openai"},
        )
        assert br.target == "custom-backend"
        assert br.target_config == {"provider": "openai"}

    def test_problem_too_short_raises(self):
        with pytest.raises(ValidationError):
            BuildRequest(problem_description="short")

    def test_problem_min_length(self):
        br = BuildRequest(problem_description="a" * 10)
        assert len(br.problem_description) == 10

    def test_problem_too_long_raises(self):
        with pytest.raises(ValidationError):
            BuildRequest(problem_description="a" * 10001)

    def test_problem_max_length(self):
        br = BuildRequest(problem_description="a" * 10000)
        assert len(br.problem_description) == 10000

    def test_target_config_default(self):
        br = BuildRequest(problem_description="x" * 10)
        assert br.target_config == {}


# ── GenerateRequest validation ──────────────────────────────────────────────

class TestGenerateRequest:
    def test_minimal(self):
        gr = GenerateRequest(problem="Describe a problem to solve with agents")
        assert gr.problem == "Describe a problem to solve with agents"
        assert gr.target == "agentsystem"
        assert gr.target_config == {}
        assert gr.project_name is None

    def test_full(self):
        gr = GenerateRequest(
            problem="x" * 100,
            target="custom",
            target_config={"model": "gpt-4"},
            project_name="my-genesis-project",
        )
        assert gr.target == "custom"
        assert gr.target_config == {"model": "gpt-4"}
        assert gr.project_name == "my-genesis-project"

    def test_problem_too_short_raises(self):
        with pytest.raises(ValidationError):
            GenerateRequest(problem="short")

    def test_problem_min_length(self):
        gr = GenerateRequest(problem="a" * 10)
        assert len(gr.problem) == 10

    def test_problem_too_long_raises(self):
        with pytest.raises(ValidationError):
            GenerateRequest(problem="a" * 10001)

    def test_problem_max_length(self):
        gr = GenerateRequest(problem="a" * 10000)
        assert len(gr.problem) == 10000

    def test_project_name_optional(self):
        gr = GenerateRequest(problem="x" * 10)
        assert gr.project_name is None

    def test_default_target(self):
        gr = GenerateRequest(problem="x" * 10)
        assert gr.target == "agentsystem"
