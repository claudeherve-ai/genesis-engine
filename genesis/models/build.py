"""Build model — represents a pipeline execution."""

from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class BuildStatus(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    ARCHITECTING = "architecting"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    ANALYZE = "analyze"
    ARCHITECT = "architect"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"


class Build(BaseModel):
    """A single pipeline execution — ANALYZE → ARCHITECT → BUILD → TEST → DEPLOY."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    status: BuildStatus = BuildStatus.QUEUED
    stage: Optional[PipelineStage] = None
    stage_progress: float = Field(0.0, ge=0.0, le=1.0)
    problem_description: str
    target: str = "agentsystem"
    target_config: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = Field(0, ge=0)
    # Rebuild lineage: the build this one was spawned from (if any), plus the
    # compact feedback summary that seeded the rebuild. Full versioning and
    # rollback live in the platform layer; this is lineage only.
    parent_build_id: Optional[str] = None
    feedback_seed: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (BuildStatus.COMPLETED, BuildStatus.FAILED)

    @property
    def is_running(self) -> bool:
        return self.status not in (
            BuildStatus.QUEUED,
            BuildStatus.COMPLETED,
            BuildStatus.FAILED,
        )


class BuildRequest(BaseModel):
    """Request model for triggering a build."""

    problem_description: str = Field(..., min_length=10, max_length=10000)
    target: str = Field(default="agentsystem")
    target_config: Dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    """Request model for one-shot generate endpoint."""

    problem: str = Field(..., min_length=10, max_length=10000)
    target: str = Field(default="agentsystem")
    target_config: Dict[str, Any] = Field(default_factory=dict)
    project_name: Optional[str] = None
