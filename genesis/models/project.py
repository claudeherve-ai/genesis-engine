"""Project model."""

from enum import Enum
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
import uuid


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    BUILDING = "building"
    DEPLOYED = "deployed"
    FAILED = "failed"


class Project(BaseModel):
    """A Genesis project — contains builds and deployed agents."""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    status: ProjectStatus = ProjectStatus.ACTIVE
    build_count: int = 0
    last_build_id: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ProjectCreate(BaseModel):
    """Request model for creating a project."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class ProjectUpdate(BaseModel):
    """Request model for updating a project."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
