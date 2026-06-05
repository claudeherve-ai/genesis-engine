"""Repository pattern for CRUD operations."""

from typing import Optional, List
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.orm import Session
from genesis.storage.database import ProjectRecord, BuildRecord
from genesis.models.project import Project
from genesis.models.build import Build


class ProjectRepository:
    """CRUD operations for projects."""

    def __init__(self, session: Session):
        self.session = session

    async def create(self, project: Project) -> Project:
        record = ProjectRecord(
            id=project.id,
            name=project.name,
            description=project.description,
            status=project.status.value,
            build_count=project.build_count,
            last_build_id=project.last_build_id,
            active_build_id=project.active_build_id,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )
        self.session.add(record)
        self.session.commit()
        return project

    async def get(self, project_id: str) -> Optional[Project]:
        record = self.session.get(ProjectRecord, project_id)
        if not record:
            return None
        return self._to_model(record)

    async def list_all(self) -> List[Project]:
        records = (
            self.session.query(ProjectRecord)
            .order_by(ProjectRecord.updated_at.desc(), text("rowid DESC"))
            .all()
        )
        return [self._to_model(r) for r in records]

    async def update(self, project: Project) -> Project:
        record = self.session.get(ProjectRecord, project.id)
        if not record:
            raise ValueError(f"Project {project.id} not found")
        record.name = project.name
        record.description = project.description
        record.status = getattr(project.status, "value", project.status)
        record.build_count = project.build_count
        record.last_build_id = project.last_build_id
        record.active_build_id = project.active_build_id
        record.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        project.updated_at = record.updated_at
        return project

    async def delete(self, project_id: str) -> bool:
        record = self.session.get(ProjectRecord, project_id)
        if not record:
            return False
        self.session.delete(record)
        self.session.commit()
        return True

    @staticmethod
    def _to_model(record: ProjectRecord) -> Project:
        return Project(
            id=record.id,
            name=record.name,
            description=record.description,
            status=record.status,
            build_count=record.build_count,
            last_build_id=record.last_build_id,
            active_build_id=getattr(record, "active_build_id", None),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class BuildRepository:
    """CRUD operations for builds."""

    def __init__(self, session: Session):
        self.session = session

    async def create(self, build: Build) -> Build:
        record = BuildRecord(
            id=build.id,
            project_id=build.project_id,
            status=build.status.value,
            stage=build.stage.value if build.stage else None,
            stage_progress=build.stage_progress,
            problem_description=build.problem_description,
            target=build.target,
            target_config=build.target_config,
            artifacts=build.artifacts,
            test_results=build.test_results,
            error=build.error,
            retries=build.retries,
            parent_build_id=build.parent_build_id,
            feedback_seed=build.feedback_seed,
            created_at=build.created_at,
            completed_at=build.completed_at,
        )
        self.session.add(record)
        self.session.commit()
        return build

    async def get(self, build_id: str) -> Optional[Build]:
        record = self.session.get(BuildRecord, build_id)
        if not record:
            return None
        return self._to_model(record)

    async def list_by_project(self, project_id: str) -> List[Build]:
        records = (
            self.session.query(BuildRecord)
            .filter(BuildRecord.project_id == project_id)
            .order_by(BuildRecord.created_at.desc(), text("rowid DESC"))
            .all()
        )
        return [self._to_model(r) for r in records]

    async def update(self, build: Build) -> Build:
        record = self.session.get(BuildRecord, build.id)
        if not record:
            raise ValueError(f"Build {build.id} not found")
        record.status = build.status.value
        record.stage = build.stage.value if build.stage else None
        record.stage_progress = build.stage_progress
        record.artifacts = build.artifacts
        record.test_results = build.test_results
        record.error = build.error
        record.retries = build.retries
        record.completed_at = build.completed_at
        self.session.commit()
        return build

    @staticmethod
    def _to_model(record: BuildRecord) -> Build:
        return Build(
            id=record.id,
            project_id=record.project_id,
            status=record.status,
            stage=record.stage,
            stage_progress=record.stage_progress,
            problem_description=record.problem_description,
            target=record.target,
            target_config=record.target_config or {},
            artifacts=record.artifacts,
            test_results=record.test_results,
            error=record.error,
            retries=record.retries or 0,
            parent_build_id=getattr(record, "parent_build_id", None),
            feedback_seed=getattr(record, "feedback_seed", None),
            created_at=record.created_at,
            completed_at=record.completed_at,
        )
