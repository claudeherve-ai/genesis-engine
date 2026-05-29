"""SQLite database setup with SQLAlchemy ORM."""

import os
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("GENESIS_DB", "sqlite:///genesis.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class ProjectRecord(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False, default="")
    status = Column(String, default="active")
    build_count = Column(Integer, default=0)
    last_build_id = Column(String, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class BuildRecord(Base):
    __tablename__ = "builds"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    status = Column(String, default="queued")
    stage = Column(String, nullable=True)
    stage_progress = Column(Float, default=0.0)
    problem_description = Column(Text, nullable=False)
    target = Column(String, default="agentsystem")
    target_config = Column(JSON, default={})
    artifacts = Column(JSON, nullable=True)
    test_results = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    retries = Column(Integer, default=0)
    created_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()


# Auto-init on import for simplicity
init_db()
