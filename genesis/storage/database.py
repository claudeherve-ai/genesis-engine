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
    parent_build_id = Column(String, nullable=True, index=True)
    feedback_seed = Column(Text, nullable=True)
    created_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)


class KnowledgeDocumentRecord(Base):
    __tablename__ = "knowledge_documents"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False, default="")
    text = Column(Text, nullable=False)
    source = Column(String, nullable=True)
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime)


def init_db() -> None:
    """Create all tables if they don't exist, then apply lightweight migrations.

    SQLAlchemy's ``create_all`` will not add new columns to a table that
    already exists in an older ``genesis.db``. We additively reconcile a small
    set of known columns so upgrades are seamless and idempotent.
    """
    Base.metadata.create_all(engine)
    _apply_additive_migrations()


# Columns added after initial release, by table. Each is applied with
# ``ALTER TABLE ... ADD COLUMN`` only when missing (idempotent, additive-only).
_ADDITIVE_COLUMNS = {
    "builds": {
        "parent_build_id": "VARCHAR",
        "feedback_seed": "TEXT",
    },
}


def _apply_additive_migrations() -> None:
    if "sqlite" not in DATABASE_URL:
        return
    from sqlalchemy import inspect as _inspect, text as _text

    inspector = _inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in _ADDITIVE_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table)}
            for col_name, col_type in columns.items():
                if col_name not in present:
                    conn.execute(
                        _text(f'ALTER TABLE {table} ADD COLUMN {col_name} {col_type}')
                    )


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()


# Auto-init on import for simplicity
init_db()
