# GENESIS ENGINE — Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement task-by-task.
> **Based on:** DESIGN.md (approved 2026-05-28)
> **Repo:** `~/genesis-engine/`

**Goal:** Build a meta-agent factory that takes a problem description and returns a deployed, tested multi-agent system.

**Architecture:** Async Python (FastAPI) pipeline with 5 stages (ANALYZE → ARCHITECT → BUILD → TEST → DEPLOY), SQLite storage, pluggable LLM providers, and an AgentSystem deployment adapter. CLI-first v0.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, Pydantic, httpx, Rich, pytest, SQLite

---

## Phase 1: Project Scaffold & Core Models

### Task 1: Create project structure and dependencies

**Objective:** Initialize the project skeleton with package structure and dev tooling.

**Files:**
- Create: `~/genesis-engine/genesis/__init__.py`
- Create: `~/genesis-engine/requirements.txt`
- Create: `~/genesis-engine/pyproject.toml`
- Create: `~/genesis-engine/tests/__init__.py`
- Create: `~/genesis-engine/tests/conftest.py`

**Step 1: Create directory structure**
```bash
mkdir -p ~/genesis-engine/genesis/{api,orchestrator,pipeline,adapters,models,storage,llm,cli}
mkdir -p ~/genesis-engine/tests/{test_api,test_pipeline,test_adapters,test_models}
```

**Step 2: Write pyproject.toml**
```toml
[project]
name = "genesis-engine"
version = "0.1.0"
description = "Meta-agent factory — AI that builds AI"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "httpx>=0.27.0",
    "openai>=1.30.0",
    "anthropic>=0.30.0",
    "rich>=13.0.0",
    "typer>=0.12.0",
    "cryptography>=42.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
]
```

**Step 3: Write requirements.txt**
```
fastapi>=0.115.0
uvicorn>=0.30.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
httpx>=0.27.0
openai>=1.30.0
anthropic>=0.30.0
rich>=13.0.0
typer>=0.12.0
cryptography>=42.0.0
```

**Step 4: Write conftest.py**
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    """In-memory SQLite session for tests."""
    engine = create_engine("sqlite:///:memory:")
    from genesis.storage.database import Base
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
```

**Step 5: Install dependencies**
```bash
cd ~/genesis-engine
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]" --break-system-packages
```

**Step 6: Verify**
```bash
python -c "import genesis; print('Genesis Engine ready')"
```

**Step 7: Commit**
```bash
cd ~/genesis-engine
git init
git add -A
git commit -m "chore: initialize genesis-engine project scaffold"
```

---

### Task 2: Core Pydantic models — Project, Build, Agent

**Objective:** Define all data models from DESIGN.md as Pydantic models with validation.

**Files:**
- Create: `genesis/models/__init__.py`
- Create: `genesis/models/project.py`
- Create: `genesis/models/build.py`
- Create: `genesis/models/agent.py`
- Create: `genesis/models/test_results.py`
- Create: `tests/test_models/test_project.py`
- Create: `tests/test_models/test_build.py`
- Create: `tests/test_models/test_agent.py`

**Step 1: Write failing test for Project model**
```python
# tests/test_models/test_project.py
import pytest
from genesis.models.project import Project, ProjectStatus


def test_project_creation_with_required_fields():
    project = Project(
        name="test-project",
        description="A test project"
    )
    assert project.name == "test-project"
    assert project.status == ProjectStatus.ACTIVE
    assert project.build_count == 0


def test_project_status_enum():
    assert ProjectStatus.ACTIVE == "active"
    assert ProjectStatus.BUILDING == "building"
    assert ProjectStatus.DEPLOYED == "deployed"
    assert ProjectStatus.FAILED == "failed"
```

**Step 2: Run test to verify failure**
```bash
cd ~/genesis-engine && source venv/bin/activate
python -m pytest tests/test_models/test_project.py -v
# Expected: FAIL — ModuleNotFoundError
```

**Step 3: Write Project model**
```python
# genesis/models/project.py
from enum import Enum
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    BUILDING = "building"
    DEPLOYED = "deployed"
    FAILED = "failed"


class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    status: ProjectStatus = ProjectStatus.ACTIVE
    build_count: int = 0
    last_build_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

**Step 4: Run test to verify pass**
```bash
python -m pytest tests/test_models/test_project.py -v
# Expected: PASS — 2 passed
```

**Step 5: Write Build, Agent, TestResults models (TDD same pattern)**
```python
# genesis/models/build.py
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    status: BuildStatus = BuildStatus.QUEUED
    stage: Optional[PipelineStage] = None
    stage_progress: float = 0.0
    problem_description: str
    target: str = "agentsystem"
    target_config: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
```

```python
# genesis/models/agent.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    name: str
    description: str
    schema: Dict[str, Any] = Field(default_factory=dict)


class SkillFile(BaseModel):
    name: str
    content: str


class CoordinationConfig(BaseModel):
    handoff_format: str = "json"
    shared_context: List[str] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    name: str
    role: str
    system_prompt: str
    tools: List[ToolConfig] = Field(default_factory=list)
    skills: List[SkillFile] = Field(default_factory=list)
    coordination_rules: CoordinationConfig = Field(default_factory=CoordinationConfig)
    config_yaml: str = ""


class DomainModel(BaseModel):
    domain: str
    actors: List[str] = Field(default_factory=list)
    intents: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    edge_cases: List[str] = Field(default_factory=list)


class AgentArchitecture(BaseModel):
    topology: str
    agents: List[Dict[str, Any]] = Field(default_factory=list)
    routing: Dict[str, Any] = Field(default_factory=dict)
```

```python
# genesis/models/test_results.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TestFailure(BaseModel):
    scenario: str
    agent: str
    expected: str
    actual: str
    metric: str


class TestResults(BaseModel):
    scenarios_run: int = 0
    scenarios_passed: int = 0
    overall_score: float = 0.0
    metrics: Dict[str, float] = Field(default_factory=dict)
    failures: List[TestFailure] = Field(default_factory=list)
    passed: bool = False
```

**Step 6: Write tests for all models, then commit**
```bash
python -m pytest tests/test_models/ -v
# Expected: ALL PASS
git add -A
git commit -m "feat: add core Pydantic models — Project, Build, Agent, TestResults"
```

---

## Phase 2: Storage Layer

### Task 3: SQLite database setup and repository

**Objective:** SQLAlchemy ORM models and repository pattern for CRUD operations.

**Files:**
- Create: `genesis/storage/__init__.py`
- Create: `genesis/storage/database.py`
- Create: `genesis/storage/repository.py`
- Create: `tests/test_storage/__init__.py`
- Create: `tests/test_storage/test_repository.py`

**Step 1: Write failing test**
```python
# tests/test_storage/test_repository.py
import pytest
from genesis.storage.repository import ProjectRepository
from genesis.models.project import Project, ProjectStatus


@pytest.mark.asyncio
async def test_create_and_get_project(db_session):
    repo = ProjectRepository(db_session)
    project = Project(name="test", description="test desc")
    
    created = await repo.create(project)
    assert created.id is not None
    
    retrieved = await repo.get(created.id)
    assert retrieved.name == "test"
    assert retrieved.status == ProjectStatus.ACTIVE


@pytest.mark.asyncio
async def test_list_projects(db_session):
    repo = ProjectRepository(db_session)
    await repo.create(Project(name="a", description="first"))
    await repo.create(Project(name="b", description="second"))
    
    projects = await repo.list_all()
    assert len(projects) == 2
```

**Step 2: Implement database.py**
```python
# genesis/storage/database.py
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DATABASE_URL = os.getenv("GENESIS_DB", "sqlite:///genesis.db")
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class ProjectRecord(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, default="active")
    build_count = Column(Integer, default=0)
    last_build_id = Column(String, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class BuildRecord(Base):
    __tablename__ = "builds"
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
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


def init_db():
    Base.metadata.create_all(engine)
```

**Step 3: Implement repository.py**
```python
# genesis/storage/repository.py
from typing import Optional, List
from sqlalchemy.orm import Session
from genesis.storage.database import ProjectRecord, BuildRecord
from genesis.models.project import Project
from genesis.models.build import Build


class ProjectRepository:
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
        records = self.session.query(ProjectRecord).all()
        return [self._to_model(r) for r in records]

    async def update(self, project: Project) -> Project:
        record = self.session.get(ProjectRecord, project.id)
        if not record:
            raise ValueError(f"Project {project.id} not found")
        record.name = project.name
        record.description = project.description
        record.status = project.status.value
        record.build_count = project.build_count
        record.last_build_id = project.last_build_id
        record.updated_at = project.updated_at
        self.session.commit()
        return project

    async def delete(self, project_id: str) -> bool:
        record = self.session.get(ProjectRecord, project_id)
        if not record:
            return False
        self.session.delete(record)
        self.session.commit()
        return True

    def _to_model(self, record: ProjectRecord) -> Project:
        return Project(
            id=record.id,
            name=record.name,
            description=record.description,
            status=record.status,
            build_count=record.build_count,
            last_build_id=record.last_build_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
```

**Step 4: Run tests, commit**
```bash
python -m pytest tests/test_storage/ -v
git add -A && git commit -m "feat: add SQLite storage layer with project repository"
```

---

## Phase 3: LLM Provider Abstraction

### Task 4: Abstract LLM provider with OpenAI implementation

**Objective:** Provider interface that supports multiple LLM backends. OpenAI first.

**Files:**
- Create: `genesis/llm/__init__.py`
- Create: `genesis/llm/provider.py`
- Create: `genesis/llm/openai.py`
- Create: `tests/test_llm/__init__.py`
- Create: `tests/test_llm/test_provider.py`

**Step 1: Write abstract provider**
```python
# genesis/llm/provider.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Send a completion request and return structured response."""
        ...
```

**Step 2: Write OpenAI implementation (with mock-based tests)**
```python
# genesis/llm/openai.py
import os
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
from genesis.llm.provider import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        kwargs = {
            "model": model or "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )
```

**Step 3: Write tests with mocking, verify, commit**

---

## Phase 4: Pipeline Stages

### Task 5: Stage 1 — ANALYZE (Domain Analysis)

**Objective:** Takes a problem description, returns a structured DomainModel.

**Files:**
- Create: `genesis/pipeline/__init__.py`
- Create: `genesis/pipeline/analyze.py`
- Create: `tests/test_pipeline/__init__.py`
- Create: `tests/test_pipeline/test_analyze.py`

**Implementation:**
```python
# genesis/pipeline/analyze.py
import json
from genesis.llm.provider import LLMProvider
from genesis.models.agent import DomainModel

ANALYZE_SYSTEM_PROMPT = """You are a domain analyst. Given a problem description,
decompose it into a structured domain model. Output valid JSON only.

Return format:
{
  "domain": "short domain name",
  "actors": ["list of actors/roles involved"],
  "intents": [{"actor": "actor_name", "intent": "what they want", "priority": "high|medium|low"}],
  "constraints": ["list of constraints or requirements"],
  "edge_cases": ["list of edge cases to handle"],
  "success_criteria": ["how to know if the system is working"]
}"""


class AnalyzeStage:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(self, problem_description: str) -> DomainModel:
        response = await self.llm.complete(
            system_prompt=ANALYZE_SYSTEM_PROMPT,
            user_prompt=problem_description,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.content)
        return DomainModel(**data)
```

**TDD:**
1. Write test with mock LLM returning sample JSON
2. Verify DomainModel is correctly parsed
3. Test error handling for invalid JSON (retry logic)
4. Commit

### Task 6: Stage 2 — ARCHITECT (Agent Topology)

Creates `AgentArchitecture` from `DomainModel`.

**Files:** `genesis/pipeline/architect.py`, `tests/test_pipeline/test_architect.py`

### Task 7: Stage 3 — BUILD (Prompt & Tool Generation)

Creates `List[AgentDefinition]` from `AgentArchitecture`.

**Files:** `genesis/pipeline/build.py`, `tests/test_pipeline/test_build.py`

### Task 8: Stage 4 — TEST (Simulation & Evaluation)

Runs simulated scenarios, scores agents, produces `TestResults`.

**Files:** `genesis/pipeline/test.py`, `tests/test_pipeline/test_test.py`

### Task 9: Stage 5 — DEPLOY (Target Deployment)

Calls the target adapter to provision agents.

**Files:** `genesis/pipeline/deploy.py`, `tests/test_pipeline/test_deploy.py`

---

## Phase 5: Orchestrator & State Machine

### Task 10: Build orchestrator with state machine

**Objective:** Manages the 5-stage pipeline lifecycle with retries.

**Files:**
- Create: `genesis/orchestrator/__init__.py`
- Create: `genesis/orchestrator/state_machine.py`
- Create: `tests/test_orchestrator/test_state_machine.py`

**Implementation outline:**
```python
# genesis/orchestrator/state_machine.py
from enum import Enum
from genesis.models.build import Build, BuildStatus, PipelineStage
from genesis.pipeline.analyze import AnalyzeStage
from genesis.pipeline.architect import ArchitectStage
from genesis.pipeline.build import BuildStage
from genesis.pipeline.test import TestStage
from genesis.pipeline.deploy import DeployStage


class Orchestrator:
    MAX_RETRIES = 3
    TEST_THRESHOLD = 0.80

    def __init__(self, llm, storage, deployment_target):
        self.analyze = AnalyzeStage(llm)
        self.architect = ArchitectStage(llm)
        self.build = BuildStage(llm)
        self.test = TestStage(llm)
        self.deploy = DeployStage(deployment_target)
        self.storage = storage

    async def run_pipeline(self, build: Build) -> Build:
        try:
            # Stage 1: ANALYZE
            build = self._transition(build, BuildStatus.ANALYZING)
            domain_model = await self.analyze.run(build.problem_description)
            
            # Stage 2: ARCHITECT
            build = self._transition(build, BuildStatus.ARCHITECTING)
            architecture = await self.architect.run(domain_model)
            
            # Stage 3: BUILD
            build = self._transition(build, BuildStatus.BUILDING)
            agents = await self.build.run(architecture)
            
            # Stage 4-5: TEST → BUILD loop
            while build.retries < self.MAX_RETRIES:
                build = self._transition(build, BuildStatus.TESTING)
                test_results = await self.test.run(agents)
                
                if test_results.overall_score >= self.TEST_THRESHOLD:
                    break
                
                build.retries += 1
                # Feed test failures back to BUILD
                agents = await self.build.run(architecture, feedback=test_results)
            
            # Stage 5: DEPLOY
            build = self._transition(build, BuildStatus.DEPLOYING)
            deployment = await self.deploy.run(agents, build.target_config)
            
            build.status = BuildStatus.COMPLETED
            build.artifacts = {
                "domain_model": domain_model.model_dump(),
                "architecture": architecture.model_dump(),
                "agents": [a.model_dump() for a in agents],
                "deployment": deployment.model_dump(),
            }
            build.test_results = test_results.model_dump()
            
        except Exception as e:
            build.status = BuildStatus.FAILED
            build.error = str(e)
        
        return build
```

---

## Phase 6: API Layer

### Task 11: FastAPI routes

**Objective:** Wire the orchestrator to REST endpoints.

**Files:**
- Create: `genesis/api/__init__.py`
- Create: `genesis/api/routes.py`
- Create: `genesis/api/dependencies.py`
- Create: `genesis/api/app.py`
- Create: `tests/test_api/test_routes.py`

### Task 12: SSE streaming logs endpoint

**Objective:** `GET /v1/builds/{id}/logs` streams stage transitions as Server-Sent Events.

---

## Phase 7: AgentSystem Adapter

### Task 13: AgentSystem deployment adapter

**Objective:** Translate Genesis agent definitions into AgentSystem API calls.

**Files:**
- Create: `genesis/adapters/__init__.py`
- Create: `genesis/adapters/base.py`
- Create: `genesis/adapters/agentsystem.py`
- Create: `tests/test_adapters/test_agentsystem.py`

---

## Phase 8: CLI

### Task 14: CLI with Rich output

**Objective:** Beautiful terminal interface using Typer + Rich.

**Files:**
- Create: `genesis/cli/__init__.py`
- Create: `genesis/cli/main.py`
- Create: `genesis/cli/display.py`

**Key commands:**
```bash
genesis build "problem description"    # One-shot pipeline
genesis project create "name"          # Create project
genesis project list                   # List projects
genesis project status <id>            # Check status
genesis logs <build_id> --follow       # Stream build logs
```

**Rich display during build:**
```
  ⏳ ANALYZE    Decomposing problem domain...      ✓ (2.1s)
  ⏳ ARCHITECT  Designing agent topology...         ✓ (4.3s)
  ⏳ BUILD      Generating prompts and tools...     ✓ (8.7s)
  ⏳ TEST       Running simulations...              ✓ (12.4s)
  ⏳ DEPLOY     Provisioning on AgentSystem...       ✓ (3.1s)

  🚀 System deployed!
```

---

## Phase 9: Python SDK

### Task 15: Python SDK package

**Objective:** Clean Python SDK wrapping the REST API.

**Files:**
- Create: `genesis/sdk/__init__.py`
- Create: `genesis/sdk/client.py`

```python
# Usage:
import genesis
system = genesis.generate("Build a support system for my SaaS")
```

---

## Phase 10: Integration & Polish

### Task 16: End-to-end integration test

**Objective:** Full pipeline test with mock LLM and mock AgentSystem.

### Task 17: README and documentation

**Objective:** Usage docs, API reference, example walkthrough.

### Task 18: Docker support

**Objective:** Dockerfile for containerized deployment.

---

## Verification Checklist

- [ ] All model validation tests pass
- [ ] Storage CRUD operations pass
- [ ] LLM provider returns correct format
- [ ] Each pipeline stage produces correct output type
- [ ] Orchestrator transitions through all stages in order
- [ ] TEST→BUILD retry loop works correctly (max 3)
- [ ] API returns correct status codes and response bodies
- [ ] SSE logs stream stage transitions
- [ ] AgentSystem adapter provisions agents correctly
- [ ] CLI displays Rich progress during build
- [ ] SDK one-shot call returns deployed system URL
- [ ] Integration test passes end-to-end (mock LLM)

---

*End of IMPLEMENTATION_PLAN.md — 18 tasks, ~4-6 hours of focused implementation.*
