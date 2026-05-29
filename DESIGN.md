# GENESIS ENGINE — Design Document

> **Status:** Approved (CEO ✓ | Designer ✓ | EM ✓)
> **Date:** 2026-05-28
> **Author:** Claude Djale + Hermes Agent
> **Repo:** `~/genesis-engine/`

## Product Vision

**"Describe a problem → working multi-agent system in 90 seconds."**

Genesis Engine is a meta-agent factory. You give it a problem description. It designs the agent architecture, generates prompts and tools, tests the system, and deploys it to your target platform. It's infrastructure for the agent economy — the tool that builds the tools.

**Positioning:** Infrastructure, not a feature. The Stripe of multi-agent systems.

**Go-to-Market Demo:** "Build me a customer support system for a B2B SaaS." 90 seconds. Deployed, tested, running on AgentSystem. That's the mic-drop moment.

---

## Architecture

```
                          ┌──────────────────────────┐
                          │     GENESIS ENGINE        │
                          │                           │
   ┌──────────┐          │  ┌─────────────────────┐  │
   │ REST API │─────────▶│  │   Orchestrator       │  │
   │ (public) │          │  │   (job queue,        │  │
   └──────────┘          │  │    state machine)    │  │
                          │  └──────┬──────────────┘  │
   ┌──────────┐          │         │                  │
   │   CLI    │─────────▶│         ▼                  │
   └──────────┘          │  ┌─────────────────────┐  │
                          │  │   Pipeline Stages   │  │
   ┌──────────┐          │  │                     │  │
   │ Python   │─────────▶│  │ 1. ANALYZE          │  │
   │ SDK      │          │  │    Domain modeling   │  │
   └──────────┘          │  │                     │  │
                          │  │ 2. ARCHITECT        │  │
   ┌──────────┐          │  │    Agent topology    │  │
   │ Web UI   │─────────▶│  │                     │  │
   │ (v1)     │          │  │ 3. BUILD            │  │
   └──────────┘          │  │    Prompts/tools     │  │
                          │  │                     │  │
                          │  │ 4. TEST             │  │
                          │  │    Sim + evaluate   │  │
                          │  │    ↻ retry (max 3x) │  │
                          │  │                     │  │
                          │  │ 5. DEPLOY           │  │
                          │  │    Target adapter   │  │
                          │  └─────────────────────┘  │
                          │                           │
                          │  Storage: SQLite + files   │
                          └───────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.11+ (FastAPI) | Matches existing stack (AgentSystem) |
| Storage | SQLite (v0) → PostgreSQL (v1) | Zero-dependency start, easy migration |
| Async | Full async (asyncio, httpx) | Pipeline stages are I/O bound (LLM calls) |
| Job model | State machine + queue | Builds are 30-120s; must be async/pollable |
| LLM provider | Abstracted (OpenAI, Anthropic, OpenRouter) | Swap models per stage, per quality tier |
| Deployment | Adapter pattern | AgentSystem first, then Hermes, then generic |

---

## API Surface

### REST API

```
POST   /v1/projects              Create a new project
GET    /v1/projects/{id}         Get project details + status
DELETE /v1/projects/{id}         Delete a project
GET    /v1/projects              List all projects

POST   /v1/projects/{id}/build   Trigger a build pipeline
GET    /v1/projects/{id}/builds  List builds for a project
GET    /v1/builds/{id}           Build status + stage progress
GET    /v1/builds/{id}/logs      Streaming build logs (SSE)
GET    /v1/builds/{id}/artifacts Download generated agent artifacts

POST   /v1/deploy/{project_id}   Deploy to target platform
GET    /v1/deploy/{id}/status    Deployment status

GET    /v1/templates             List reusable agent templates

POST   /v1/generate             One-shot: analyze → deploy in one call
```

### The "Holy Shit" Endpoint

```json
POST /v1/generate
{
  "problem": "I need a customer support system for a B2B SaaS company.
              Triage requests, handle technical questions, billing
              issues, and escalate complex cases to humans.",
  "target": "agentsystem",
  "target_config": {
    "endpoint": "https://agentsystem.example.com",
    "api_key": "sk-..."
  }
}

→ 202 Accepted
→ Location: /v1/builds/gen_20260528_a1b2c3
→ 90 seconds later: deployed, tested, running
```

### Python SDK

```python
import genesis

# One-shot
system = genesis.generate("Build a support system for my SaaS")
print(system.url)  # https://agentsystem.example.com/deploy/support-system

# Step-by-step
project = genesis.create("Support system for Acme Corp")
build = project.build()
build.wait()
print(build.agents)  # [triage_agent, technical_agent, billing_agent, ...]
print(build.test_score)  # 0.87
```

### CLI

```bash
# One-shot
$ genesis build "I need a customer support system with triage, tech,
                 billing, and escalation agents"

  ⏳ ANALYZE    Decomposing problem domain...      ✓ (2.1s)
  ⏳ ARCHITECT  Designing agent topology...         ✓ (4.3s)
  ⏳ BUILD      Generating prompts and tools...     ✓ (8.7s)
  ⏳ TEST       Running simulations...              ✓ (12.4s)
  ⏳ DEPLOY     Provisioning on AgentSystem...       ✓ (3.1s)

  🚀 System deployed!
     URL: https://agentsystem.example.com/deploy/support-system
     Agents: triage_agent, technical_agent, billing_agent, escalation_agent
     Test score: 87/100

# Interactive
$ genesis init my-project
$ genesis status
$ genesis logs --follow
```

---

## Data Model

### Project

```python
class Project(BaseModel):
    id: str                          # UUID
    name: str
    description: str
    status: ProjectStatus            # active, building, deployed, failed
    build_count: int
    last_build_id: Optional[str]
    created_at: datetime
    updated_at: datetime
```

### Build

```python
class Build(BaseModel):
    id: str                          # UUID
    project_id: str
    status: BuildStatus              # queued, analyzing, architecting,
                                     # building, testing, deploying,
                                     # completed, failed
    stage: PipelineStage
    stage_progress: float            # 0.0 - 1.0
    problem_description: str
    target: str                      # agentsystem, hermes, generic
    target_config: Dict[str, Any]
    artifacts: Optional[BuildArtifacts]
    test_results: Optional[TestResults]
    error: Optional[str]
    retries: int                     # TEST→BUILD retry counter
    created_at: datetime
    completed_at: Optional[datetime]
```

### BuildArtifacts

```python
class BuildArtifacts(BaseModel):
    domain_model: DomainModel        # ANALYZE output
    architecture: AgentArchitecture  # ARCHITECT output
    agents: List[AgentDefinition]    # BUILD output
    deployment: Optional[DeploymentResult]  # DEPLOY output

class AgentDefinition(BaseModel):
    name: str
    role: str
    system_prompt: str
    tools: List[ToolConfig]
    skills: List[SkillFile]
    coordination_rules: CoordinationConfig
    config_yaml: str                 # Target-platform config
```

### TestResults

```python
class TestResults(BaseModel):
    scenarios_run: int
    scenarios_passed: int
    overall_score: float             # 0.0 - 1.0
    metrics: Dict[str, float]        # per-metric scores
    failures: List[TestFailure]
    passed: bool                     # score >= threshold

class TestFailure(BaseModel):
    scenario: str
    agent: str
    expected: str
    actual: str
    metric: str
```

---

## Pipeline Stages

### Stage 1: ANALYZE — Domain Understanding

**Input:** Problem description (natural language)
**Output:** Structured domain model
**LLM role:** Domain analyst

Extracts: actors, intents, information flows, constraints, edge cases, success criteria.

### Stage 2: ARCHITECT — Agent Topology Design

**Input:** Domain model
**Output:** Agent architecture (topology, agents, tools, routing)
**LLM role:** Systems architect

Designs: agent count, responsibilities, communication pattern (router/sequential/swarm), tool assignments, escalation paths.

### Stage 3: BUILD — Prompt & Tool Generation

**Input:** Agent architecture
**Output:** Runnable agent definitions
**LLM role:** Prompt engineer + tool designer

Generates for each agent: system prompt, tool configs, skill files, coordination rules, target-platform YAML.

### Stage 4: TEST — Simulation & Evaluation

**Input:** Agent definitions
**Output:** Quality score + pass/fail + specific failures
**LLM role:** QA engineer (simulates conversations, evaluates outputs)

Generates 10-20 test scenarios per agent set. Runs simulations. Evaluates: intent classification, resolution rate, handoff correctness, tool usage. Scores against configurable threshold (default: 0.80).

**Retry loop:** If score < threshold, feedback goes back to BUILD stage with specific issues. Max 3 retries.

### Stage 5: DEPLOY — Target Adapter

**Input:** Validated agent definitions
**Output:** Live, running system + endpoint URL
**Adapter:** AgentSystem (v0), Hermes, Generic REST (v1)

---

## Deployment Target Adapter Interface

```python
class DeploymentTarget(ABC):
    """Abstract interface for deployment targets."""

    @abstractmethod
    async def provision(self, agents: List[AgentDefinition]) -> DeploymentResult:
        """Provision agents on the target platform."""

    @abstractmethod
    async def health_check(self, deployment_id: str) -> bool:
        """Verify the deployment is healthy."""

    @abstractmethod
    async def teardown(self, deployment_id: str) -> None:
        """Remove the deployment."""

@dataclass
class DeploymentResult:
    deployment_id: str
    endpoint_url: str
    status: str
    agent_count: int
```

### AgentSystem Adapter (v0)

Translates Genesis agent definitions into AgentSystem format:
- Creates agent configs matching AgentSystem's schema
- Provisions skills in the correct directory structure
- Calls AgentSystem's webapi to register agents
- Returns the deployment endpoint

---

## Project Structure

```
~/genesis-engine/
├── genesis/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # FastAPI route definitions
│   │   ├── dependencies.py    # DI (db, llm, config)
│   │   └── middleware.py      # Logging, rate limiting
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── state_machine.py   # Build lifecycle state machine
│   │   └── queue.py           # Job queue management
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── analyze.py         # Stage 1: Domain analysis
│   │   ├── architect.py       # Stage 2: Agent topology
│   │   ├── build.py           # Stage 3: Prompt/tool generation
│   │   ├── test.py            # Stage 4: Simulation + eval
│   │   └── deploy.py          # Stage 5: Target deployment
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract adapter interface
│   │   └── agentsystem.py     # AgentSystem deployment adapter
│   ├── models/
│   │   ├── __init__.py
│   │   ├── project.py         # Project/domain models
│   │   ├── build.py           # Build/artifact models
│   │   ├── agent.py           # Agent/tool/skill models
│   │   └── test_results.py    # Test/evaluation models
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py        # SQLite + SQLAlchemy setup
│   │   └── repository.py      # CRUD operations
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py        # Abstract LLM provider
│   │   ├── openai.py          # OpenAI provider
│   │   └── anthropic.py       # Anthropic provider
│   └── cli/
│       ├── __init__.py
│       ├── main.py            # CLI entry point (Click/Typer)
│       └── display.py         # Rich output formatting
├── tests/
│   ├── test_api/
│   ├── test_pipeline/
│   ├── test_adapters/
│   ├── test_models/
│   └── conftest.py
├── DESIGN.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
└── README.md
```

---

## Security

- Target platform API keys encrypted at rest (Fernet symmetric encryption)
- Generated artifacts redacted before storage (strip API keys from configs)
- Rate limiting on all endpoints (especially `/v1/generate`)
- Input validation on all API endpoints (Pydantic + custom validators)
- Build artifacts expire after 30 days (configurable)

---

## Edge Cases & Error Handling

| Edge Case | Behavior |
|-----------|----------|
| Empty/malformed problem description | 400 with validation error |
| Ambiguous problem (can't extract domain) | 422 with suggestion to clarify |
| LLM returns invalid JSON | Retry with stricter prompt (max 3x) |
| Build exceeds 5-minute timeout | 504 timeout, build marked as failed |
| TEST score below threshold | Retry BUILD→TEST loop (max 3x) |
| Target platform unreachable | Build marked "deploy_failed", artifacts saved |
| Concurrent builds on same project | Queued, executed sequentially |
| Agent name conflict on target | Auto-suffix or return conflict error |

---

## Azure Cloud Deployment

Genesis Engine is designed to run on Azure Container Apps, matching the existing AgentSystem deployment pattern.

### Architecture on Azure

```
Azure Container Apps (eastus2)
├── genesis-engine (this project)
│   ├── FastAPI + Uvicorn
│   ├── SQLite (local) → Azure SQL (production)
│   └── Managed Identity for secrets
├── AgentSystem (existing, same region)
│   └── Deployment target
└── Azure Key Vault
    └── API keys (OpenAI, Anthropic, AgentSystem)
```

### Azure-Ready Configuration

**Environment variables (Azure-native):**
```
GENESIS_DB=sqlite:///genesis.db           # Local/SQLite
GENESIS_DB=postgresql://...                # Azure PostgreSQL Flexible Server
AZURE_KEY_VAULT_URL=https://kv-genesis.vault.azure.net/
OPENAI_API_KEY@keyvault=genesis-openai     # Resolved via Key Vault
ANTHROPIC_API_KEY@keyvault=genesis-anthropic
AGENTSYSTEM_API_KEY@keyvault=agentsystem-key
```

**Container setup:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY genesis/ genesis/
EXPOSE 8000
CMD ["uvicorn", "genesis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Deployment (mirrors AgentSystem pattern):**
```bash
# Build and push to ACR
az acr build --registry acragenesis --image genesis-engine:v1 .

# Deploy to Container Apps
az containerapp update \
  --name genesis-engine \
  --resource-group rg-genesis-prod \
  --image acragenesis.azurecr.io/genesis-engine:v1 \
  --cpu 1.0 --memory 2.0Gi \
  --env-vars GENESIS_DB=... OPENAI_API_KEY=... \
  --ingress external --target-port 8000
```

### Storage Strategy

| Environment | Database | Rationale |
|-------------|----------|-----------|
| Local/dev | SQLite | Zero-config, fast iteration |
| Azure (single instance) | SQLite on Azure Files | Persists across restarts |
| Azure (production) | Azure PostgreSQL Flexible Server | Multi-replica, backups |
| Azure (future) | Cosmos DB | Global distribution, agent graph queries |

### Secrets Management

- **Local:** `.env` file (gitignored)
- **Azure:** Key Vault references in Container Apps env vars (`@keyvault` syntax)
- **Fallback:** Direct env vars for CI/CD testing

---

## v0 Scope (Ship This)

- [x] REST API with all endpoints above
- [x] Python SDK
- [x] CLI with live progress
- [x] Full 5-stage pipeline
- [x] AgentSystem deployment adapter
- [x] SQLite storage
- [x] OpenAI + Anthropic LLM providers
- [ ] Test quality threshold at 0.80

## v1 Scope (Next)

- [ ] Web dashboard (Matrix/cyber-themed)
- [ ] Hermes deployment adapter
- [ ] Generic REST deployment adapter
- [ ] Template system (reusable agent patterns)
- [ ] PostgreSQL storage
- [ ] Multi-model per stage (cheap for ANALYZE, best for ARCHITECT)

---

*End of DESIGN.md — approved for implementation.*
