# Genesis Engine

> **AI that builds AI.**

Describe a problem in one sentence. Genesis Engine designs, builds, **really executes & tests**,
and deploys a complete multi-agent system — with verified tools, citations from live docs,
cost tracking, and a one-click deploy package.

**Version 0.4.0** · FastAPI · Async pipeline · Runs green **offline** (no API keys required for tests/UI).

---

## What it does

You give it a sentence. A 5-stage async pipeline does the rest:

```
ANALYZE  →  ARCHITECT  →  BUILD  →  TEST  →  DEPLOY
   ↓            ↓           ↓        ↓         ↓
 Domain      Agent       Agent    REAL     Deploy
 Model      Topology +   Defs +   exec +   package +
           Scenarios    Tool      scored   guides
                        validation transcripts
                          ↻ retry on hallucinated tools / low score (max 3x)
```

### What makes it different
- **Real execution, not simulation.** The TEST stage runs the generated agents against concrete
  scenarios (input + expected outputs + expected tools) through `AgentRuntime`, then scores from
  reality and persists **visible transcripts**.
- **Hallucination gate.** BUILD validates every tool an agent claims to use; hallucinated tools
  trigger a retry. Research is grounded against **live Microsoft Learn docs** (MCP) and a real
  search provider, with **citations** surfaced end-to-end.
- **Enterprise-ready.** API-key auth + RBAC, rate limiting, Fernet-encrypted secrets in the deploy
  path, cost/token tracking with per-stage budgets, and optional OpenTelemetry traces.
- **Smart & sticky.** Multi-model routing, a persistent knowledge store with grounding, a
  feedback → auto-rebuild loop, and an offline eval regression suite with golden datasets.
- **A platform.** Template/recipe gallery, agent versioning + rollback, human-in-the-loop approval
  gate, and a one-click deploy package (Docker / Compose / Azure Container Apps / Kubernetes).

---

## 60-second quickstart (local)

```bash
git clone https://github.com/claudedjale/genesis-engine.git
cd genesis-engine
python -m venv .venv

# Activate the venv:
#   Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
#   macOS / Linux:         source .venv/bin/activate

pip install -e ".[dev]"

# Start the server (serves the UI at http://localhost:8000)
genesis server
```

Open **http://localhost:8000** for the UI, or **http://localhost:8000/docs-guide** for the
full build & deploy guide. The interactive API reference (Swagger) is at **/docs**.

> **Offline mode:** With no LLM keys configured, the server, UI, docs, REST API, and the full
> test suite all run. The build *pipeline* needs an LLM provider (see below) to generate agents,
> but everything else — including auth (open by default), templates, metrics, and the deploy
> package endpoints — works keyless.

---

## Build your first system

### CLI

```bash
# Start the server in one terminal
genesis server --host 0.0.0.0 --port 8000

# In another terminal, kick off a build
genesis build "I need a customer support system with triage, tech support, and billing agents"

# Track a build and list projects
genesis status <build_id>
genesis projects
```

### REST API

```bash
# One-shot: problem -> queued build
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"problem":"Build a support system for my SaaS","target":"local"}'

# -> { "build_id": "...", "project_id": "...", "status": "queued",
#      "status_url": "/v1/builds/<id>", "logs_url": "/v1/builds/<id>/logs" }

# Stream build logs (Server-Sent Events)
curl -N http://localhost:8000/v1/builds/<build_id>/logs

# List validated tools
curl http://localhost:8000/v1/tools

# Get the one-click deploy package
curl http://localhost:8000/v1/builds/<build_id>/deploy-package
```

---

## Run it anywhere

### Docker

```bash
docker build -t genesis-engine:0.4.0 .
docker run -p 8000:8000 \
  -e AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY \
  -e AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT \
  genesis-engine:0.4.0
```

### Docker Compose

```yaml
services:
  genesis:
    build: .
    ports:
      - "8000:8000"
    environment:
      AZURE_OPENAI_API_KEY: ${AZURE_OPENAI_API_KEY}
      AZURE_OPENAI_ENDPOINT: ${AZURE_OPENAI_ENDPOINT}
      GENESIS_API_KEYS: ${GENESIS_API_KEYS}      # optional: enables auth
      GENESIS_SECRET_KEY: ${GENESIS_SECRET_KEY}  # optional: Fernet secret encryption
```

```bash
docker compose up --build
```

### Azure Container Apps

```bash
# Build and push to ACR
az acr build --registry <your_registry> --image genesis-engine:0.4.0 .

# Deploy
az containerapp create \
  --name genesis-engine \
  --resource-group <your_rg> \
  --environment <your_aca_env> \
  --image <your_registry>.azurecr.io/genesis-engine:0.4.0 \
  --target-port 8000 --ingress external \
  --env-vars \
    AZURE_OPENAI_API_KEY=secretref:openai-key \
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com \
    GENESIS_SECRET_KEY=secretref:genesis-secret
```

### Kubernetes

```bash
kubectl create deployment genesis --image=<registry>/genesis-engine:0.4.0
kubectl set env deployment/genesis \
  AZURE_OPENAI_API_KEY=<key> AZURE_OPENAI_ENDPOINT=<endpoint>
kubectl expose deployment genesis --port=80 --target-port=8000 --type=LoadBalancer
```

> The **/docs-guide** page (and the Docs button in the UI) contains copy-paste tabs for each of
> these environments, plus troubleshooting and an env-var reference.

---

## Configuration

All configuration is via environment variables. Everything has a safe default; the server runs
with **none** of these set (auth open, no LLM = pipeline disabled, SQLite database).

| Variable | Default | Purpose |
|----------|---------|---------|
| `GENESIS_HOST` | `0.0.0.0` | Server bind host |
| `GENESIS_PORT` | `8000` | Server port |
| `GENESIS_DB` | `sqlite:///genesis.db` | Database URL |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `AZURE_OPENAI_API_KEY` | — | Enables the build pipeline (Azure OpenAI) |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | Azure OpenAI API version |
| `ANTHROPIC_API_KEY` | — | Optional model provider (routing) |
| `GEMINI_API_KEY` | — | Optional model provider (routing) |
| `DEEPSEEK_API_KEY` | — | Optional model provider (routing) |
| `TAVILY_API_KEY` | — | Real web search (DuckDuckGo fallback when unset) |
| `AGENTSYSTEM_ENDPOINT` | `http://localhost:8000` | Deploy target endpoint |
| `AGENTSYSTEM_API_KEY` | — | Deploy target auth |
| `GENESIS_API_KEYS` | — | `admin:KEY,user:KEY` — enables auth + RBAC (open when unset) |
| `GENESIS_SECRET_KEY` | — | Fernet key for secret encryption in deploy configs |
| `AZURE_KEY_VAULT_URL` | — | Optional Key Vault for secret resolution |
| `GENESIS_RATE_LIMIT` | `600` | Requests/min per key or IP |
| `GENESIS_RATE_BURST` | = rate limit | Burst allowance |
| `GENESIS_OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OTLP traces export endpoint |

### Auth & RBAC
Set `GENESIS_API_KEYS="admin:<adminkey>,user:<userkey>"` to require an `X-API-Key` header.
`admin` keys can approve/reject builds and roll back projects; `user` keys can build and read.
When `GENESIS_API_KEYS` is unset, the API is **open** (anonymous admin) for local dev and tests.

---

## Key API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/generate` | One-shot: problem → queued build |
| POST | `/v1/projects` | Create project |
| GET | `/v1/projects` | List projects |
| POST | `/v1/projects/{id}/build` | Trigger build pipeline |
| GET | `/v1/builds/{id}` | Build status |
| GET | `/v1/builds/{id}/logs` | Streaming build logs (SSE) |
| GET | `/v1/builds/{id}/artifacts` | Generated artifacts (incl. transcripts) |
| GET | `/v1/builds/{id}/deploy-package` | One-click deploy package (Docker/Compose/ACA/K8s) |
| GET | `/v1/builds/{id}/approval` | Approval state for a paused build |
| POST | `/v1/builds/{id}/approve` · `/reject` | Human-in-the-loop gate (admin) |
| GET | `/v1/projects/{id}/versions` | Agent version history |
| POST | `/v1/projects/{id}/rollback` | Roll back to a previous version (admin) |
| GET | `/v1/templates` | Recipe / template gallery |
| GET | `/v1/tools` | Validated tool catalog |
| GET | `/v1/metrics` | Cost / token / usage metrics |
| GET | `/health` | Liveness probe |
| GET | `/docs` | Interactive API reference (Swagger) |
| GET | `/docs-guide` | Full build & deploy guide |

---

## Development

```bash
pip install -e ".[dev]"

# Run the full test suite (389 tests, runs fully offline)
python -m pytest -q

# Sanity-check the app imports
python -c "import genesis.api.app"
```

The eval regression suite (`genesis/eval/`) runs against golden JSON datasets with a mocked LLM,
so it stays deterministic and green without any API keys.

---

## License

MIT
