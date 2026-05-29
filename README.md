# Genesis Engine

> **"AI that builds AI."**

Describe a problem. Get a deployed, tested multi-agent system in ~90 seconds.

## Quick Start

```bash
# Install
git clone <repo> && cd genesis-engine
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]" --break-system-packages

# Set up your LLM provider
export OPENAI_API_KEY="sk-..."

# Start the server
genesis server

# In another terminal: build an agent system
genesis build "I need a customer support system with triage, tech support, and billing agents"
```

## API

```python
import genesis

# One-shot — the "holy shit" endpoint
system = genesis.generate("Build a support system for my SaaS")
print(system.url)  # deployed endpoint
print(system.agents)  # [triage_agent, technical_agent, billing_agent]

# Step-by-step
project = genesis.create("My Project")
build = project.build("Build me an HR onboarding system")
result = build.wait()
print(f"Score: {result.test_score:.0%}")
```

## CLI

```bash
genesis build "I need a code review system with static analysis and security scanning"

# Output:
#   🔍 ANALYZE    Decomposing problem domain...      ✓
#   🏗️ ARCHITECT  Designing agent topology...         ✓
#   🔧 BUILD      Generating prompts and tools...     ✓
#   🧪 TEST       Running simulations...              ✓
#   🚀 DEPLOY     Provisioning on AgentSystem...       ✓
#   ✓ Build completed in 87.3s
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/generate` | One-shot: problem → deployed system |
| POST | `/v1/projects` | Create project |
| GET | `/v1/projects` | List projects |
| POST | `/v1/projects/{id}/build` | Trigger build pipeline |
| GET | `/v1/builds/{id}` | Build status |
| GET | `/v1/builds/{id}/logs` | Streaming build logs (SSE) |
| GET | `/v1/builds/{id}/artifacts` | Download generated artifacts |

## Azure Deployment

```bash
# Build and push
az acr build --registry acragenesis --image genesis-engine:v1 .

# Deploy
az containerapp update \
  --name genesis-engine \
  --resource-group rg-genesis-prod \
  --image acragenesis.azurecr.io/genesis-engine:v1 \
  --env-vars OPENAI_API_KEY=... AGENTSYSTEM_ENDPOINT=... \
  --ingress external --target-port 8000
```

## Architecture

```
ANALYZE → ARCHITECT → BUILD → TEST → DEPLOY
   ↓          ↓          ↓       ↓        ↓
Domain    Agent      Agent    Sim +   Target
Model    Topology    Defs    Eval    Adapter
                         ↻ retry (max 3x)
```

## License

MIT
