"""Real deployment guide generator.

Generates step-by-step deployment instructions, Dockerfiles,
docker-compose files, Kubernetes manifests, and cloud deployment
configs for generated multi-agent systems.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from genesis.models.agent import AgentDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class DeploymentOption:
    name: str  # "docker", "azure", "aws", "kubernetes", "local"
    description: str
    complexity: str  # "simple", "moderate", "advanced"
    files: Dict[str, str] = field(default_factory=dict)  # filename → content
    steps: List[str] = field(default_factory=list)
    estimated_time: str = "10 minutes"

@dataclass
class DeploymentPackage:
    project_name: str
    agents: List[AgentDefinition]
    options: List[DeploymentOption] = field(default_factory=list)
    recommended: str = "docker"  # which option is recommended
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "agent_count": len(self.agents),
            "agents": [a.model_dump() for a in self.agents],
            "options": [
                {
                    "name": o.name,
                    "description": o.description,
                    "complexity": o.complexity,
                    "files": o.files,
                    "steps": o.steps,
                    "estimated_time": o.estimated_time,
                }
                for o in self.options
            ],
            "recommended": self.recommended,
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Guide generator
# ---------------------------------------------------------------------------

class DeploymentGuideGenerator:
    """Generates deployment packages for multi-agent systems."""

    def generate(
        self,
        agents: List[AgentDefinition],
        project_name: str = "multi-agent-system",
        topology: str = "router",
    ) -> DeploymentPackage:
        """Generate a full deployment package with all options."""
        logger.info("Generating deployment package for '%s' (%d agents)...",
                    project_name, len(agents))

        package = DeploymentPackage(
            project_name=project_name,
            agents=agents,
            summary=f"Multi-agent system with {len(agents)} agents using {topology} topology.",
        )

        # Generate all deployment options
        package.options.append(self._docker_deployment(agents, project_name))
        package.options.append(self._azure_deployment(agents, project_name))
        package.options.append(self._kubernetes_deployment(agents, project_name))
        package.options.append(self._local_deployment(agents, project_name))
        package.options.append(self._agentsystem_deployment(agents, project_name))

        # Recommend based on agent count
        if len(agents) <= 3:
            package.recommended = "docker"
        elif len(agents) <= 8:
            package.recommended = "azure"
        else:
            package.recommended = "kubernetes"

        return package

    # ---- Deployment options ----

    def _docker_deployment(self, agents: List[AgentDefinition], name: str) -> DeploymentOption:
        agent_names = [a.name for a in agents]
        service_list = "\n".join(f"  - {n}" for n in agent_names)

        dockerfile = f"""FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn httpx pydantic
COPY agents/ /app/agents/
COPY main.py /app/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1
"""

        compose = f"""version: '3.8'
services:
  router:
    build: .
    ports: ["8000:8000"]
    environment:
      - AGENTS={','.join(agent_names)}
      - OPENAI_API_KEY=${{OPENAI_API_KEY}}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
{chr(10).join(f'  {n}:\n    build: ./agents/{n}\n    depends_on: [router]' for n in agent_names)}
"""

        return DeploymentOption(
            name="docker",
            description="Single-machine deployment using Docker Compose. Best for development and small-scale production.",
            complexity="simple",
            files={
                "Dockerfile": dockerfile,
                "docker-compose.yml": compose,
                ".env.example": "# Copy to .env and fill in:\nOPENAI_API_KEY=your-key-here\n",
            },
            steps=[
                "1. Install Docker and Docker Compose",
                "2. Copy .env.example to .env and fill in your API keys",
                "3. Run: docker-compose up --build -d",
                "4. Verify: curl http://localhost:8000/health",
                "5. Test: curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"message\":\"hello\"}'",
            ],
            estimated_time="5 minutes",
        )

    def _azure_deployment(self, agents: List[AgentDefinition], name: str) -> DeploymentOption:
        agent_names = [a.name for a in agents]

        return DeploymentOption(
            name="azure",
            description="Azure Container Apps deployment with managed scaling, HTTPS, and monitoring.",
            complexity="moderate",
            files={
                "Dockerfile": f"FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app/\nRUN pip install -r requirements.txt\nEXPOSE 8000\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]",
                "deploy.sh": f"""#!/bin/bash
# Azure Container Apps Deployment
RESOURCE_GROUP="rg-{name}"
ACR_NAME="acr{name.replace('-','')}"
APP_NAME="{name}"

az group create --name $RESOURCE_GROUP --location eastus2
az acr create --name $ACR_NAME --resource-group $RESOURCE_GROUP --sku Basic
az acr build --registry $ACR_NAME --image {name}:latest .
az containerapp create \\
  --name $APP_NAME \\
  --resource-group $RESOURCE_GROUP \\
  --image $ACR_NAME.azurecr.io/{name}:latest \\
  --registry-server $ACR_NAME.azurecr.io \\
  --target-port 8000 \\
  --ingress external \\
  --env-vars AGENTS='{','.join(agent_names)}' OPENAI_API_KEY=secretref:openai-key
echo "Deployed to: https://$APP_NAME.$(az containerapp show --name $APP_NAME -g $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)"
""",
            },
            steps=[
                "1. Install Azure CLI: az login",
                "2. Make deploy.sh executable: chmod +x deploy.sh",
                "3. Run: ./deploy.sh",
                "4. Set secrets: az containerapp secret set --name APP_NAME -g rg-name --secrets openai-key=YOUR_KEY",
                "5. Verify the HTTPS endpoint returned by the script",
            ],
            estimated_time="10 minutes",
        )

    def _kubernetes_deployment(self, agents: List[AgentDefinition], name: str) -> DeploymentOption:
        return DeploymentOption(
            name="kubernetes",
            description="Kubernetes deployment with horizontal pod autoscaling, ingress, and persistent storage.",
            complexity="advanced",
            files={
                "k8s/deployment.yaml": f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}-router
spec:
  replicas: 2
  selector:
    matchLabels:
      app: {name}-router
  template:
    metadata:
      labels:
        app: {name}-router
    spec:
      containers:
      - name: router
        image: {name}:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: {name}-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: {name}-router
spec:
  selector:
    app: {name}-router
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}-ingress
spec:
  rules:
  - host: {name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {name}-router
            port:
              number: 80
""",
                "k8s/hpa.yaml": f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {name}-router
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
""",
            },
            steps=[
                "1. Build image: docker build -t {name}:latest .",
                "2. Push to registry: docker tag {name}:latest your-registry/{name}:latest && docker push",
                "3. Create secrets: kubectl create secret generic {name}-secrets --from-literal=openai-api-key=YOUR_KEY",
                "4. Apply: kubectl apply -f k8s/",
                "5. Verify: kubectl get pods, kubectl get ingress",
            ],
            estimated_time="20 minutes",
        )

    def _local_deployment(self, agents: List[AgentDefinition], name: str) -> DeploymentOption:
        agent_list = "\n".join(
            f"  - {a.name}: {a.role}" for a in agents
        )
        main_py = f'''"""Auto-generated multi-agent system: {name}"""
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="{name}")

AGENTS = {json.dumps([a.model_dump() for a in agents], indent=2)}

class ChatRequest(BaseModel):
    message: str
    agent: str = "router"

@app.get("/health")
async def health():
    return {{"status": "healthy", "agents": {len(agents)}}}

@app.post("/chat")
async def chat(req: ChatRequest):
    # Route to appropriate agent
    return {{"response": f"[{{req.agent}}] Processing: {{req.message}}", "agents_available": {len(agents)}}}

@app.get("/agents")
async def list_agents():
    return {{"agents": [{{"name": a.name, "role": a.role}} for a in agents]}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

        return DeploymentOption(
            name="local",
            description="Run locally with Python + uvicorn. Best for development and testing.",
            complexity="simple",
            files={
                "main.py": main_py,
                "requirements.txt": "fastapi>=0.100.0\nuvicorn>=0.23.0\nhttpx>=0.27.0\npydantic>=2.0\n",
                "README.md": f"""# {name}

Auto-generated multi-agent system with {len(agents)} agents:

{agent_list}

## Quick Start

```bash
pip install -r requirements.txt
python main.py
# Visit http://localhost:8000/docs for API documentation
```
""",
            },
            steps=[
                "1. pip install -r requirements.txt",
                "2. python main.py",
                "3. Open http://localhost:8000/docs",
                "4. Test: curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"message\":\"hello\"}'",
            ],
            estimated_time="2 minutes",
        )

    def _agentsystem_deployment(self, agents: List[AgentDefinition], name: str) -> DeploymentOption:
        return DeploymentOption(
            name="agentsystem",
            description="Deploy directly to AgentSystem — your enterprise multi-agent platform running on Azure Container Apps.",
            complexity="simple",
            files={
                "agents.json": json.dumps([a.model_dump() for a in agents], indent=2),
                "deploy.sh": f"""#!/bin/bash
# Deploy to AgentSystem
ENDPOINT="${{AGENTSYSTEM_ENDPOINT:-http://localhost:8080}}"
API_KEY="${{AGENTSYSTEM_API_KEY}}"

if [ -z "$API_KEY" ]; then
  echo "Set AGENTSYSTEM_API_KEY environment variable"
  exit 1
fi

# Import agents
for agent_file in agents/*.json; do
  echo "Deploying $agent_file..."
  curl -X POST "$ENDPOINT/v1/agents/import" \\
    -H "Authorization: Bearer $API_KEY" \\
    -H "Content-Type: application/json" \\
    -d @"$agent_file"
done

echo "Deployment complete! Visit $ENDPOINT/dashboard"
""",
            },
            steps=[
                "1. Set AGENTSYSTEM_API_KEY environment variable",
                "2. Set AGENTSYSTEM_ENDPOINT (default: your Azure Container Apps URL)",
                "3. Run: bash deploy.sh",
                "4. Visit the AgentSystem dashboard to see your agents",
            ],
            estimated_time="3 minutes",
        )


__all__ = ["DeploymentGuideGenerator", "DeploymentPackage", "DeploymentOption"]
