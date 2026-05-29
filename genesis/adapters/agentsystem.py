"""AgentSystem deployment adapter.

Translates Genesis AgentDefinitions into AgentSystem API calls
for provisioning, health checking, and teardown.
"""

import os
import httpx
from typing import List, Dict, Any
from dataclasses import dataclass, field

from genesis.adapters.base import DeploymentTarget, DeploymentResult
from genesis.models.agent import AgentDefinition


@dataclass
class AgentSystemAdapter(DeploymentTarget):
    """Deploy Genesis agents to an AgentSystem instance."""

    endpoint: str
    api_key: str
    verify_ssl: bool = True
    timeout: float = 30.0

    def __post_init__(self):
        self.endpoint = self.endpoint.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers=headers,
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
        return self._client

    async def provision(
        self, agents: List[AgentDefinition]
    ) -> DeploymentResult:
        """Provision agents on the AgentSystem platform.

        If no API key is configured, returns a dry-run result
        (all agents listed but not actually deployed).
        """
        deployment_id = f"genesis_{len(agents)}_agents"

        # Dry-run mode — no API key configured
        if not self.api_key:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "AgentSystem API key not configured — returning dry-run deployment "
                "for %d agents", len(agents)
            )
            return DeploymentResult(
                deployment_id=deployment_id,
                endpoint_url=f"{self.endpoint}/deploy/{deployment_id}",
                status="dry_run",
                agent_count=len(agents),
                metadata={
                    "mode": "dry_run",
                    "agents": [a.name for a in agents],
                    "note": "Set AGENTSYSTEM_API_KEY to deploy to a live AgentSystem",
                },
            )

        client = await self._get_client()
        registered: list[str] = []

        for agent in agents:
            payload = {
                "name": agent.name,
                "role": agent.role,
                "system_prompt": agent.system_prompt,
                "tools": [t.model_dump() for t in agent.tools],
                "skills": [s.model_dump() for s in agent.skills],
                "coordination": agent.coordination_rules.model_dump(),
            }

            try:
                response = await client.post(
                    "/api/agents", json=payload
                )
                response.raise_for_status()
                registered.append(agent.name)
            except httpx.HTTPError as e:
                # Attempt teardown of already-registered agents
                for name in registered:
                    try:
                        await client.delete(f"/api/agents/{name}")
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Failed to provision agent '{agent.name}': {e}"
                ) from e

        return DeploymentResult(
            deployment_id=deployment_id,
            endpoint_url=f"{self.endpoint}/deploy/{deployment_id}",
            status="active",
            agent_count=len(registered),
        )

    async def health_check(self, deployment_id: str) -> bool:
        """Verify the deployment is healthy."""
        # Dry-run mode — no real deployment to check
        if not self.api_key:
            return True

        client = await self._get_client()
        try:
            response = await client.get("/api/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def teardown(self, deployment_id: str) -> None:
        """Remove all agents in the deployment."""
        client = await self._get_client()
        try:
            await client.delete(f"/api/deploy/{deployment_id}")
        except httpx.HTTPError:
            pass  # Best-effort teardown

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def create_agentsystem_adapter(
    endpoint: str | None = None,
    api_key: str | None = None,
) -> AgentSystemAdapter:
    """Factory function — reads config from environment if not provided."""
    return AgentSystemAdapter(
        endpoint=endpoint or os.getenv(
            "AGENTSYSTEM_ENDPOINT", "http://localhost:8000"
        ),
        api_key=api_key or os.getenv("AGENTSYSTEM_API_KEY", ""),
    )
