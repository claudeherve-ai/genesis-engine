"""Deployment target adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
from genesis.models.agent import AgentDefinition


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""

    deployment_id: str
    endpoint_url: str
    status: str = "active"
    agent_count: int = 0
    metadata: dict = field(default_factory=dict)


class DeploymentTarget(ABC):
    """Abstract interface for deployment targets.

    Implementations handle provisioning Genesis-generated agents
    on specific platforms (AgentSystem, Hermes, generic REST, etc.).
    """

    @abstractmethod
    async def provision(
        self, agents: List[AgentDefinition]
    ) -> DeploymentResult:
        """Provision agents on the target platform.

        Args:
            agents: List of fully-defined agent definitions.

        Returns:
            DeploymentResult with endpoint URL and status.

        Raises:
            RuntimeError: If provisioning fails.
        """
        ...

    @abstractmethod
    async def health_check(self, deployment_id: str) -> bool:
        """Verify the deployment is healthy and reachable."""
        ...

    @abstractmethod
    async def teardown(self, deployment_id: str) -> None:
        """Remove the deployment from the target platform.

        Best-effort — should not raise on cleanup failures.
        """
        ...
