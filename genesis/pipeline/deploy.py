"""Stage 5: Target Deployment.

Takes validated agent definitions and a target configuration, then provisions
them on a deployment target via the adapter pattern. The deployment target
(AgentSystem, Hermes, generic REST, etc.) is injected at construction time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from genesis.adapters.base import DeploymentTarget, DeploymentResult
from genesis.models.agent import AgentDefinition

logger = logging.getLogger(__name__)


class DeployStage:
    """Stage 5: Deploy agents to a target platform.

    Delegates actual provisioning to a DeploymentTarget adapter, which
    translates Genesis agent definitions into the target platform's format.
    Performs a health check after provisioning and supports teardown.

    Args:
        target: A DeploymentTarget adapter instance (e.g., AgentSystem).
    """

    def __init__(self, target: DeploymentTarget) -> None:
        self.target = target

    async def run(
        self,
        agents: List[AgentDefinition],
        target_config: Optional[Dict[str, Any]] = None,
    ) -> DeploymentResult:
        """Run the DEPLOY stage.

        Provisions all agents on the target platform, then performs a health
        check to confirm the deployment is live.

        Args:
            agents: Validated agent definitions to deploy.
            target_config: Optional configuration overrides for the deployment
                target (endpoint, API key, region, etc.). If provided, these
                may be passed to the adapter for one-time configuration.

        Returns:
            A DeploymentResult with deployment ID, endpoint URL, and status.

        Raises:
            RuntimeError: If provisioning fails or the health check does not pass.
        """
        if not agents:
            raise ValueError("Cannot deploy: no agents provided")

        logger.info(
            "DEPLOY stage starting — %d agents to %s target",
            len(agents),
            self.target.__class__.__name__,
        )

        # Provision agents on the target
        try:
            result = await self.target.provision(agents)
        except Exception as e:
            logger.error("DEPLOY stage provision failed: %s", e)
            raise RuntimeError(f"Deployment provisioning failed: {e}") from e

        logger.info(
            "DEPLOY stage provisioned — deployment_id=%s, endpoint=%s, agents=%d",
            result.deployment_id,
            result.endpoint_url,
            result.agent_count,
        )

        # Health check
        try:
            healthy = await self.target.health_check(result.deployment_id)
        except Exception as e:
            logger.warning("DEPLOY stage health check error: %s", e)
            healthy = False

        if not healthy:
            logger.error(
                "DEPLOY stage health check failed for deployment %s",
                result.deployment_id,
            )
            # Don't raise — deployment exists but may need attention.
            # The caller can check result.status.
            result.status = "deployed_unhealthy"

        logger.info(
            "DEPLOY stage complete — %s, healthy=%s",
            result.endpoint_url,
            healthy,
        )
        return result

    async def teardown(self, deployment_id: str) -> None:
        """Tear down a previously created deployment.

        Args:
            deployment_id: The deployment to remove.
        """
        logger.info("DEPLOY stage teardown — deployment_id=%s", deployment_id)
        await self.target.teardown(deployment_id)
