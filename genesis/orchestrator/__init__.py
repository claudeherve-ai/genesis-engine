"""Genesis Engine orchestrator — pipeline lifecycle management.

The orchestrator manages the 5-stage build lifecycle (ANALYZE → ARCHITECT →
BUILD → TEST → DEPLOY) with a TEST→BUILD retry loop (max 3 retries, default
threshold 0.80). It coordinates pipeline stages, updates build status and
storage after each transition, and handles errors gracefully.
"""

from genesis.orchestrator.state_machine import Orchestrator

__all__ = ["Orchestrator"]