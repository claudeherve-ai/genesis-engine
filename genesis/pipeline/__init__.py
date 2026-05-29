"""Genesis Engine pipeline stages.

This package contains the 5-stage pipeline for transforming a natural-language
problem description into a deployed multi-agent system:

    1. analyze  — Domain Analysis (problem → DomainModel)
    2. architect — Agent Topology Design (DomainModel → AgentArchitecture)
    3. build     — Prompt & Tool Generation (AgentArchitecture → AgentDefinition[])
    4. test      — Simulation & Evaluation (AgentDefinition[] → TestResults)
    5. deploy    — Target Deployment (AgentDefinition[] + config → DeploymentResult)

Each stage accepts an LLMProvider (or DeploymentTarget for deploy) in its
__init__, making them testable with mock providers.
"""

from genesis.pipeline.analyze import AnalyzeStage, ANALYZE_SYSTEM_PROMPT
from genesis.pipeline.architect import ArchitectStage, ARCHITECT_SYSTEM_PROMPT
from genesis.pipeline.build import BuildStage, BUILD_SYSTEM_PROMPT
from genesis.pipeline.test import TestStage, TEST_SYSTEM_PROMPT
from genesis.pipeline.deploy import DeployStage

__all__ = [
    # Stage classes
    "AnalyzeStage",
    "ArchitectStage",
    "BuildStage",
    "TestStage",
    "DeployStage",
    # System prompts (exported for testing and customisation)
    "ANALYZE_SYSTEM_PROMPT",
    "ARCHITECT_SYSTEM_PROMPT",
    "BUILD_SYSTEM_PROMPT",
    "TEST_SYSTEM_PROMPT",
]
