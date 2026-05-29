"""Test results models — simulation evaluation output."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class TestFailure(BaseModel):
    """A single test failure with details for the BUILD feedback loop."""

    scenario: str
    agent: str
    expected: str
    actual: str
    metric: str


class TestResults(BaseModel):
    """Output of TEST stage — quality evaluation."""

    scenarios_run: int = Field(0, ge=0)
    scenarios_passed: int = Field(0, ge=0)
    overall_score: float = Field(0.0, ge=0.0, le=1.0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    failures: List[TestFailure] = Field(default_factory=list)
    passed: bool = False

    @property
    def pass_rate(self) -> float:
        if self.scenarios_run == 0:
            return 0.0
        return self.scenarios_passed / self.scenarios_run

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    def check_threshold(self, threshold: float = 0.80) -> bool:
        """Check if overall score meets threshold."""
        self.passed = self.overall_score >= threshold
        return self.passed
