"""Production feedback loop — observe agent performance, feed insights back.

Collects metrics from deployed agents, analyzes performance patterns,
and feeds learnings back to improve future agent generation.

CITATION: Built for continuous improvement of generated agents.
Session: Hermes Agent, 2026-06-01.
BACK-LINK: /home/tedch/genesis-engine/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Metrics collected from a deployed agent."""
    agent_name: str
    agent_version: str  # hash of the agent definition
    total_requests: int = 0
    successful_responses: int = 0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    error_rate: float = 0.0
    user_satisfaction: float = 0.0  # 0-1
    tool_usage: Dict[str, int] = field(default_factory=dict)
    common_escalations: List[str] = field(default_factory=list)
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeedbackInsight:
    """Actionable insight derived from metrics."""
    agent_name: str
    category: str  # "prompt", "tool", "handoff", "performance"
    severity: str  # "low", "medium", "high", "critical"
    insight: str
    suggested_fix: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeedbackReport:
    """Full feedback report with metrics and insights."""
    build_id: str = ""
    agents: List[AgentMetrics] = field(default_factory=list)
    insights: List[FeedbackInsight] = field(default_factory=list)
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackCollector:
    """Collects and analyzes production metrics from deployed agents."""

    def __init__(self, storage_dir: str = "~/.genesis/feedback"):
        self._dir = Path(storage_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def record_metrics(self, metrics: AgentMetrics) -> None:
        """Record metrics for an agent."""
        agent_dir = self._dir / metrics.agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        filepath = agent_dir / f"{metrics.collected_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath.write_text(json.dumps({
            "agent_name": metrics.agent_name,
            "agent_version": metrics.agent_version,
            "total_requests": metrics.total_requests,
            "successful_responses": metrics.successful_responses,
            "avg_latency_ms": metrics.avg_latency_ms,
            "avg_confidence": metrics.avg_confidence,
            "error_rate": metrics.error_rate,
            "user_satisfaction": metrics.user_satisfaction,
            "tool_usage": metrics.tool_usage,
            "common_escalations": metrics.common_escalations,
            "collected_at": metrics.collected_at.isoformat(),
        }, indent=2))

        logger.info("Recorded metrics for %s: %d requests, %.1f%% success",
                    metrics.agent_name, metrics.total_requests,
                    (metrics.successful_responses / max(metrics.total_requests, 1)) * 100)

    def load_metrics(self, agent_name: str, days: int = 30) -> List[AgentMetrics]:
        """Load metrics for an agent from the last N days."""
        agent_dir = self._dir / agent_name
        if not agent_dir.exists():
            return []

        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        metrics_list = []

        for f in sorted(agent_dir.glob("*.json"), reverse=True):
            if f.stat().st_mtime < cutoff:
                continue
            try:
                data = json.loads(f.read_text())
                metrics_list.append(AgentMetrics(
                    agent_name=data["agent_name"],
                    agent_version=data.get("agent_version", ""),
                    total_requests=data.get("total_requests", 0),
                    successful_responses=data.get("successful_responses", 0),
                    avg_latency_ms=data.get("avg_latency_ms", 0),
                    avg_confidence=data.get("avg_confidence", 0),
                    error_rate=data.get("error_rate", 0),
                    user_satisfaction=data.get("user_satisfaction", 0),
                    tool_usage=data.get("tool_usage", {}),
                    common_escalations=data.get("common_escalations", []),
                ))
            except Exception as e:
                logger.debug("Failed to load metrics file %s: %s", f, e)

        return metrics_list

    def analyze(self, agent_name: str, days: int = 7) -> List[FeedbackInsight]:
        """Analyze metrics and generate insights."""
        metrics_list = self.load_metrics(agent_name, days)
        if not metrics_list:
            return []

        insights: List[FeedbackInsight] = []

        # Aggregate
        latest = metrics_list[0]
        success_rate = latest.successful_responses / max(latest.total_requests, 1)

        # Performance insight
        if success_rate < 0.80:
            insights.append(FeedbackInsight(
                agent_name=agent_name,
                category="performance",
                severity="high",
                insight=f"Success rate is {success_rate:.1%} — below 80% threshold",
                suggested_fix="Review agent system prompt for clarity. Check tool error handling.",
                evidence={"success_rate": success_rate, "total_requests": latest.total_requests},
            ))

        # Latency insight
        if latest.avg_latency_ms > 5000:
            insights.append(FeedbackInsight(
                agent_name=agent_name,
                category="performance",
                severity="medium",
                insight=f"Avg latency {latest.avg_latency_ms:.0f}ms is high",
                suggested_fix="Consider reducing tool calls, adding caching, or optimizing prompts.",
                evidence={"avg_latency_ms": latest.avg_latency_ms},
            ))

        # Tool usage insight
        unused_tools = [t for t, count in latest.tool_usage.items() if count == 0]
        if unused_tools:
            insights.append(FeedbackInsight(
                agent_name=agent_name,
                category="tool",
                severity="low",
                insight=f"Unused tools: {', '.join(unused_tools)}",
                suggested_fix="Remove unused tools or add scenarios where they are needed.",
                evidence={"unused_tools": unused_tools},
            ))

        # Escalation insight
        if latest.common_escalations and latest.error_rate > 0.2:
            insights.append(FeedbackInsight(
                agent_name=agent_name,
                category="handoff",
                severity="medium",
                insight=f"High escalation rate: {latest.error_rate:.1%}",
                suggested_fix="Improve agent's ability to handle common cases before escalating.",
                evidence={"error_rate": latest.error_rate, "escalations": latest.common_escalations},
            ))

        return insights

    def generate_report(self, build_id: str, agent_names: List[str]) -> FeedbackReport:
        """Generate a full feedback report for a build."""
        all_metrics: List[AgentMetrics] = []
        all_insights: List[FeedbackInsight] = []

        for name in agent_names:
            metrics_list = self.load_metrics(name)
            if metrics_list:
                all_metrics.append(metrics_list[0])
            insights = self.analyze(name)
            all_insights.extend(insights)

        overall = 0.0
        if all_metrics:
            rates = [
                m.successful_responses / max(m.total_requests, 1)
                for m in all_metrics
            ]
            overall = sum(rates) / len(rates)

        recommendations = list(dict.fromkeys(
            i.suggested_fix for i in all_insights if i.severity in ("high", "critical")
        ))

        return FeedbackReport(
            build_id=build_id,
            agents=all_metrics,
            insights=all_insights,
            overall_score=round(overall, 4),
            recommendations=recommendations,
        )

    def feed_back_to_generator(self, report: FeedbackReport) -> Dict[str, Any]:
        """Convert feedback into structured input for the BUILD stage retry loop."""
        return {
            "overall_score": report.overall_score,
            "top_issues": [
                {"agent": i.agent_name, "issue": i.insight, "fix": i.suggested_fix}
                for i in report.insights[:5]
            ],
            "recommendations": report.recommendations,
        }


__all__ = [
    "AgentMetrics", "FeedbackInsight", "FeedbackReport",
    "FeedbackCollector",
]
