"""Curated recipe gallery.

A *recipe* is a battle-tested prompt + suggested target + tags that gives users
a one-click starting point ("HOLY SHIT it built the thing I was imagining").
Recipes are static, in-process data — no I/O, no LLM, fully offline-safe.

Each recipe's ``prompt`` is a complete, high-signal problem statement designed
to flow straight into ``POST /v1/generate`` (or the build trigger). They are
intentionally concrete so the downstream pipeline produces sharp agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Recipe:
    """A curated, ready-to-run system template."""

    id: str
    title: str
    category: str
    summary: str
    prompt: str
    target: str = "agentsystem"
    tags: tuple = ()
    suggested_agents: tuple = ()
    icon: str = "✨"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "summary": self.summary,
            "prompt": self.prompt,
            "target": self.target,
            "tags": list(self.tags),
            "suggested_agents": list(self.suggested_agents),
            "icon": self.icon,
        }


_RECIPES: List[Recipe] = [
    Recipe(
        id="support-triage",
        title="Customer Support Triage",
        category="Support",
        summary="Route, draft, and escalate inbound customer tickets with a "
        "knowledge-grounded responder.",
        prompt=(
            "Build a customer support system that triages inbound tickets by "
            "urgency and topic, drafts grounded replies from our help-center "
            "docs, and escalates anything involving billing disputes or "
            "security to a human with a concise summary."
        ),
        target="agentsystem",
        tags=("support", "triage", "rag", "escalation"),
        suggested_agents=("Triage Router", "Knowledge Responder", "Escalation Summarizer"),
        icon="🎧",
    ),
    Recipe(
        id="research-analyst",
        title="Market Research Analyst",
        category="Research",
        summary="Search the live web, synthesize findings with citations, and "
        "produce an executive brief.",
        prompt=(
            "Build a research assistant that takes a market or competitor "
            "question, searches the live web, cross-checks claims across "
            "sources, and writes a one-page executive brief with inline "
            "citations and a confidence rating per claim."
        ),
        target="agentsystem",
        tags=("research", "web-search", "citations", "synthesis"),
        suggested_agents=("Query Planner", "Web Researcher", "Synthesis Writer"),
        icon="🔎",
    ),
    Recipe(
        id="data-copilot",
        title="Analytics Data Copilot",
        category="Data",
        summary="Turn plain-English questions into validated SQL and narrated "
        "insights over a warehouse.",
        prompt=(
            "Build a data analytics copilot that converts natural-language "
            "questions into validated SQL for a Postgres warehouse, runs the "
            "query, explains the result in plain English, and flags when a "
            "question is ambiguous or the data looks anomalous."
        ),
        target="agentsystem",
        tags=("data", "sql", "analytics", "nl2sql"),
        suggested_agents=("Schema Mapper", "SQL Generator", "Insight Narrator"),
        icon="📊",
    ),
    Recipe(
        id="devops-incident",
        title="DevOps Incident Responder",
        category="DevOps",
        summary="Diagnose alerts, propose safe mitigations, and draft the "
        "incident timeline.",
        prompt=(
            "Build an incident-response system that ingests an alert and recent "
            "logs, ranks likely root causes with evidence, proposes "
            "reversible mitigations (never destructive), and drafts a "
            "customer-safe status update plus a post-incident timeline."
        ),
        target="agentsystem",
        tags=("devops", "sre", "incident", "observability"),
        suggested_agents=("Triage Analyst", "Mitigation Planner", "Comms Writer"),
        icon="🚨",
    ),
    Recipe(
        id="content-studio",
        title="Content Marketing Studio",
        category="Marketing",
        summary="Plan, draft, and fact-check on-brand content across channels.",
        prompt=(
            "Build a content studio that takes a campaign brief, plans a "
            "multi-channel content calendar, drafts on-brand copy for blog, "
            "email, and social, and runs a fact-check pass that flags "
            "unverifiable claims before publishing."
        ),
        target="agentsystem",
        tags=("marketing", "content", "fact-check", "brand"),
        suggested_agents=("Campaign Planner", "Copywriter", "Fact Checker"),
        icon="✍️",
    ),
    Recipe(
        id="code-reviewer",
        title="Pull Request Reviewer",
        category="Engineering",
        summary="Review diffs for bugs, security issues, and missing tests with "
        "high signal-to-noise.",
        prompt=(
            "Build a code-review system that analyzes a pull-request diff, "
            "surfaces only genuine bugs, security vulnerabilities, and logic "
            "errors (never style nits), checks for missing test coverage, and "
            "writes a concise, actionable review summary."
        ),
        target="agentsystem",
        tags=("engineering", "code-review", "security", "testing"),
        suggested_agents=("Diff Analyzer", "Security Auditor", "Review Writer"),
        icon="🧑‍💻",
    ),
    Recipe(
        id="personal-concierge",
        title="Personal Life Concierge",
        category="Personal",
        summary="Plan trips, summarize inboxes, and manage a to-do backlog for "
        "everyday life.",
        prompt=(
            "Build a personal concierge that plans trips end-to-end with a "
            "day-by-day itinerary and budget, summarizes a busy inbox into "
            "what needs a reply, and keeps a prioritized to-do backlog with "
            "gentle reminders."
        ),
        target="agentsystem",
        tags=("personal", "planning", "productivity"),
        suggested_agents=("Trip Planner", "Inbox Summarizer", "Task Manager"),
        icon="🧳",
    ),
    Recipe(
        id="game-master",
        title="Tabletop Game Master",
        category="Entertainment",
        summary="Run an interactive RPG with a narrator, rules referee, and "
        "memory of the campaign.",
        prompt=(
            "Build an interactive tabletop RPG system with a vivid narrator "
            "that drives the story, a rules referee that adjudicates dice and "
            "actions fairly, and a chronicler that remembers characters, "
            "places, and past events to keep the campaign consistent."
        ),
        target="agentsystem",
        tags=("entertainment", "rpg", "memory", "storytelling"),
        suggested_agents=("Narrator", "Rules Referee", "Chronicler"),
        icon="🎲",
    ),
]

_BY_ID: Dict[str, Recipe] = {r.id: r for r in _RECIPES}


def list_recipes(
    category: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Recipe]:
    """Return recipes, optionally filtered by category and/or free-text query.

    Filtering is case-insensitive. ``q`` matches title, summary, prompt, and
    tags. Results preserve curated order.
    """
    results = list(_RECIPES)
    if category:
        cat = category.strip().lower()
        results = [r for r in results if r.category.lower() == cat]
    if q:
        needle = q.strip().lower()
        if needle:
            results = [r for r in results if _matches(r, needle)]
    return results


def _matches(recipe: Recipe, needle: str) -> bool:
    haystacks = [
        recipe.title,
        recipe.summary,
        recipe.prompt,
        recipe.category,
        " ".join(recipe.tags),
    ]
    return any(needle in h.lower() for h in haystacks)


def get_recipe(recipe_id: str) -> Optional[Recipe]:
    """Return a recipe by id, or ``None`` if unknown."""
    return _BY_ID.get(recipe_id)


def list_categories() -> List[str]:
    """Return the distinct categories in curated order."""
    seen: List[str] = []
    for r in _RECIPES:
        if r.category not in seen:
            seen.append(r.category)
    return seen
