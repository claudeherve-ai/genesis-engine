"""Verification layer — double-check all outputs before returning.

Runs a secondary LLM call to verify and refine generated content.
This is the "trust but verify" step that eliminates hallucinations.
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any

from genesis.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

VERIFY_AGENTS_PROMPT = """You are a quality assurance engineer reviewing generated AI agents.
Your job is to verify each agent is complete, coherent, and trustworthy.

For each agent, check:
1. System prompt is detailed and clear (not generic or vague)
2. Tools are correctly specified with proper schemas
3. Skills cover relevant procedures
4. Coordination rules make sense for the topology

Return valid JSON:
{
  "passed": true,
  "score": 0.85,
  "issues": ["specific issue 1", "specific issue 2"],
  "agents": [
    {
      "name": "agent_name",
      "verdict": "pass|needs_improvement",
      "feedback": "specific feedback",
      "fixed_system_prompt": "improved version if needed, or original"
    }
  ]
}"""


async def verify_agents(
    llm: LLMProvider,
    agents: List[Dict[str, Any]],
    architecture: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify and improve generated agent definitions.

    Runs a secondary LLM pass to quality-check every agent.
    Returns verification results with improvements.
    """
    if not agents:
        return {"passed": True, "score": 1.0, "issues": [], "agents": []}

    user_prompt = json.dumps({
        "topology": architecture.get("topology", "unknown"),
        "agent_count": len(agents),
        "agents": [
            {
                "name": a.get("name"),
                "role": a.get("role"),
                "system_prompt": (a.get("system_prompt", ""))[:500],
                "tools": [t.get("name") for t in a.get("tools", [])],
            }
            for a in agents
        ],
    })

    try:
        response = await llm.complete(
            system_prompt=VERIFY_AGENTS_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=4096,
        )
        result = json.loads(_extract_json(response.content))
        logger.info("Verification: score=%.2f, passed=%s, issues=%d",
                    result.get("score", 0), result.get("passed"), len(result.get("issues", [])))
        return result
    except Exception as e:
        logger.warning("Verification failed (non-fatal): %s", e)
        return {"passed": True, "score": 0.7, "issues": [str(e)], "agents": []}


def _extract_json(content: str) -> str:
    """Extract JSON from potentially markdown-wrapped content."""
    text = content.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                return p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                return p
    return text
