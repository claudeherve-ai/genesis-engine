"""Genesis Engine Python SDK.

Usage:
    import genesis

    # One-shot
    system = genesis.generate("Build a support system for my SaaS")
    print(system.url)

    # Step-by-step
    project = genesis.create("Support system for Acme Corp")
    build = project.build()
    build.wait()
    print(build.agents)
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import httpx


DEFAULT_API_URL = "http://localhost:8000/v1"


@dataclass
class BuildResult:
    """Result of a completed build."""

    build_id: str
    project_id: str
    status: str
    agents: List[Dict[str, Any]] = field(default_factory=list)
    test_score: float = 0.0
    test_results: Optional[Dict[str, Any]] = None
    url: str = ""
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == "completed"


@dataclass
class Project:
    """A Genesis project."""

    id: str
    name: str
    description: str = ""
    status: str = "active"
    build_count: int = 0
    _client: Optional[httpx.Client] = None

    def build(
        self,
        problem: str,
        target: str = "agentsystem",
        target_config: Optional[Dict[str, Any]] = None,
    ) -> "Build":
        """Trigger a build for this project."""
        client = self._get_client()
        response = client.post(
            f"/projects/{self.id}/build",
            json={
                "problem_description": problem,
                "target": target,
                "target_config": target_config or {},
            },
        )
        response.raise_for_status()
        data = response.json()
        return Build(
            id=data["id"],
            project_id=data["project_id"],
            status=data["status"],
            _client=self._client,
        )

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=DEFAULT_API_URL, timeout=300)
        return self._client


@dataclass
class Build:
    """An in-progress or completed build."""

    id: str
    project_id: str
    status: str
    _client: Optional[httpx.Client] = None

    def wait(self, poll_interval: float = 1.5, timeout: float = 300) -> BuildResult:
        """Wait for the build to complete and return the result."""
        client = self._get_client()
        start = time.time()

        while time.time() - start < timeout:
            response = client.get(f"/builds/{self.id}")
            response.raise_for_status()
            data = response.json()

            if data["status"] in ("completed", "failed", "awaiting_approval"):
                artifacts = data.get("artifacts", {}) or {}
                test_results = data.get("test_results", {}) or {}
                return BuildResult(
                    build_id=data["id"],
                    project_id=data["project_id"],
                    status=data["status"],
                    agents=artifacts.get("agents", []),
                    test_score=test_results.get("overall_score", 0),
                    test_results=test_results,
                    error=data.get("error"),
                )

            time.sleep(poll_interval)

        raise TimeoutError(f"Build {self.id} timed out after {timeout}s")

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=DEFAULT_API_URL, timeout=300)
        return self._client


class GenesisClient:
    """Client for the Genesis Engine API."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 300):
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def generate(
        self,
        problem: str,
        target: str = "agentsystem",
        target_config: Optional[Dict[str, Any]] = None,
        project_name: Optional[str] = None,
    ) -> BuildResult:
        """One-shot: describe a problem → deployed agent system.

        Blocks until the build completes.
        """
        response = self.client.post(
            "/generate",
            json={
                "problem": problem,
                "target": target,
                "target_config": target_config or {},
                "project_name": project_name,
            },
        )
        response.raise_for_status()
        data = response.json()

        build = Build(
            id=data["build_id"],
            project_id=data["project_id"],
            status=data["status"],
            _client=self.client,
        )
        return build.wait()

    def create(self, name: str, description: str = "") -> Project:
        """Create a new project."""
        response = self.client.post(
            "/projects",
            json={"name": name, "description": description},
        )
        response.raise_for_status()
        data = response.json()
        return Project(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            _client=self.client,
        )

    def list_projects(self) -> List[Project]:
        """List all projects."""
        response = self.client.get("/projects")
        response.raise_for_status()
        data = response.json()
        return [
            Project(
                id=p["id"],
                name=p["name"],
                description=p.get("description", ""),
                status=p["status"],
                build_count=p.get("build_count", 0),
                _client=self.client,
            )
            for p in data.get("items", [])
        ]

    def get_build(self, build_id: str) -> Build:
        """Get a build by ID."""
        response = self.client.get(f"/builds/{build_id}")
        response.raise_for_status()
        data = response.json()
        return Build(
            id=data["id"],
            project_id=data["project_id"],
            status=data["status"],
            _client=self.client,
        )

    def close(self):
        """Close the HTTP client."""
        self.client.close()


# Module-level convenience functions
_default_client: Optional[GenesisClient] = None


def _get_client() -> GenesisClient:
    global _default_client
    if _default_client is None:
        _default_client = GenesisClient()
    return _default_client


def generate(
    problem: str,
    target: str = "agentsystem",
    **kwargs,
) -> BuildResult:
    """One-shot convenience function."""
    return _get_client().generate(problem, target=target, **kwargs)


def create(name: str, description: str = "") -> Project:
    """Create a project."""
    return _get_client().create(name, description)


def list_projects() -> List[Project]:
    """List all projects."""
    return _get_client().list_projects()
