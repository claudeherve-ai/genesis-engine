"""Genesis Engine CLI — beautiful terminal interface."""

import time
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
import httpx

app = typer.Typer(
    name="genesis",
    help="Genesis Engine — AI that builds AI. Describe a problem, get a deployed multi-agent system.",
    add_completion=False,
)
console = Console()

API_URL = "http://localhost:8000/v1"

# Stage display config
STAGE_EMOJI = {
    "analyze": "🔍",
    "architect": "🏗️",
    "build": "🔧",
    "test": "🧪",
    "deploy": "🚀",
}
STAGE_LABEL = {
    "analyze": "ANALYZE",
    "architect": "ARCHITECT",
    "build": "BUILD",
    "test": "TEST",
    "deploy": "DEPLOY",
}


@app.command()
def build(
    problem: str = typer.Argument(..., help="Problem description for the agent system"),
    target: str = typer.Option("agentsystem", help="Deployment target platform"),
    endpoint: str = typer.Option("http://localhost:8000/v1", help="Genesis API endpoint"),
):
    """One-shot: describe a problem → deployed agent system."""
    client = httpx.Client(base_url=endpoint, timeout=300)

    # Display header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]GENESIS ENGINE[/bold cyan]\n"
            "[dim]AI that builds AI[/dim]",
            border_style="cyan",
        )
    )
    console.print(f"[dim]Problem:[/dim] {problem[:200]}{'...' if len(problem) > 200 else ''}")
    console.print()

    # Send generate request
    try:
        response = client.post(
            "/generate",
            json={"problem": problem, "target": target},
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as e:
        console.print(f"[red]✗ API error: {e}[/red]")
        raise typer.Exit(code=1)

    build_id = data["build_id"]
    project_id = data["project_id"]

    # Poll for status with live display
    stages_completed = []
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Initializing...", total=100)

        while True:
            try:
                r = client.get(f"/builds/{build_id}")
                r.raise_for_status()
                build_data = r.json()
            except httpx.HTTPError:
                time.sleep(1)
                continue

            status = build_data["status"]
            stage = build_data.get("stage")
            error = build_data.get("error")

            # Update progress display
            if stage and stage not in stages_completed:
                stages_completed.append(stage)
                emoji = STAGE_EMOJI.get(stage, "⏳")
                label = STAGE_LABEL.get(stage, stage.upper())
                progress.update(task, description=f"{emoji} {label}")

            # Map status to progress
            progress_map = {
                "analyzing": 20,
                "architecting": 40,
                "building": 60,
                "testing": 80,
                "deploying": 90,
                "completed": 100,
                "failed": 100,
            }
            progress.update(task, completed=progress_map.get(status, 0))

            if status == "completed":
                elapsed = time.time() - start_time
                console.print()
                console.print(f"[bold green]✓ Build completed in {elapsed:.1f}s[/bold green]")

                # Show agents
                artifacts = build_data.get("artifacts", {}) or {}
                agents = artifacts.get("agents", []) or []
                if agents:
                    table = Table(title="Deployed Agents", border_style="green")
                    table.add_column("Name", style="cyan")
                    table.add_column("Role", style="dim")
                    for agent in agents:
                        table.add_row(agent.get("name", "?"), agent.get("role", "?"))
                    console.print(table)

                # Show test results
                test_results = build_data.get("test_results", {}) or {}
                if test_results:
                    score = test_results.get("overall_score", 0)
                    score_color = "green" if score >= 0.80 else "yellow"
                    console.print(f"[{score_color}]Test Score: {score:.0%}[/{score_color}]")

                console.print(f"\n[dim]Project ID: {project_id}[/dim]")
                console.print(f"[dim]Build ID: {build_id}[/dim]")
                break

            elif status == "failed":
                console.print()
                console.print(f"[bold red]✗ Build failed[/bold red]")
                if error:
                    console.print(f"[red]{error}[/red]")
                raise typer.Exit(code=1)

            time.sleep(1.5)

    client.close()


@app.command()
def projects(
    endpoint: str = typer.Option("http://localhost:8000/v1", help="Genesis API endpoint"),
):
    """List all projects."""
    client = httpx.Client(base_url=endpoint)
    try:
        r = client.get("/projects")
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1)

    items = data.get("items", [])
    if not items:
        console.print("[dim]No projects yet. Run 'genesis build' to create one.[/dim]")
        return

    table = Table(title="Projects", border_style="cyan")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="dim")
    table.add_column("Builds", justify="right")
    table.add_column("Created", style="dim")

    for p in items:
        status_color = {
            "active": "green",
            "building": "yellow",
            "deployed": "green",
            "failed": "red",
        }.get(p["status"], "dim")
        table.add_row(
            p["name"],
            f"[{status_color}]{p['status']}[/{status_color}]",
            str(p["build_count"]),
            p["created_at"][:10] if p.get("created_at") else "?",
        )

    console.print(table)
    client.close()


@app.command()
def status(
    build_id: str = typer.Argument(..., help="Build ID to check"),
    endpoint: str = typer.Option("http://localhost:8000/v1", help="Genesis API endpoint"),
):
    """Check build status."""
    client = httpx.Client(base_url=endpoint)
    try:
        r = client.get(f"/builds/{build_id}")
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1)

    status_color = {
        "completed": "green",
        "failed": "red",
        "queued": "dim",
    }.get(data["status"], "yellow")

    console.print(f"Status: [{status_color}]{data['status']}[/{status_color}]")
    console.print(f"Stage: {data.get('stage', 'N/A')}")
    console.print(f"Progress: {data.get('stage_progress', 0):.0%}")
    console.print(f"Retries: {data.get('retries', 0)}")

    if data.get("test_results"):
        tr = data["test_results"]
        console.print(f"Test Score: {tr.get('overall_score', 0):.0%}")
        console.print(f"Scenarios: {tr.get('scenarios_passed', 0)}/{tr.get('scenarios_run', 0)}")

    client.close()


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
):
    """Start the Genesis Engine API server."""
    from genesis.api.app import main
    import os
    os.environ["GENESIS_HOST"] = host
    os.environ["GENESIS_PORT"] = str(port)
    console.print(f"[bold cyan]Genesis Engine[/bold cyan] starting on [green]{host}:{port}[/green]")
    main()


if __name__ == "__main__":
    app()
