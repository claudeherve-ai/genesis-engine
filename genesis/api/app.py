"""Genesis Engine FastAPI application."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from genesis.api.routes import router
from genesis.security.rate_limit import RateLimitMiddleware, rate_limit_config
from genesis.security.auth import auth_enabled

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("genesis")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Genesis Engine starting...")
    logger.info(f"  LLM: Azure Foundry (gpt-5.4)" if os.getenv("AZURE_OPENAI_API_KEY") else "  LLM: NOT CONFIGURED")
    logger.info(f"  Database: {os.getenv('GENESIS_DB', 'sqlite:///genesis.db')}")
    logger.info(f"  Target: {os.getenv('AGENTSYSTEM_ENDPOINT', 'not configured')}")
    _rl = rate_limit_config()
    logger.info(
        "  Auth: %s | Rate limit: %s",
        "ENABLED (X-API-Key)" if auth_enabled() else "OPEN (no keys configured)",
        f"{_rl.per_minute}/min" if _rl.enabled else "disabled",
    )
    yield
    logger.info("Genesis Engine shutting down.")


app = FastAPI(
    title="Genesis Engine",
    description="Meta-agent factory — AI that builds AI. Describe a problem, get a deployed multi-agent system. MCP-grounded tools prevent hallucinations.",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS — allow all origins in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token-bucket rate limiting (lenient defaults; disable with GENESIS_RATE_LIMIT=0)
app.add_middleware(RateLimitMiddleware)

app.include_router(router)


@app.get("/")
async def root():
    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    if os.path.isfile(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return {"name": "Genesis Engine", "version": "0.4.0", "status": "running", "docs": "/docs"}


@app.get("/docs-guide")
async def docs_guide():
    guide_path = os.path.join(os.path.dirname(__file__), "..", "ui", "docs.html")
    if os.path.isfile(guide_path):
        return FileResponse(guide_path, media_type="text/html")
    return {"name": "Genesis Engine", "version": "0.4.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


def main():
    """Entry point for `genesis-server` command."""
    import uvicorn

    host = os.getenv("GENESIS_HOST", "0.0.0.0")
    port = int(os.getenv("GENESIS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
