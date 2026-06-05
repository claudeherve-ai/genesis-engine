"""In-memory token-bucket rate limiting.

A lightweight, dependency-free limiter suitable for single-process
deployments and as a safety net in front of more sophisticated gateways.
Each identity (API key when present, otherwise client IP) gets its own
bucket that refills continuously at ``rate`` tokens per second up to
``capacity`` (the burst allowance).

Configuration (all optional, lenient defaults):

* ``GENESIS_RATE_LIMIT``  — sustained requests per minute (default ``600``).
                            Set to ``0`` to disable rate limiting entirely.
* ``GENESIS_RATE_BURST``  — burst capacity (default ``rate_per_minute``).

Paths in :data:`EXEMPT_PATHS` (health checks, root, docs) are never limited.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from genesis.security.auth import API_KEY_HEADER

EXEMPT_PATHS = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"})


@dataclass(frozen=True)
class RateLimitConfig:
    """Resolved rate-limit settings."""

    per_minute: int
    burst: int

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    @property
    def refill_per_second(self) -> float:
        return self.per_minute / 60.0


def rate_limit_config() -> RateLimitConfig:
    """Read rate-limit configuration from the environment."""
    try:
        per_minute = int(os.getenv("GENESIS_RATE_LIMIT", "600"))
    except ValueError:
        per_minute = 600
    per_minute = max(0, per_minute)

    try:
        burst = int(os.getenv("GENESIS_RATE_BURST", str(per_minute)))
    except ValueError:
        burst = per_minute
    burst = max(1, burst) if per_minute > 0 else 0

    return RateLimitConfig(per_minute=per_minute, burst=burst)


class RateLimiter:
    """Thread-safe token-bucket limiter keyed by identity."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, Tuple[float, float]] = {}  # id -> (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, identity: str, *, now: float | None = None) -> bool:
        """Consume one token for ``identity``. Returns False when exhausted."""
        if not self.config.enabled:
            return True

        now = time.monotonic() if now is None else now
        capacity = float(self.config.burst)
        refill = self.config.refill_per_second

        with self._lock:
            tokens, last = self._buckets.get(identity, (capacity, now))
            tokens = min(capacity, tokens + (now - last) * refill)
            if tokens < 1.0:
                self._buckets[identity] = (tokens, now)
                return False
            self._buckets[identity] = (tokens - 1.0, now)
            return True

    def retry_after(self) -> int:
        """Seconds a client should wait before retrying (best effort)."""
        if not self.config.enabled or self.config.refill_per_second <= 0:
            return 1
        return max(1, int(round(1.0 / self.config.refill_per_second)))


def _identity(request: Request) -> str:
    """Derive a stable rate-limit identity for a request."""
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return f"key:{api_key}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing the token bucket per identity."""

    def __init__(self, app, config: RateLimitConfig | None = None):
        super().__init__(app)
        self.config = config or rate_limit_config()
        self.limiter = RateLimiter(self.config)

    async def dispatch(self, request: Request, call_next):
        if not self.config.enabled or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        identity = _identity(request)
        if not self.limiter.allow(identity):
            retry = self.limiter.retry_after()
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Slow down and retry shortly.",
                    "retry_after_seconds": retry,
                },
                headers={"Retry-After": str(retry)},
            )
        return await call_next(request)
