"""Optional OpenTelemetry tracing with a zero-dependency no-op fallback.

If ``opentelemetry`` is installed *and* tracing is enabled via
``GENESIS_OTEL_ENABLED=1`` (or an ``OTEL_EXPORTER_OTLP_ENDPOINT`` is present),
spans are emitted through the global tracer. Otherwise every tracing call is a
cheap no-op, so the rest of the codebase can instrument freely without caring
whether OTel is available.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator


def _otel_enabled() -> bool:
    if os.getenv("GENESIS_OTEL_ENABLED", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


class _NoopSpan:
    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: D401
        return None

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_status(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NoopTracer:
    """Tracer that produces no spans but mimics the OTel API surface."""

    @contextmanager
    def start_as_current_span(self, _name: str, **_kwargs: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


_tracer: Any = None


def get_tracer(name: str = "genesis") -> Any:
    """Return a real OTel tracer when available/enabled, else a no-op tracer."""
    global _tracer
    if _tracer is not None:
        return _tracer

    if _otel_enabled():
        try:  # pragma: no cover - exercised only with OTel installed
            from opentelemetry import trace

            _tracer = trace.get_tracer(name)
            return _tracer
        except Exception:  # pragma: no cover - graceful fallback
            pass

    _tracer = _NoopTracer()
    return _tracer


def traced(span_name: str | None = None) -> Callable:
    """Decorator that wraps a sync/async function in a span."""

    def decorator(func: Callable) -> Callable:
        name = span_name or func.__qualname__

        if _is_coroutine(func):

            @wraps(func)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with get_tracer().start_as_current_span(name):
                    return await func(*args, **kwargs)

            return awrapper

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with get_tracer().start_as_current_span(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _is_coroutine(func: Callable) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)
