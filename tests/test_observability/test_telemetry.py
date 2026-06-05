"""Tests for the optional OpenTelemetry tracing shim."""

from genesis.observability.telemetry import get_tracer, traced


def test_get_tracer_returns_usable_object(monkeypatch):
    monkeypatch.delenv("GENESIS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    tracer = get_tracer("test")
    # No-op tracer still supports the span context-manager protocol.
    with tracer.start_as_current_span("unit") as span:
        assert span is not None


async def test_traced_async_passthrough():
    @traced("my-async-span")
    async def work(x):
        return x * 2

    assert await work(21) == 42


def test_traced_sync_passthrough():
    @traced()
    def work(x):
        return x + 1

    assert work(1) == 2


def test_traced_preserves_exceptions():
    @traced("boom")
    def work():
        raise ValueError("nope")

    try:
        work()
        assert False, "should have raised"
    except ValueError as exc:
        assert str(exc) == "nope"
