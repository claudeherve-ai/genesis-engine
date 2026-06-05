"""Smoke tests for the observability metrics endpoints."""

from fastapi.testclient import TestClient

from genesis.api.app import app
from genesis.observability.cost import get_cost_tracker


def _client(monkeypatch):
    # Keep auth open so the public metrics endpoints are reachable.
    monkeypatch.delenv("GENESIS_API_KEYS", raising=False)
    monkeypatch.delenv("GENESIS_RATE_LIMIT", raising=False)
    return TestClient(app)


def test_metrics_endpoint(monkeypatch):
    tracker = get_cost_tracker()
    tracker.reset()
    tracker.record("gpt-4o", 1000, 500, stage="BUILD", build_id="bX")
    with _client(monkeypatch) as client:
        resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"]["calls"] >= 1
    assert "by_model" in body
    assert "by_stage" in body
    assert "budget" in body


def test_build_metrics_endpoint(monkeypatch):
    tracker = get_cost_tracker()
    tracker.reset()
    tracker.record("gpt-4o", 100, 50, stage="ANALYZE", build_id="bMetrics")
    with _client(monkeypatch) as client:
        resp = client.get("/v1/builds/bMetrics/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["build"]["calls"] == 1
    assert body["build"]["prompt_tokens"] == 100
