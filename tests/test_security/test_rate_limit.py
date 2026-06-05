"""Tests for the token-bucket rate limiter."""

from genesis.security.rate_limit import (
    RateLimitConfig,
    RateLimiter,
    rate_limit_config,
)


def test_disabled_when_zero(monkeypatch):
    monkeypatch.setenv("GENESIS_RATE_LIMIT", "0")
    cfg = rate_limit_config()
    assert cfg.enabled is False
    limiter = RateLimiter(cfg)
    # Always allowed when disabled.
    assert all(limiter.allow("ip:1.2.3.4") for _ in range(1000))


def test_default_is_lenient(monkeypatch):
    monkeypatch.delenv("GENESIS_RATE_LIMIT", raising=False)
    cfg = rate_limit_config()
    assert cfg.per_minute == 600
    assert cfg.enabled is True


def test_allows_within_capacity():
    cfg = RateLimitConfig(per_minute=60, burst=5)
    limiter = RateLimiter(cfg)
    # Freeze time so no refill happens between calls.
    allowed = [limiter.allow("id", now=100.0) for _ in range(5)]
    assert all(allowed)


def test_denies_when_exhausted():
    cfg = RateLimitConfig(per_minute=60, burst=3)
    limiter = RateLimiter(cfg)
    for _ in range(3):
        assert limiter.allow("id", now=100.0) is True
    # Fourth request in the same instant is denied.
    assert limiter.allow("id", now=100.0) is False


def test_refills_over_time():
    cfg = RateLimitConfig(per_minute=60, burst=1)  # 1 token/sec
    limiter = RateLimiter(cfg)
    assert limiter.allow("id", now=0.0) is True
    assert limiter.allow("id", now=0.0) is False
    # One second later, exactly one token has refilled.
    assert limiter.allow("id", now=1.0) is True


def test_identities_are_independent():
    cfg = RateLimitConfig(per_minute=60, burst=1)
    limiter = RateLimiter(cfg)
    assert limiter.allow("a", now=0.0) is True
    assert limiter.allow("b", now=0.0) is True
    assert limiter.allow("a", now=0.0) is False


def test_burst_invalid_env(monkeypatch):
    monkeypatch.setenv("GENESIS_RATE_LIMIT", "not-a-number")
    cfg = rate_limit_config()
    assert cfg.per_minute == 600
