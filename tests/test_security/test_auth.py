"""Tests for API-key authentication and RBAC."""

import pytest
from fastapi import HTTPException

from genesis.security import auth


def test_open_mode_returns_anonymous_admin(monkeypatch):
    monkeypatch.delenv("GENESIS_API_KEYS", raising=False)
    assert auth.auth_enabled() is False
    principal = auth.authenticate(None)
    assert principal.anonymous is True
    assert principal.is_admin is True


def test_enabled_when_keys_present(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-user-abc:user")
    assert auth.auth_enabled() is True


def test_missing_key_rejected_when_enabled(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-user-abc:user")
    with pytest.raises(HTTPException) as exc:
        auth.authenticate(None)
    assert exc.value.status_code == 401


def test_bad_key_rejected(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-user-abc:user")
    with pytest.raises(HTTPException) as exc:
        auth.authenticate("wrong-key")
    assert exc.value.status_code == 401


def test_valid_user_key(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-user-abc:user, sk-admin-xyz:admin")
    principal = auth.authenticate("sk-user-abc")
    assert principal.role == auth.Role.USER
    assert principal.is_admin is False
    # key id is masked, never the raw key
    assert "sk-user-abc" != principal.key_id


def test_valid_admin_key(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-admin-xyz:admin")
    principal = auth.authenticate("sk-admin-xyz")
    assert principal.role == auth.Role.ADMIN
    assert principal.is_admin is True


def test_bare_key_defaults_to_user(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-legacy-key")
    principal = auth.authenticate("sk-legacy-key")
    assert principal.role == auth.Role.USER


def test_require_admin_rejects_user(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-user-abc:user")
    principal = auth.authenticate("sk-user-abc")
    with pytest.raises(HTTPException) as exc:
        auth.require_admin(principal)
    assert exc.value.status_code == 403


def test_require_admin_allows_admin(monkeypatch):
    monkeypatch.setenv("GENESIS_API_KEYS", "sk-admin-xyz:admin")
    principal = auth.authenticate("sk-admin-xyz")
    assert auth.require_admin(principal) is principal


def test_require_user_passthrough_open_mode(monkeypatch):
    monkeypatch.delenv("GENESIS_API_KEYS", raising=False)
    principal = auth.authenticate(None)
    assert auth.require_user(principal) is principal
