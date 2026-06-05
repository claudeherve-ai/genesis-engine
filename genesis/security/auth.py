"""API-key authentication and role-based access control (RBAC).

Keys are configured via the ``GENESIS_API_KEYS`` environment variable using
a compact ``key:role`` syntax::

    GENESIS_API_KEYS="sk-admin-abc:admin, sk-user-def:user, sk-legacy-xyz"

A bare key (no ``:role``) defaults to the ``user`` role. Authentication is
performed against the ``X-API-Key`` request header.

**Open-when-unset:** if ``GENESIS_API_KEYS`` is empty or unset the API runs in
open mode — every request is treated as an anonymous ``admin`` principal. This
keeps local development and the offline test-suite zero-config while allowing
production to lock down by simply setting the variable.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

from fastapi import Depends, Header, HTTPException, status

API_KEY_HEADER = "X-API-Key"
_ENV_VAR = "GENESIS_API_KEYS"


class Role(IntEnum):
    """Coarse RBAC roles. Higher value ⇒ more privilege."""

    USER = 10
    ADMIN = 100

    @classmethod
    def parse(cls, raw: str) -> "Role":
        key = (raw or "").strip().lower()
        if key == "admin":
            return cls.ADMIN
        return cls.USER


@dataclass(frozen=True)
class Principal:
    """An authenticated caller."""

    key_id: str
    role: Role
    anonymous: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role >= Role.ADMIN


# Anonymous principal used in open mode — full privilege so local/dev is frictionless.
ANONYMOUS = Principal(key_id="anonymous", role=Role.ADMIN, anonymous=True)


def _load_keys() -> Dict[str, Role]:
    """Parse ``GENESIS_API_KEYS`` into a ``{key: Role}`` mapping."""
    raw = os.getenv(_ENV_VAR, "").strip()
    if not raw:
        return {}
    keys: Dict[str, Role] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            key, _, role = entry.partition(":")
            keys[key.strip()] = Role.parse(role)
        else:
            keys[entry] = Role.USER
    return keys


def auth_enabled() -> bool:
    """True when at least one API key is configured."""
    return bool(_load_keys())


def _mask(key: str) -> str:
    """Return a non-reversible, log-safe identifier for a key."""
    if len(key) <= 8:
        return "key-****"
    return f"{key[:4]}…{key[-4:]}"


def _match(provided: str, keys: Dict[str, Role]) -> Optional[Principal]:
    """Constant-time-ish lookup of ``provided`` against configured keys."""
    for candidate, role in keys.items():
        if hmac.compare_digest(provided, candidate):
            return Principal(key_id=_mask(candidate), role=role)
    return None


def authenticate(api_key: Optional[str]) -> Principal:
    """Resolve a principal from a raw API key.

    Raises ``HTTPException(401)`` when auth is enabled and the key is missing
    or invalid. Returns the anonymous admin principal in open mode.
    """
    keys = _load_keys()
    if not keys:
        return ANONYMOUS

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    principal = _match(api_key.strip(), keys)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return principal


async def _principal_dep(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
) -> Principal:
    return authenticate(x_api_key)


def require_user(principal: Principal = Depends(_principal_dep)) -> Principal:
    """FastAPI dependency: require a valid principal (any role)."""
    return principal


def require_admin(principal: Principal = Depends(_principal_dep)) -> Principal:
    """FastAPI dependency: require an admin principal."""
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation.",
        )
    return principal
