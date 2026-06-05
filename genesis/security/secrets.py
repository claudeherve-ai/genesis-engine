"""Secret encryption for the deployment path.

Deployment target configuration frequently carries credentials (API keys,
connection strings, tokens). Genesis encrypts these at rest with
`Fernet <https://cryptography.io/en/latest/fernet/>`_ symmetric encryption so
that persisted builds never store plaintext secrets.

Key resolution order:

1. ``GENESIS_SECRET_KEY`` — a urlsafe-base64 32-byte Fernet key (recommended;
   generate one with :func:`generate_key`).
2. If unset, an **ephemeral** key is generated for the process lifetime and a
   warning is logged. Encryption still works, but secrets cannot be decrypted
   after a restart — acceptable for local/dev, never for production.

An optional Azure Key Vault hook (:class:`KeyVaultSecretStore`) is provided for
teams that prefer managed secret storage; it is only imported on demand so the
dependency stays optional.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("genesis.security.secrets")

# Marker prefix so we can distinguish already-encrypted values and detect
# which config keys hold secrets when round-tripping.
ENCRYPTED_PREFIX = "enc::"

# Heuristic: config keys whose name contains any of these tokens are secrets.
_SECRET_HINTS = ("key", "secret", "token", "password", "passwd", "credential", "connection_string")

_ENV_VAR = "GENESIS_SECRET_KEY"


def generate_key() -> str:
    """Generate a new urlsafe-base64 Fernet key as a string."""
    return Fernet.generate_key().decode("utf-8")


class SecretCipher:
    """Thin wrapper over Fernet with marker-aware encrypt/decrypt."""

    def __init__(self, key: str | bytes | None = None, *, ephemeral: bool = False):
        if key is None:
            key = Fernet.generate_key()
            ephemeral = True
        if isinstance(key, str):
            key = key.encode("utf-8")
        self._fernet = Fernet(key)
        self.ephemeral = ephemeral

    def encrypt(self, plaintext: str) -> str:
        """Encrypt ``plaintext`` and return a marker-prefixed token."""
        if plaintext is None:
            return plaintext  # type: ignore[return-value]
        if self.is_encrypted(plaintext):
            return plaintext
        token = self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")
        return f"{ENCRYPTED_PREFIX}{token}"

    def decrypt(self, value: str) -> str:
        """Decrypt a marker-prefixed token; pass through plaintext unchanged."""
        if not self.is_encrypted(value):
            return value
        token = value[len(ENCRYPTED_PREFIX):]
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:  # pragma: no cover - defensive
            raise ValueError("Unable to decrypt secret: invalid token or wrong key.") from exc

    @staticmethod
    def is_encrypted(value: Any) -> bool:
        return isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)


_cipher: SecretCipher | None = None


def get_cipher() -> SecretCipher:
    """Return the process-wide cipher, building it from the environment once."""
    global _cipher
    if _cipher is not None:
        return _cipher

    key = os.getenv(_ENV_VAR, "").strip()
    if key:
        _cipher = SecretCipher(key)
    else:
        logger.warning(
            "%s is not set — using an EPHEMERAL encryption key. Secrets cannot be "
            "decrypted after a restart. Set %s for persistent encryption.",
            _ENV_VAR,
            _ENV_VAR,
        )
        _cipher = SecretCipher(ephemeral=True)
    return _cipher


def _looks_secret(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def encrypt_config_secrets(
    config: Dict[str, Any],
    *,
    secret_keys: Iterable[str] | None = None,
    cipher: SecretCipher | None = None,
) -> Dict[str, Any]:
    """Return a copy of ``config`` with secret-looking string values encrypted.

    By default keys are detected heuristically (anything containing ``key``,
    ``secret``, ``token``, ``password``, ``credential`` …). Pass ``secret_keys``
    to encrypt an explicit allow-list instead.
    """
    if not config:
        return dict(config or {})
    cipher = cipher or get_cipher()
    explicit = set(secret_keys) if secret_keys is not None else None

    out: Dict[str, Any] = {}
    for name, value in config.items():
        is_secret = name in explicit if explicit is not None else _looks_secret(name)
        if is_secret and isinstance(value, str) and value:
            out[name] = cipher.encrypt(value)
        else:
            out[name] = value
    return out


def decrypt_config_secrets(
    config: Dict[str, Any],
    *,
    cipher: SecretCipher | None = None,
) -> Dict[str, Any]:
    """Return a copy of ``config`` with any encrypted values decrypted."""
    if not config:
        return dict(config or {})
    cipher = cipher or get_cipher()
    out: Dict[str, Any] = {}
    for name, value in config.items():
        if SecretCipher.is_encrypted(value):
            out[name] = cipher.decrypt(value)
        else:
            out[name] = value
    return out


class KeyVaultSecretStore:
    """Optional Azure Key Vault backend for managed secret storage.

    Imported lazily so ``azure-keyvault-secrets`` stays an optional runtime
    dependency. Construct with the vault URL or set ``AZURE_KEY_VAULT_URL``.
    """

    def __init__(self, vault_url: str | None = None):
        self.vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL", "").strip()
        if not self.vault_url:
            raise ValueError("AZURE_KEY_VAULT_URL is required for KeyVaultSecretStore.")
        from azure.identity import DefaultAzureCredential  # noqa: WPS433 (lazy import)
        from azure.keyvault.secrets import SecretClient  # noqa: WPS433

        self._client = SecretClient(
            vault_url=self.vault_url, credential=DefaultAzureCredential()
        )

    def set_secret(self, name: str, value: str) -> None:
        self._client.set_secret(name, value)

    def get_secret(self, name: str) -> str:
        return self._client.get_secret(name).value
