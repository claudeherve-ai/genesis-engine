"""Enterprise security primitives for Genesis Engine.

Authentication (API keys + RBAC), rate limiting, and secret encryption.
Every component degrades gracefully so local development and the offline
test-suite keep working with zero configuration:

* Auth is **open** when ``GENESIS_API_KEYS`` is unset.
* Rate limiting uses lenient defaults and can be disabled.
* Secret encryption falls back to an ephemeral key when ``GENESIS_SECRET_KEY``
  is unset (with a warning), and can be backed by Azure Key Vault.
"""

from genesis.security.auth import (
    Principal,
    Role,
    auth_enabled,
    authenticate,
    require_admin,
    require_user,
)
from genesis.security.rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    rate_limit_config,
)
from genesis.security.secrets import (
    SecretCipher,
    decrypt_config_secrets,
    encrypt_config_secrets,
    generate_key,
    get_cipher,
)

__all__ = [
    "Principal",
    "Role",
    "auth_enabled",
    "authenticate",
    "require_admin",
    "require_user",
    "RateLimiter",
    "RateLimitMiddleware",
    "rate_limit_config",
    "SecretCipher",
    "decrypt_config_secrets",
    "encrypt_config_secrets",
    "generate_key",
    "get_cipher",
]
