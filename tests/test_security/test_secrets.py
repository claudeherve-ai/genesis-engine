"""Tests for Fernet secret encryption in the deploy path."""

from genesis.security.secrets import (
    ENCRYPTED_PREFIX,
    SecretCipher,
    decrypt_config_secrets,
    encrypt_config_secrets,
    generate_key,
)


def _cipher() -> SecretCipher:
    return SecretCipher(generate_key())


def test_round_trip():
    cipher = _cipher()
    token = cipher.encrypt("super-secret")
    assert token.startswith(ENCRYPTED_PREFIX)
    assert token != "super-secret"
    assert cipher.decrypt(token) == "super-secret"


def test_is_encrypted_marker():
    cipher = _cipher()
    token = cipher.encrypt("value")
    assert SecretCipher.is_encrypted(token) is True
    assert SecretCipher.is_encrypted("value") is False
    assert SecretCipher.is_encrypted(None) is False


def test_double_encrypt_is_idempotent():
    cipher = _cipher()
    once = cipher.encrypt("value")
    twice = cipher.encrypt(once)
    assert once == twice


def test_decrypt_passes_through_plaintext():
    cipher = _cipher()
    assert cipher.decrypt("plain") == "plain"


def test_encrypt_config_only_touches_secret_keys():
    cipher = _cipher()
    config = {
        "endpoint": "https://example.com",
        "api_key": "sk-123",
        "auth_token": "tok-xyz",
        "region": "eastus",
        "password": "hunter2",
    }
    out = encrypt_config_secrets(config, cipher=cipher)
    assert out["endpoint"] == "https://example.com"
    assert out["region"] == "eastus"
    assert SecretCipher.is_encrypted(out["api_key"])
    assert SecretCipher.is_encrypted(out["auth_token"])
    assert SecretCipher.is_encrypted(out["password"])


def test_config_round_trip():
    cipher = _cipher()
    config = {"endpoint": "https://e", "api_key": "sk-123", "token": "t"}
    enc = encrypt_config_secrets(config, cipher=cipher)
    dec = decrypt_config_secrets(enc, cipher=cipher)
    assert dec == config


def test_empty_config():
    assert encrypt_config_secrets({}) == {}
    assert decrypt_config_secrets({}) == {}


def test_explicit_secret_keys():
    cipher = _cipher()
    config = {"weird_field": "value", "normal": "x"}
    out = encrypt_config_secrets(config, secret_keys=["weird_field"], cipher=cipher)
    assert SecretCipher.is_encrypted(out["weird_field"])
    assert out["normal"] == "x"


def test_non_string_secret_left_alone():
    cipher = _cipher()
    config = {"api_key": 12345}
    out = encrypt_config_secrets(config, cipher=cipher)
    assert out["api_key"] == 12345
