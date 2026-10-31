"""Optional credential storage. No provider calls, automatic migration or plaintext fallback.

Only trusted setup/managed executors may call these APIs. References are not grants.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Protocol
import uuid

_REF = re.compile(r"^[0-9a-f]{32}$")


class CredentialError(ValueError):
    """Static, safe error code; backend exceptions never cross this boundary."""


class CredentialBackend(Protocol):
    def put(self, reference: str, payload: bytes) -> None: ...
    def get(self, reference: str) -> bytes: ...
    def delete(self, reference: str) -> None: ...


def _reference(reference: str) -> str:
    if not _REF.fullmatch(reference):
        raise CredentialError("credential_reference_invalid")
    return reference


class MacOSKeychainBackend:
    """Select the OS backend directly, never user-configured keyring fallbacks."""
    service = "org.alphonse.oauth.v1"

    def _keyring(self):
        if sys.platform != "darwin":
            raise CredentialError("credential_backend_unsupported")
        from keyring.backends.macOS import Keyring
        return Keyring()

    def put(self, reference: str, payload: bytes) -> None:
        self._keyring().set_password(self.service, _reference(reference), payload.decode("utf-8"))

    def get(self, reference: str) -> bytes:
        value = self._keyring().get_password(self.service, _reference(reference))
        if value is None:
            raise CredentialError("credential_unavailable")
        return value.encode("utf-8")

    def delete(self, reference: str) -> None:
        backend = self._keyring()
        reference = _reference(reference)
        if backend.get_password(self.service, reference) is not None:
            backend.delete_password(self.service, reference)


class EncryptedFileBackend:
    """AES-256-GCM, fresh nonces, reference AAD; key provisioned separately.

    The operator provisions a 0700 root and a separate 0600 raw 32-byte key.
    Paths must be canonical, owner-only, non-symlink resources. No auto key creation.
    """
    def __init__(self, root: str | Path, key_file: str | Path):
        self.root = Path(root)
        self.key_file = Path(key_file)

    @staticmethod
    def _check(path: Path, *, directory: bool = False) -> None:
        info = path.lstat()
        if (not path.is_absolute() or path.resolve() != path or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid() or info.st_mode & 0o077
                or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
            raise CredentialError("credential_backend_permissions")

    def _cipher(self):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        self._check(self.root, directory=True)
        self._check(self.key_file)
        if self.key_file.is_relative_to(self.root):
            raise CredentialError("credential_key_not_separate")
        key = self.key_file.read_bytes()
        if len(key) != 32:
            raise CredentialError("credential_key_invalid")
        return AESGCM(key)

    def put(self, reference: str, payload: bytes) -> None:
        reference = _reference(reference)
        cipher = self._cipher()
        nonce = os.urandom(12)
        data = nonce + cipher.encrypt(nonce, payload, reference.encode("ascii"))
        path = self.root / reference
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def get(self, reference: str) -> bytes:
        reference = _reference(reference)
        cipher = self._cipher()
        path = self.root / reference
        self._check(path)
        data = path.read_bytes()
        return cipher.decrypt(data[:12], data[12:], reference.encode("ascii"))

    def delete(self, reference: str) -> None:
        # Deletion requires no decryption key (offline local disconnect).
        self._check(self.root, directory=True)
        path = self.root / _reference(reference)
        if path.exists() or path.is_symlink():
            self._check(path)
            path.unlink()


def configured_backend() -> CredentialBackend:
    backend = os.environ.get("ALPHONSE_CREDENTIAL_BACKEND", "macos" if sys.platform == "darwin" else "")
    if backend == "macos":
        return MacOSKeychainBackend()
    if backend == "encrypted-file":
        root = os.environ.get("ALPHONSE_CREDENTIAL_DIRECTORY", "")
        key = os.environ.get("ALPHONSE_CREDENTIAL_KEY_FILE", "")
        if root and key:
            return EncryptedFileBackend(root, key)
    raise CredentialError("credential_backend_unconfigured")


class CredentialVault:
    """Immutable records; SQLite reference CAS owns refresh serialization."""
    def __init__(self, backend: CredentialBackend | None = None):
        self.backend = backend

    def _backend(self):
        return self.backend if self.backend is not None else configured_backend()

    def put(self, connection_id: str, generation: int, credentials: dict[str, str]) -> str:
        if not credentials or any(not isinstance(k, str) or not isinstance(v, str) or not v
                                  for k, v in credentials.items()):
            raise CredentialError("credential_invalid")
        reference = uuid.uuid4().hex
        payload = json.dumps({"connection_id": connection_id, "generation": generation,
                              "credentials": credentials}).encode("utf-8")
        try:
            self._backend().put(reference, payload)
            if self.get(reference, connection_id, generation) != credentials:
                raise ValueError()
        except Exception:
            raise CredentialError("credential_backend_unavailable") from None
        return reference

    def get(self, reference: str, connection_id: str, generation: int) -> dict[str, str]:
        try:
            payload = json.loads(self._backend().get(_reference(reference)))
            if payload["connection_id"] != connection_id or payload["generation"] != generation:
                raise ValueError()
            credentials = payload["credentials"]
            if not isinstance(credentials, dict) or not credentials or any(
                    not isinstance(k, str) or not isinstance(v, str) or not v for k, v in credentials.items()):
                raise ValueError()
            return credentials
        except Exception:
            raise CredentialError("credential_backend_unavailable") from None

    def delete(self, reference: str) -> None:
        try:
            self._backend().delete(_reference(reference))
        except Exception:
            raise CredentialError("credential_backend_unavailable") from None

    def health(self, reference: str, connection_id: str, generation: int) -> str:
        try:
            self.get(reference, connection_id, generation)
            return "available"
        except CredentialError:
            return "credential_backend_unavailable"
