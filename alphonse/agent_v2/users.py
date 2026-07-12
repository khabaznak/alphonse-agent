"""Daemon-owned canonical users, profile files, and channel addresses for v2."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

USER_CONTEXT_FILENAME = "user_context.md"


@dataclass(frozen=True)
class V2User:
    user_id: str
    display_name: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return {"user_id": self.user_id, "display_name": self.display_name, "role": self.role,
                "is_active": self.is_active, "created_at": self.created_at, "updated_at": self.updated_at}


@dataclass(frozen=True)
class UserAddress:
    address_id: str
    user_id: str
    integration_id: str
    provider_key: str
    provider_user_id: str
    channel_target: str
    is_preferred: bool
    is_active: bool

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class V2UserStore:
    """SQLite boundary that keeps v2 independent from legacy identity at runtime."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "V2UserStore":
        path = os.getenv("ALPHONSE_V2_USERS_DB_PATH") or os.getenv("ALPHONSE_V2_DB_PATH") or str(Path.home() / ".alphonse" / "v2-users.sqlite3")
        return cls(path)

    @property
    def is_ephemeral(self) -> bool:
        return self._memory is not None

    def status(self) -> dict[str, object]:
        admin = self.admin_user()
        return {"onboarded": admin is not None, "admin_user": admin.to_dict() if admin else None,
                "users_root": self.users_root()}

    def users_root(self) -> Path:
        value = self._setting("users_root")
        return Path(value) if value else Path.home() / ".alphonse" / "users"

    def set_users_root(self, root: str | Path) -> str:
        path = Path(root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._set_setting("users_root", str(path))
        for user in self.list_users():
            self.ensure_profile(user.user_id)
        return str(path)

    def onboard(self, *, display_name: str, users_root: str | Path | None = None, user_id: str | None = None) -> V2User:
        if self.admin_user() is not None:
            raise ValueError("v2_already_onboarded")
        if users_root is not None:
            self.set_users_root(users_root)
        user = self.create_user(display_name=display_name, role="admin", user_id=user_id)
        self._set_setting("admin_user_id", user.user_id)
        self._set_setting("onboarded", "1")
        return user

    def create_user(self, *, display_name: str, role: str = "member", user_id: str | None = None, is_active: bool = True) -> V2User:
        name = str(display_name or "").strip()
        if not name:
            raise ValueError("user_display_name_required")
        uid = str(user_id or uuid4())
        if uid in {"local", "admin"}:
            raise ValueError("canonical_user_id_reserved")
        now = _now()
        with self._connect() as conn:
            conn.execute("INSERT INTO v2_users (user_id, display_name, role, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (uid, name, str(role or "member"), int(bool(is_active)), now, now))
        self.ensure_profile(uid)
        return self.get_user(uid)  # type: ignore[return-value]

    def update_user(self, user_id: str, *, display_name: str | None = None, role: str | None = None, is_active: bool | None = None) -> V2User:
        current = self.get_user(user_id)
        if current is None:
            raise KeyError("user_not_found")
        with self._connect() as conn:
            conn.execute("UPDATE v2_users SET display_name=?, role=?, is_active=?, updated_at=? WHERE user_id=?", (
                str(display_name if display_name is not None else current.display_name).strip(),
                str(role if role is not None else current.role).strip() or "member",
                int(current.is_active if is_active is None else bool(is_active)), _now(), current.user_id))
        return self.get_user(user_id)  # type: ignore[return-value]

    def get_user(self, user_id: str) -> V2User | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_users WHERE user_id = ?", (str(user_id),)).fetchone()
        return _user(row)

    def list_users(self, *, active_only: bool = False) -> list[V2User]:
        query = "SELECT * FROM v2_users" + (" WHERE is_active = 1" if active_only else "") + " ORDER BY lower(display_name)"
        with self._connect() as conn:
            return [_user(row) for row in conn.execute(query).fetchall() if _user(row) is not None]  # type: ignore[list-item]

    def admin_user(self) -> V2User | None:
        admin_id = self._setting("admin_user_id")
        return self.get_user(admin_id) if admin_id else None

    def is_admin(self, user_id: str) -> bool:
        return str(user_id or "") == str(self._setting("admin_user_id") or "")

    def ensure_profile(self, user_id: str) -> Path:
        root = self.users_root() / str(user_id)
        root.mkdir(parents=True, exist_ok=True)
        context = root / USER_CONTEXT_FILENAME
        if not context.exists():
            context.write_text("# User Context\n", encoding="utf-8")
        return root

    def profile_path(self, user_id: str) -> Path:
        return self.ensure_profile(user_id) / USER_CONTEXT_FILENAME

    def read_user_context(self, user_id: str) -> str:
        if self.get_user(user_id) is None:
            return ""
        return self.profile_path(user_id).read_text(encoding="utf-8")

    def write_user_context(self, user_id: str, content: str) -> str:
        if self.get_user(user_id) is None:
            raise KeyError("user_not_found")
        path = self.profile_path(user_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(str(content or ""), encoding="utf-8")
        temporary.replace(path)
        return str(path)

    def managed_project_root(self, user_id: str) -> Path:
        return self.ensure_profile(user_id)

    def bind_address(self, *, user_id: str, integration_id: str, provider_key: str, provider_user_id: str, channel_target: str = "", is_preferred: bool = True, is_active: bool = True) -> UserAddress:
        if self.get_user(user_id) is None:
            raise KeyError("user_not_found")
        integration = str(integration_id or provider_key).strip()
        provider = str(provider_key or "").strip().lower()
        provider_user = str(provider_user_id or "").strip()
        target = str(channel_target or provider_user).strip()
        if not integration or not provider or not provider_user or not target:
            raise ValueError("channel_address_required")
        with self._connect() as conn:
            conflict = conn.execute("SELECT user_id FROM v2_user_addresses WHERE integration_id=? AND provider_user_id=?", (integration, provider_user)).fetchone()
            if conflict and str(conflict[0]) != str(user_id):
                raise ValueError("provider_user_already_mapped")
            if is_preferred:
                conn.execute("UPDATE v2_user_addresses SET is_preferred=0 WHERE user_id=?", (user_id,))
            address_id = str(uuid4())
            conn.execute("INSERT INTO v2_user_addresses (address_id,user_id,integration_id,provider_key,provider_user_id,channel_target,is_preferred,is_active,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(integration_id,provider_user_id) DO UPDATE SET user_id=excluded.user_id, provider_key=excluded.provider_key, channel_target=excluded.channel_target, is_preferred=excluded.is_preferred, is_active=excluded.is_active, updated_at=excluded.updated_at", (address_id,user_id,integration,provider,provider_user,target,int(is_preferred),int(is_active),_now(),_now()))
        return self.address_for_inbound(integration_id=integration, provider_user_id=provider_user)  # type: ignore[return-value]

    def remove_address(self, address_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM v2_user_addresses WHERE address_id=?", (str(address_id),)).rowcount > 0

    def list_addresses(self, user_id: str) -> list[UserAddress]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM v2_user_addresses WHERE user_id=? ORDER BY is_preferred DESC, created_at", (str(user_id),)).fetchall()
        return [_address(row) for row in rows if _address(row) is not None]  # type: ignore[list-item]

    def address_for_inbound(self, *, integration_id: str, provider_user_id: str) -> UserAddress | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_user_addresses WHERE integration_id=? AND provider_user_id=? AND is_active=1", (str(integration_id), str(provider_user_id))).fetchone()
        return _address(row)

    def address_for_outbound(self, user_id: str, *, integration_id: str = "") -> UserAddress | None:
        with self._connect() as conn:
            if integration_id:
                row = conn.execute("SELECT * FROM v2_user_addresses WHERE user_id=? AND integration_id=? AND is_active=1 ORDER BY is_preferred DESC LIMIT 1", (str(user_id), str(integration_id))).fetchone()
            else:
                row = conn.execute("SELECT * FROM v2_user_addresses WHERE user_id=? AND is_active=1 ORDER BY is_preferred DESC, updated_at DESC LIMIT 1", (str(user_id),)).fetchone()
        return _address(row)

    def import_v1(self) -> dict[str, object]:
        """Explicit compatibility import. v1 is never read by normal v2 routing."""
        from alphonse.agent import identity
        imported = 0
        mappings = 0
        admin = identity.get_active_admin_user()
        for legacy in identity.list_users(active_only=False):
            uid = str(legacy.get("user_id") or "").strip()
            if not uid or self.get_user(uid):
                continue
            self.create_user(display_name=str(legacy.get("display_name") or uid), role=str(legacy.get("role") or ("admin" if legacy.get("is_admin") else "member")), user_id=uid, is_active=bool(legacy.get("is_active", True)))
            imported += 1
        if self.admin_user() is None and isinstance(admin, dict) and admin.get("user_id"):
            self._set_setting("admin_user_id", str(admin["user_id"]))
            self._set_setting("onboarded", "1")
        # Service rows are intentionally copied through public v1 resolution only for known providers.
        for user in self.list_users():
            for provider in ("telegram", "teams", "whatsapp", "discord"):
                service_id = identity.resolve_service_id(provider)
                provider_user = identity.resolve_service_user_id(user_id=user.user_id, service_id=service_id) if service_id is not None else None
                if provider_user:
                    try:
                        self.bind_address(user_id=user.user_id, integration_id=provider, provider_key=provider, provider_user_id=provider_user)
                        mappings += 1
                    except ValueError:
                        pass
        return {"users_imported": imported, "addresses_imported": mappings, "admin_user_id": (self.admin_user().user_id if self.admin_user() else "")}

    def _setting(self, key: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM v2_user_settings WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else ""

    def _set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO v2_user_settings (key,value,updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at", (key, value, _now()))

    def _connect(self):
        if self._memory is not None:
            return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS v2_users (user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS v2_user_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS v2_user_addresses (address_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, integration_id TEXT NOT NULL, provider_key TEXT NOT NULL, provider_user_id TEXT NOT NULL, channel_target TEXT NOT NULL, is_preferred INTEGER NOT NULL DEFAULT 0, is_active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(integration_id, provider_user_id)) STRICT;
            CREATE INDEX IF NOT EXISTS idx_v2_user_addresses_user ON v2_user_addresses(user_id, is_preferred);
            """)


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()

def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _user(row: sqlite3.Row | None) -> V2User | None:
    return V2User(str(row["user_id"]), str(row["display_name"]), str(row["role"]), bool(row["is_active"]), str(row["created_at"]), str(row["updated_at"])) if row else None
def _address(row: sqlite3.Row | None) -> UserAddress | None:
    return UserAddress(str(row["address_id"]), str(row["user_id"]), str(row["integration_id"]), str(row["provider_key"]), str(row["provider_user_id"]), str(row["channel_target"]), bool(row["is_preferred"]), bool(row["is_active"])) if row else None
