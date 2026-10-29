"""Actor-scoped household bindings, separate from portable skill packages.

This is a managed access boundary, not a sandbox for trusted host code.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(repr=False)
class ExecutionBindings:
    """Ephemeral trusted-executor payload. Never put values in task evidence."""

    values: dict[str, Any] = field(repr=False)
    secret_values: tuple[str, ...] = field(repr=False)

    def __repr__(self) -> str:
        return "ExecutionBindings(<private>)"

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            tokens = set(self.secret_values)
            tokens.update(json.dumps(token, ensure_ascii=True)[1:-1] for token in self.secret_values)
            tokens.update(json.dumps(token, ensure_ascii=False)[1:-1] for token in self.secret_values)
            tokens.update(repr(token)[1:-1] for token in self.secret_values)
            for token in sorted(tokens, key=len, reverse=True):
                if token:
                    value = value.replace(token, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self.redact(str(key)): self.redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        return value


class SkillBindings:
    """Persist references/ACLs in the existing integration DB; never copy secrets.

    Integration records are household/admin managed and have no actor ownership.
    Only an active administrator can provision a binding. Owners may share/revoke
    it afterward. Consumers resolve again immediately before use.
    """

    def __init__(self, integrations: SQLiteIntegrationStore, users: V2UserStore, skills: SkillStore):
        self.integrations = integrations
        self.users = users
        self.skills = skills
        with integrations._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS v2_skill_bindings (
                    binding_id TEXT PRIMARY KEY,
                    skill_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('account', 'data', 'configuration')),
                    target TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    requirements TEXT NOT NULL,
                    shared_with TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS v2_skill_binding_choices (
                    actor_id TEXT NOT NULL,
                    skill_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    binding_id TEXT NOT NULL,
                    PRIMARY KEY(actor_id, skill_id, name)
                );
            """)

    def _actor(self, actor_id: str) -> None:
        user = self.users.get_user(actor_id)
        if user is None or not user.is_active:
            raise PermissionError("skill_binding_actor_denied")

    def configure(self, *, actor_id: str, binding_id: str, skill_id: str, owner_id: str,
                  kind: str, target: str = "", configuration: dict[str, Any] | None = None,
                  required_secrets: tuple[str, ...] = (), executables: tuple[str, ...] = ()) -> None:
        """Trusted admin setup; configuration MUST be non-secret. No file writes."""
        self._actor(actor_id)
        if not self.users.is_admin(actor_id):
            raise PermissionError("skill_binding_admin_required")
        self._actor(owner_id)
        if not _NAME.fullmatch(binding_id):
            raise ValueError("skill_binding_id_invalid")
        skill = self.skills.get(skill_id)
        if skill is None:
            raise ValueError("skill_binding_skill_missing")
        if kind not in {"account", "data", "configuration"}:
            raise ValueError("skill_binding_kind_invalid")
        if kind == "account":
            account = self.integrations.get(target)
            if account is None or not account.enabled:
                raise ValueError("skill_binding_account_unavailable")
        elif kind == "data":
            path = Path(target).expanduser()
            if not path.is_absolute():
                raise ValueError("skill_binding_data_absolute_required")
            try:
                path = path.resolve(strict=True)
            except OSError:
                raise ValueError("skill_binding_data_missing") from None
            if not path.is_file() and not path.is_dir():
                raise ValueError("skill_binding_data_missing")
            self._outside_packages(path)
            target = str(path)
        elif target or required_secrets:
            raise ValueError("skill_binding_configuration_target_invalid")
        if required_secrets and kind != "account":
            raise ValueError("skill_binding_account_requirements_invalid")
        if any(not isinstance(item, str) or not item for item in (*required_secrets, *executables)):
            raise ValueError("skill_binding_requirements_invalid")
        requirements = {"secrets": list(required_secrets), "executables": list(executables)}
        with self.integrations._connect() as conn:
            if conn.execute("SELECT 1 FROM v2_skill_bindings WHERE binding_id = ?", (binding_id,)).fetchone():
                raise ValueError("skill_binding_already_exists")
            conn.execute("""INSERT INTO v2_skill_bindings
                (binding_id, skill_id, owner_id, kind, target, configuration, requirements)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (binding_id, skill.skill_id, owner_id, kind, target,
                 json.dumps(configuration or {}), json.dumps(requirements)))

    def _outside_packages(self, path: Path) -> None:
        roots = [self.skills.skills_dir.resolve(), self.skills._bundled_skills_dir().resolve()]
        if any(path.is_relative_to(root) or root.is_relative_to(path) for root in roots):
            raise ValueError("skill_binding_data_inside_package")

    def set_access(self, binding_id: str, *, actor_id: str, shared_with: tuple[str, ...] = (),
                   enabled: bool = True) -> None:
        self._actor(actor_id)
        row = self._get(binding_id)
        if row["owner_id"] != actor_id and not self.users.is_admin(actor_id):
            raise PermissionError("skill_binding_access_denied")
        for user_id in shared_with:
            self._actor(user_id)
        with self.integrations._connect() as conn:
            conn.execute("UPDATE v2_skill_bindings SET shared_with = ?, enabled = ? WHERE binding_id = ?",
                         (json.dumps(sorted(set(shared_with))), int(enabled), binding_id))

    def _get(self, binding_id: str) -> Any:
        with self.integrations._connect() as conn:
            row = conn.execute("SELECT * FROM v2_skill_bindings WHERE binding_id = ?", (binding_id,)).fetchone()
        if row is None:
            raise ValueError("skill_binding_unavailable")
        return row

    def _authorize(self, row: Any, actor_id: str) -> None:
        self._actor(actor_id)
        self._actor(row["owner_id"])
        if not row["enabled"]:
            raise PermissionError("skill_binding_revoked")
        if (actor_id != row["owner_id"] and actor_id not in json.loads(row["shared_with"])
                and not self.users.is_admin(actor_id)):
            raise PermissionError("skill_binding_access_denied")

    def select(self, *, actor_id: str, skill_id: str, name: str, binding_id: str) -> None:
        """Explicit member choice for a logical role; no provider/owner fallback."""
        if not _NAME.fullmatch(name):
            raise ValueError("skill_binding_name_invalid")
        row = self._get(binding_id)
        self._authorize(row, actor_id)
        if row["skill_id"] != skill_id:
            raise ValueError("skill_binding_skill_mismatch")
        with self.integrations._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO v2_skill_binding_choices VALUES (?, ?, ?, ?)",
                         (actor_id, skill_id, name, binding_id))

    def prepare(self, *, actor_id: str, skill_id: str, names: list[str]) -> ExecutionBindings:
        """For trusted native execution only; never expose the payload to an LLM."""
        self._actor(actor_id)
        if self.skills.get(skill_id) is None:
            raise ValueError("skill_binding_skill_missing")
        values: dict[str, Any] = {}
        secrets: list[str] = []
        for name in names:
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                raise ValueError("skill_binding_name_invalid")
            with self.integrations._connect() as conn:
                choice = conn.execute("""SELECT binding_id FROM v2_skill_binding_choices
                    WHERE actor_id = ? AND skill_id = ? AND name = ?""", (actor_id, skill_id, name)).fetchone()
            if choice is None:
                raise ValueError("skill_binding_choice_missing")
            row = self._get(choice["binding_id"])
            self._authorize(row, actor_id)
            if row["skill_id"] != skill_id:
                raise ValueError("skill_binding_skill_mismatch")
            requirements = json.loads(row["requirements"])
            for executable in requirements["executables"]:
                if not shutil.which(executable):
                    raise ValueError("skill_binding_dependency_missing")
            value = {"kind": row["kind"], "configuration": json.loads(row["configuration"])}
            if row["kind"] == "account":
                account = self.integrations.get(row["target"])
                if account is None or not account.enabled:
                    raise ValueError("skill_binding_account_unavailable")
                if any(not isinstance(item, str) for item in account.secrets.values()):
                    raise ValueError("skill_binding_credential_invalid")
                if any(not account.secrets.get(key) for key in requirements["secrets"]):
                    raise ValueError("skill_binding_credential_missing")
                value.update(provider=account.provider_key, configuration={**account.config, **value["configuration"]},
                             credentials={key: account.secrets[key] for key in requirements["secrets"]})
                secrets.extend(item for item in account.secrets.values() if item)
            elif row["kind"] == "data":
                path = Path(row["target"])
                if not path.exists() or path.resolve() != path:
                    raise ValueError("skill_binding_data_missing_or_changed")
                self._outside_packages(path)
                value["path"] = str(path)
            values[name] = value
        return ExecutionBindings(values, tuple(secrets))

    def use(self, *, actor_id: str, skill_id: str, names: list[str],
            operation: Callable[[dict[str, Any]], Any]) -> Any:
        """Native adapter: check at use, sanitize result/error before return."""
        prepared = self.prepare(actor_id=actor_id, skill_id=skill_id, names=names)
        try:
            result = operation(prepared.values)
            # Native adapters return JSON evidence, never custom objects with an
            # unredacted __repr__ or exception-producing serializer downstream.
            result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
        except Exception:
            raise RuntimeError("skill_binding_operation_failed") from None
        return prepared.redact(result)
