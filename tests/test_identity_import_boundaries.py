from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHONSE_ROOT = REPO_ROOT / "alphonse"

ALLOWED_USERS_IMPORTERS: set[str] = set()

ALLOWED_RESOLVER_IMPORTERS = {
    "alphonse/agent/nervous_system/telegram_chat_access.py",
    "alphonse/agent/io/telegram_channel.py",
}


def test_only_identity_layer_imports_users_store_directly() -> None:
    offenders = _find_direct_importers("alphonse.agent.nervous_system.users")
    assert offenders == sorted(ALLOWED_USERS_IMPORTERS)


def test_only_identity_or_provider_local_modules_import_resolver_store_directly() -> None:
    offenders = _find_direct_importers("alphonse.agent.nervous_system.user_service_resolvers")
    assert offenders == sorted(ALLOWED_RESOLVER_IMPORTERS)


def _find_direct_importers(target_module: str) -> list[str]:
    matches: list[str] = []
    for path in ALPHONSE_ROOT.rglob("*.py"):
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _imports_target(tree, target_module):
            matches.append(rel_path)
    return sorted(matches)


def _imports_target(tree: ast.AST, target_module: str) -> bool:
    package_name, _, symbol_name = target_module.rpartition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module == target_module:
                return True
            if module == package_name:
                for alias in node.names:
                    if str(alias.name or "") == symbol_name:
                        return True
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if str(alias.name or "") == target_module:
                    return True
    return False
