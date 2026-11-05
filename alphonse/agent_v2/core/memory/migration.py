"""Offline, additive memory migration and verified recovery copies.

No provider, extractor, runtime, or household integration is loaded here. Stop
writers before taking a database/filesystem checkpoint or applying a migration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from .daily_ledger import DailyLedgerProjector

VERSION = 1
RECEIPT = ".memory-migration-v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained(root: Path, relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError("memory_migration_invalid_path")
    path = root / name
    current = root
    for part in ("", *name.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError("memory_migration_symlink")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("memory_migration_path_escape")
    return path


def migrate_scope(root: Path, *, owner: str, project: str, apply: bool = False) -> dict:
    """Copy audits once; index summaries as unverified reconciliation candidates.

    A receipt is written last. Identical copies left by interruption are adopted;
    changed sources or collisions require reconciliation rather than guessing.
    Existing canonical events, knowledge and daily views are never rewritten.
    """
    root = Path(root).absolute()
    if not owner.strip() or not root.is_dir():
        raise ValueError("memory_migration_scope_required")
    receipt_path = _contained(root, RECEIPT)
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        "version": VERSION, "owner": owner, "project": project, "files": [],
    }
    if (receipt.get("version"), receipt.get("owner"), receipt.get("project")) != (VERSION, owner, project):
        raise ValueError("memory_migration_receipt_mismatch")
    if (not isinstance(receipt.get("files"), list)
            or any(not isinstance(entry, dict) for entry in receipt["files"])):
        raise ValueError("memory_migration_invalid_receipt")
    existing = {entry["source"]: entry for entry in receipt["files"]}
    if len(existing) != len(receipt["files"]):
        raise ValueError("memory_migration_duplicate_receipt_source")
    # Check the entire previous receipt before making any new copies.
    for entry in existing.values():
        for name in {entry["source"], entry["resource"]}:
            path = _contained(root, name)
            if not path.is_file() or _digest(path) != entry["sha256"]:
                raise ValueError("memory_migration_reconciliation_required:" + name)
    paths = sorted({*root.glob("ledger-*.md"), *root.glob("legacy/ledger-*.md"),
                    *root.glob("project-summary.md"), *root.glob("sessions/*/summary.md")})
    added = []
    copies = []
    known_resources = {entry["resource"] for entry in existing.values()}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        _contained(root, relative)
        if relative in existing or relative in known_resources:
            continue
        digest = _digest(path)
        is_audit = path.name.startswith("ledger-")
        resource = "legacy/" + path.name if path.parent == root and is_audit else relative
        target = _contained(root, resource)
        if target.exists() and _digest(target) != digest:
            raise ValueError("memory_migration_archive_conflict:" + resource)
        identity = json.dumps([VERSION, owner, project, relative, digest], ensure_ascii=False)
        entry = {"source": relative, "resource": resource, "sha256": digest,
                 "candidate_id": "legacy-candidate:" + hashlib.sha256(identity.encode()).hexdigest(),
                 "owner_user_id": owner, "visibility": "private", "basis": "legacy_unverified",
                 "kind": "audit" if is_audit else "derived_summary", "requires_reconciliation": True}
        added.append(entry)
        known_resources.add(resource)
        if resource != relative and not target.exists():
            copies.append((path, target))
    result = {**receipt, "files": sorted([*existing.values(), *added], key=lambda item: item["source"])}
    if apply and (added or not receipt_path.exists()):
        for source, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents another migrator from overwriting data.
            try:
                with target.open("xb") as output:
                    output.write(source.read_bytes())
            except FileExistsError:
                if _digest(target) != _digest(source):
                    raise ValueError("memory_migration_archive_conflict:" + target.name)
        for entry in result["files"]:
            if any(_digest(_contained(root, name)) != entry["sha256"]
                   for name in {entry["source"], entry["resource"]}):
                raise ValueError("memory_migration_sources_changed")
        DailyLedgerProjector._atomic_write(receipt_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {**result, "dry_run": not apply, "added": len(added), "copied_audits": len(copies)}


def _inventory(root: Path) -> dict[str, str]:
    if root.is_symlink():
        raise ValueError("memory_migration_symlink")
    result = {}
    for path in sorted(root.rglob("*")):
        _contained(root, path.relative_to(root).as_posix())
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _digest(path)
        elif not path.is_dir():
            raise ValueError("memory_backup_unsupported_file")
    return result


def _new_destination(destination: Path, sources: list[Path]) -> Path:
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("memory_recovery_destination_exists")
    resolved = destination.resolve()
    if any(resolved.is_relative_to(source.resolve()) for source in sources):
        raise ValueError("memory_recovery_destination_inside_source")
    return destination


def backup(database: Path, roots: list[Path], destination: Path) -> dict:
    """Create a versioned checkpoint, using SQLite backup to include WAL data.

    Call only with the daemon and other writers stopped. Explicit roots must
    include user memory and every project memory directory in the rollout.
    """
    database, roots = Path(database).absolute(), [Path(root).absolute() for root in roots]
    if not database.is_file() or database.is_symlink() or any(not root.is_dir() for root in roots):
        raise ValueError("memory_backup_source_missing")
    destination = _new_destination(Path(destination), [database, *roots])
    before = [_inventory(root) for root in roots]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete verified backups. Temporary state contains private data.
    with tempfile.TemporaryDirectory(prefix=".memory-backup-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(staging / "database.sqlite3") as target:
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("memory_backup_database_integrity")
        for index, root in enumerate(roots):
            shutil.copytree(root, staging / "roots" / str(index))
            if _inventory(root) != before[index] or _inventory(staging / "roots" / str(index)) != before[index]:
                raise ValueError("memory_backup_sources_changed")
        manifest = {"version": VERSION, "database_source": str(database),
                    "root_sources": [str(root) for root in roots], "files": _inventory(staging)}
        (staging / "backup.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging.rename(destination)
    return manifest


def restore(checkpoint: Path, destination: Path) -> dict:
    """Verify and restore into a new recovery directory, never over live state."""
    checkpoint = Path(checkpoint).absolute()
    _contained(checkpoint, "backup.json")
    manifest = json.loads((checkpoint / "backup.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("memory_backup_unsupported_version")
    verified = _inventory(checkpoint)
    if {name: digest for name, digest in verified.items() if name != "backup.json"} != manifest.get("files"):
        raise ValueError("memory_backup_integrity_failed")
    destination = _new_destination(Path(destination), [checkpoint])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memory-restore-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "recovery"
        shutil.copytree(checkpoint, staging)
        if _inventory(staging) != verified or _inventory(checkpoint) != verified:
            raise ValueError("memory_backup_changed_during_restore")
        staging.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser("migrate")
    migrate.add_argument("--root", type=Path, required=True)
    migrate.add_argument("--owner", required=True)
    migrate.add_argument("--project", default="")
    migrate.add_argument("--apply", action="store_true")
    snapshot = commands.add_parser("backup")
    snapshot.add_argument("--database", type=Path, required=True)
    snapshot.add_argument("--memory-root", type=Path, action="append", required=True)
    snapshot.add_argument("--destination", type=Path, required=True)
    recover = commands.add_parser("restore")
    recover.add_argument("--checkpoint", type=Path, required=True)
    recover.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "migrate":
        report = migrate_scope(args.root, owner=args.owner, project=args.project, apply=args.apply)
    elif args.command == "backup":
        report = backup(args.database, args.memory_root, args.destination)
    else:
        report = restore(args.checkpoint, args.destination)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
