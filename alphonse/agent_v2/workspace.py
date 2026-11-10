"""Workspace path helpers and loss-averse file migration utilities."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

_SAFE_RMTREE_SUPPORTED = bool(getattr(shutil.rmtree, 'avoids_symlink_attacks', False))


def remove_directory_beneath(root: Path, parts: tuple[str, ...], *, error: str) -> None:
    """Remove one contained directory through pinned, no-follow descriptors.

    The configured root is a trusted boundary; no descendant symlink is
    followed. If the platform cannot provide fd-relative safe rmtree, fail
    closed. A missing descendant is an idempotently completed removal.
    """
    if not parts or any(not isinstance(p, str) or not p or p in {'.', '..'} or '/' in p for p in parts):
        raise ValueError(error)
    if (not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY')
            or os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd
            or not _SAFE_RMTREE_SUPPORTED):
        raise ValueError('managed_cleanup_unsupported')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        try:
            current = os.open(root, flags)
        except OSError:
            # The root may be unmounted or renamed; absence does not prove
            # that the owner's data was removed. Do not expose host paths.
            raise ValueError('managed_cleanup_root_unavailable') from None
        descriptors.append(current)
        for part in parts[:-1]:
            try:
                current = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                return
            except OSError:
                raise ValueError(error) from None
            descriptors.append(current)
        try:
            info = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(error)
        # rmtree's fd-based implementation rechecks the leaf after opening it.
        try:
            shutil.rmtree(parts[-1], dir_fd=current)
        except FileNotFoundError:
            # A concurrent removal of an inner entry may interrupt rmtree
            # while the root remains. Only an absent root is complete.
            try:
                os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def copy_tree_missing(source: str | Path, destination: str | Path) -> int:
    """Copy files that do not yet exist at destination, preserving the source.

    This is safe to repeat after an interrupted migration: completed files are
    left untouched and the source remains available for recovery.
    """
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_dir() or source_path == destination_path:
        return 0
    try:
        destination_path.relative_to(source_path)
    except ValueError:
        pass
    else:
        raise ValueError("workspace_migration_destination_inside_source")

    copied = 0
    for current, directories, filenames in os.walk(source_path, followlinks=False):
        current_path = Path(current)
        relative = current_path.relative_to(source_path)
        target_dir = destination_path / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for directory in list(directories):
            source_child = current_path / directory
            if source_child.is_symlink():
                directories.remove(directory)
                target_child = target_dir / directory
                if not target_child.exists() and not target_child.is_symlink():
                    target_child.symlink_to(os.readlink(source_child), target_is_directory=True)
                    copied += 1
        for filename in filenames:
            source_file = current_path / filename
            target_file = target_dir / filename
            if target_file.exists() or target_file.is_symlink():
                continue
            if source_file.is_symlink():
                target_file.symlink_to(os.readlink(source_file))
                copied += 1
            elif source_file.is_file():
                temporary = target_file.with_name(f".{target_file.name}.workspace-copy")
                shutil.copy2(source_file, temporary)
                try:
                    # Hard-link publication is atomic and fails if another
                    # migration already published the same destination.
                    os.link(temporary, target_file)
                    copied += 1
                except FileExistsError:
                    pass
                finally:
                    temporary.unlink(missing_ok=True)
    return copied
