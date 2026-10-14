"""Workspace path helpers and loss-averse file migration utilities."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


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
