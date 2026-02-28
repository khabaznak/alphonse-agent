from __future__ import annotations

import os
import re
from pathlib import Path


def resolve_memory_root() -> Path:
    configured = str(os.getenv("ALPHONSE_MEMORY_ROOT") or "").strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = _repo_root() / path
        resolved = path.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    default_root = (_repo_root() / "data" / "memory").resolve()
    default_root.mkdir(parents=True, exist_ok=True)
    return default_root


def user_root(user_id: str, *, root: Path | None = None) -> Path:
    base = root or resolve_memory_root()
    path = base / sanitize_segment(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "anonymous"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
