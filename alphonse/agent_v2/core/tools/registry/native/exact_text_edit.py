"""Narrow, verified text replacement for project files."""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

EXACT_TEXT_EDIT_TOOL_ID = "native.exact_text_edit"
EXACT_TEXT_EDIT_TOOL_NAME = "exact_text_edit"
MAX_DIFF_CHARS = 8_000
MAX_READ_BACK_CHARS = 2_000

EXACT_TEXT_EDIT_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "path": {"type": "string", "minLength": 1, "description": "Project-relative path to an existing UTF-8 text file."},
        "expected_text": {"type": "string", "minLength": 1, "description": "Exact text that must currently exist."},
        "replacement_text": {"type": "string", "description": "Exact replacement text."},
        "expected_occurrences": {
            "type": "integer", "minimum": 1, "maximum": 100, "default": 1,
            "description": "Required number of exact matches. The edit is rejected if the count differs.",
        },
    },
    "required": ["path", "expected_text", "replacement_text"],
}


def build_exact_text_edit_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(
        tool_id=EXACT_TEXT_EDIT_TOOL_ID,
        name=EXACT_TEXT_EDIT_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description=(
            "Atomically replace an exact string in one existing project text file. It rejects ambiguous match counts "
            "and returns structured before/after hashes, a bounded unified diff, and verified post-write read-back. "
            "Prefer this over Bash or broad regular expressions for precise file mutations."
        ),
        argument_schema=dict(EXACT_TEXT_EDIT_ARGUMENT_SCHEMA),
        capabilities=("filesystem", "exact_mutation", "verified_read_back"),
        tags=("native", "filesystem", "mutation"),
    )
    return ToolDefinition(
        descriptor=descriptor,
        callable=execute_exact_text_edit,
        argument_schema=dict(EXACT_TEXT_EDIT_ARGUMENT_SCHEMA),
        enabled=True,
        accepts_context=True,
    )


def execute_exact_text_edit(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    expected = str(arguments.get("expected_text") or "")
    replacement = str(arguments.get("replacement_text") or "")
    if not expected:
        raise ValueError("exact_text_edit_expected_text_required")
    try:
        required_count = int(arguments.get("expected_occurrences", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("exact_text_edit_occurrence_count_invalid") from exc
    if not 1 <= required_count <= 100:
        raise ValueError("exact_text_edit_occurrence_count_invalid")

    path, display_path = _resolve_project_path(arguments.get("path"), context)
    before = path.read_text(encoding="utf-8")
    observed_count = before.count(expected)
    if observed_count != required_count:
        raise ValueError(f"exact_text_edit_match_count_mismatch: expected={required_count} observed={observed_count}")

    after = before.replace(expected, replacement, required_count)
    if after == before:
        raise ValueError("exact_text_edit_no_change")
    before_hash = _sha256(before)
    after_hash = _sha256(after)
    _atomic_write(path, after)

    read_back = path.read_text(encoding="utf-8")
    verified = read_back == after and _sha256(read_back) == after_hash
    if not verified:
        raise RuntimeError("exact_text_edit_read_back_failed")
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{display_path}", tofile=f"b/{display_path}", n=3,
        )
    )
    anchor = read_back.find(replacement)
    excerpt_start = max(0, anchor - 500)
    excerpt = read_back[excerpt_start : excerpt_start + MAX_READ_BACK_CHARS]
    return {
        "status": "updated",
        "affected_paths": [display_path],
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "diff": _bounded(diff, MAX_DIFF_CHARS),
        "verification": {
            "status": "verified",
            "expected_occurrences": required_count,
            "observed_before_occurrences": observed_count,
            "observed_after_replacement_occurrences": read_back.count(replacement),
            "post_write_sha256": _sha256(read_back),
            "read_back_excerpt": excerpt,
        },
    }


def _resolve_project_path(raw_path: Any, context: ToolExecutionContext | None) -> tuple[Path, str]:
    rendered = str(raw_path or "").strip()
    if not rendered:
        raise ValueError("exact_text_edit_path_required")
    root = _project_root(context)
    candidate = Path(rendered).expanduser()
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        display = str(candidate.relative_to(root))
    except ValueError as exc:
        raise PermissionError("exact_text_edit_path_outside_project") from exc
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"exact_text_edit_file_not_found: {display}")
    return candidate, display


def _project_root(context: ToolExecutionContext | None) -> Path:
    if context is None or context.project_store is None:
        return Path.cwd().resolve()
    task = context.task
    project_id = str(getattr(task, "project_id", "") or "").strip()
    get_project = getattr(context.project_store, "get_project", None)
    project = get_project(project_id, requester_user_id=getattr(task, "user", None)) if callable(get_project) and project_id else None
    if project is None:
        raise PermissionError("exact_text_edit_project_not_authorized")
    root = Path(str(project.root_path)).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError("exact_text_edit_project_root_unavailable")
    return root


def _atomic_write(path: Path, content: str) -> None:
    mode = path.stat().st_mode
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _bounded(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 18].rstrip() + "\n... [truncated]"
