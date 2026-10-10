"""Bounded, read-only project discovery tools for V3 tactical execution."""

from __future__ import annotations

import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

PROJECT_SEARCH_TOOL_ID = "native.project_search"
PROJECT_READ_TOOL_ID = "native.read_project_file"
_PROTECTED_PARTS = {".alphonse", ".git", ".venv", "node_modules", "vendor", "dist", "build"}
_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv"}


def build_project_search_tool_definition() -> ToolDefinition:
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Case-insensitive keywords separated by whitespace or OR; quote an exact phrase.",
            },
            "max_results": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
        },
        "required": ["query"],
    }
    return ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id=PROJECT_SEARCH_TOOL_ID,
            name="project_search",
            kind=ToolKind.NATIVE,
            description=(
                "Search authorized project file paths and bounded text contents using case-insensitive keywords. "
                "Whitespace and OR separate alternatives; quote a multi-word exact phrase. "
                "Internal .alphonse memory and metadata are always excluded."
            ),
            argument_schema=schema,
            capabilities=("project_search", "filesystem"),
            tags=("native", "filesystem", "read_only"),
            metadata={"v3_capabilities": ["project_record_search", "project_file_inspection"]},
            read_only=True,
        ),
        callable=execute_project_search,
        argument_schema=schema,
        enabled=True,
        accepts_context=True,
    )


def build_project_read_tool_definition() -> ToolDefinition:
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 20000, "default": 12000},
        },
        "required": ["path"],
    }
    return ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id=PROJECT_READ_TOOL_ID,
            name="read_project_file",
            kind=ToolKind.NATIVE,
            description="Read one authorized project text file. Internal .alphonse memory and metadata are always excluded.",
            argument_schema=schema,
            capabilities=("project_read", "filesystem"),
            tags=("native", "filesystem", "read_only"),
            metadata={"v3_capabilities": ["project_file_inspection"]},
            read_only=True,
        ),
        callable=execute_project_read,
        argument_schema=schema,
        enabled=True,
        accepts_context=True,
    )


def execute_project_search(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("project_search_query_required")
    terms = _search_terms(query)
    limit = max(1, min(25, int(arguments.get("max_results", 10))))
    root = _project_root(context)
    matches: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        if len(matches) >= limit:
            break
        relative = path.relative_to(root)
        if not path.is_file() or _is_protected(relative) or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        scanned += 1
        display = relative.as_posix()
        path_term = _first_match(display, terms)
        if path_term:
            matches.append({
                "path": display,
                "line_number": 0,
                "line": "",
                "matched_term": path_term,
                "match_source": "path",
            })
            if len(matches) >= limit:
                break
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(lines, 1):
            matched_term = _first_match(line, terms)
            if not matched_term:
                continue
            matches.append({
                "path": display,
                "line_number": line_number,
                "line": line[:1000],
                "matched_term": matched_term,
                "match_source": "content",
            })
            if len(matches) >= limit:
                break
    if not matches:
        raise LookupError("project_search_no_matches")
    return {
        "query": query,
        "terms": terms,
        "matches": matches,
        "match_count": len(matches),
        "files_scanned": scanned,
        "truncated": len(matches) >= limit,
    }


def _search_terms(query: str) -> list[str]:
    try:
        tokens = shlex.split(query)
    except ValueError:
        tokens = query.split()
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = str(token or "").strip()
        folded = normalized.casefold()
        if not normalized or folded == "or" or folded in seen:
            continue
        seen.add(folded)
        terms.append(normalized)
    return terms or [query]


def _first_match(value: str, terms: list[str]) -> str:
    folded = str(value or "").casefold()
    return next((term for term in terms if term.casefold() in folded), "")


def execute_project_read(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    root = _project_root(context)
    path, display = _resolve_path(root, arguments.get("path"))
    limit = max(1, min(20_000, int(arguments.get("max_chars", 12_000))))
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ValueError("project_read_not_utf8") from exc
    return {"path": display, "content": content[:limit], "total_chars": len(content), "truncated": len(content) > limit}


def _project_root(context: ToolExecutionContext | None) -> Path:
    if context is None or context.project_store is None:
        raise PermissionError("project_tool_context_required")
    task = context.task
    project = context.project_store.get_project(
        str(task.project_id or ""), requester_user_id=str(task.user or "") or None,
    )
    if project is None:
        raise PermissionError("project_tool_project_not_authorized")
    root = Path(str(project.root_path)).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("project_tool_root_unavailable")
    return root


def _resolve_path(root: Path, raw_path: Any) -> tuple[Path, str]:
    rendered = str(raw_path or "").strip().replace("\\", "/")
    pure = PurePosixPath(rendered)
    if not rendered or pure.is_absolute() or ".." in pure.parts or _is_protected(pure):
        raise PermissionError("project_tool_path_protected")
    path = (root / Path(*pure.parts)).resolve()
    try:
        display = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PermissionError("project_tool_path_outside_project") from exc
    if not path.is_file():
        raise ValueError("project_tool_file_not_found")
    return path, display


def _is_protected(path: PurePosixPath | Path) -> bool:
    return any(part in _PROTECTED_PARTS for part in path.parts)
