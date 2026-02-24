from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from alphonse.agent.nervous_system.sandbox_dirs import get_sandbox_alias


class ScratchpadService:
    def __init__(self, *, root: str | Path | None = None) -> None:
        base = Path(root) if root is not None else _default_scratchpad_root()
        self._root = base.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def create_doc(
        self,
        *,
        user_id: str,
        title: str,
        scope: str = "project",
        tags: list[str] | None = None,
        template: str | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        clean_title = str(title or "").strip()
        if not clean_title:
            raise ValueError("title is required")
        now = _now_iso()
        safe_tags = _normalize_tags(tags or [])
        safe_scope = str(scope or "project").strip() or "project"
        doc_id = _new_doc_id()
        user_root = self._user_root(user_id)
        index = self._read_index(user_id)
        rel_path = self._build_relative_path(doc_id=doc_id, title=clean_title, created_at=now)
        doc_abs = self._resolve_doc_path(user_root=user_root, rel_path=rel_path)
        doc_abs.parent.mkdir(parents=True, exist_ok=True)
        content = f"# {clean_title}\n"
        if template is not None and str(template).strip():
            content += f"\n{str(template).strip()}\n"
        doc_abs.write_text(content, encoding="utf-8")
        meta = {
            "doc_id": doc_id,
            "title": clean_title,
            "scope": safe_scope,
            "tags": safe_tags,
            "path": rel_path,
            "created_at": now,
            "updated_at": now,
            "version": int(version),
        }
        docs = index.get("docs")
        if not isinstance(docs, dict):
            docs = {}
            index["docs"] = docs
        docs[doc_id] = meta
        self._write_index(user_id, index)
        return {
            "doc_id": doc_id,
            "title": clean_title,
            "scope": safe_scope,
            "tags": safe_tags,
            "created_at": now,
        }

    def append_doc(self, *, user_id: str, doc_id: str, text: str) -> dict[str, Any]:
        clean_id = str(doc_id or "").strip()
        clean_text = str(text or "").strip()
        if not clean_id:
            raise ValueError("doc_id is required")
        if not clean_text:
            raise ValueError("text is required")
        index = self._read_index(user_id)
        entry = self._index_entry(index, clean_id)
        doc_abs = self._resolve_doc_path(
            user_root=self._user_root(user_id),
            rel_path=str(entry.get("path") or ""),
        )
        if not doc_abs.exists():
            raise ValueError("doc_not_found")
        stamp = _now_iso()
        block = f"\n\n---\n\n### {stamp}\n{clean_text}\n"
        with doc_abs.open("a", encoding="utf-8") as handle:
            handle.write(block)
        entry["updated_at"] = stamp
        self._write_index(user_id, index)
        return {
            "doc_id": clean_id,
            "appended": True,
            "updated_at": stamp,
        }

    def read_doc(
        self,
        *,
        user_id: str,
        doc_id: str,
        mode: str = "tail",
        max_chars: int = 6000,
    ) -> dict[str, Any]:
        clean_id = str(doc_id or "").strip()
        if not clean_id:
            raise ValueError("doc_id is required")
        index = self._read_index(user_id)
        entry = self._index_entry(index, clean_id)
        doc_abs = self._resolve_doc_path(
            user_root=self._user_root(user_id),
            rel_path=str(entry.get("path") or ""),
        )
        if not doc_abs.exists():
            raise ValueError("doc_not_found")
        raw = doc_abs.read_text(encoding="utf-8")
        read_mode = str(mode or "tail").strip().lower() or "tail"
        limit = _normalize_limit(max_chars=max_chars, default=6000, minimum=200, maximum=20000)
        content = self._render_read_content(raw=raw, mode=read_mode, max_chars=limit, title=str(entry.get("title") or ""))
        return {
            "doc_id": clean_id,
            "title": str(entry.get("title") or ""),
            "scope": str(entry.get("scope") or ""),
            "tags": list(entry.get("tags") or []),
            "content": content,
            "updated_at": str(entry.get("updated_at") or ""),
            "mode": read_mode,
            "max_chars": limit,
        }

    def list_docs(
        self,
        *,
        user_id: str,
        scope: str | None = None,
        tag: str | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        index = self._read_index(user_id)
        docs = index.get("docs") if isinstance(index.get("docs"), dict) else {}
        scope_filter = str(scope or "").strip()
        tag_filter = str(tag or "").strip().lower()
        rows: list[dict[str, Any]] = []
        for value in docs.values():
            if not isinstance(value, dict):
                continue
            doc_scope = str(value.get("scope") or "")
            doc_tags = [str(item).strip() for item in (value.get("tags") or []) if str(item).strip()]
            if scope_filter and doc_scope != scope_filter:
                continue
            if tag_filter and tag_filter not in {item.lower() for item in doc_tags}:
                continue
            rows.append(
                {
                    "doc_id": str(value.get("doc_id") or ""),
                    "title": str(value.get("title") or ""),
                    "scope": doc_scope,
                    "tags": doc_tags,
                    "updated_at": str(value.get("updated_at") or ""),
                }
            )
        rows.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return {"docs": rows[: _normalize_limit(max_chars=limit, default=25, minimum=1, maximum=200)]}

    def search_docs(
        self,
        *,
        user_id: str,
        query: str,
        scope: str | None = None,
        tags_any: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        needle = str(query or "").strip().lower()
        if not needle:
            raise ValueError("query is required")
        index = self._read_index(user_id)
        docs = index.get("docs") if isinstance(index.get("docs"), dict) else {}
        scope_filter = str(scope or "").strip()
        any_tags = {str(item).strip().lower() for item in (tags_any or []) if str(item).strip()}
        hits: list[dict[str, Any]] = []
        for value in docs.values():
            if not isinstance(value, dict):
                continue
            doc_scope = str(value.get("scope") or "")
            if scope_filter and doc_scope != scope_filter:
                continue
            doc_tags = [str(item).strip() for item in (value.get("tags") or []) if str(item).strip()]
            tag_set = {item.lower() for item in doc_tags}
            if any_tags and not (tag_set & any_tags):
                continue

            doc_abs = self._resolve_doc_path(
                user_root=self._user_root(user_id),
                rel_path=str(value.get("path") or ""),
            )
            content = doc_abs.read_text(encoding="utf-8") if doc_abs.exists() else ""
            title_lower = str(value.get("title") or "").lower()
            score = 0
            if needle in title_lower:
                score += 100
            if needle in " ".join(tag_set):
                score += 60
            if needle in content.lower():
                score += 20
            if score <= 0:
                continue
            hits.append(
                {
                    "doc_id": str(value.get("doc_id") or ""),
                    "title": str(value.get("title") or ""),
                    "snippet": _build_snippet(content=content, needle=needle),
                    "score": score,
                    "updated_at": str(value.get("updated_at") or ""),
                }
            )
        hits.sort(key=lambda item: (int(item.get("score") or 0), str(item.get("updated_at") or "")), reverse=True)
        return {"hits": hits[: _normalize_limit(max_chars=limit, default=10, minimum=1, maximum=200)]}

    def fork_doc(
        self,
        *,
        user_id: str,
        doc_id: str,
        new_title: str | None = None,
    ) -> dict[str, Any]:
        clean_id = str(doc_id or "").strip()
        if not clean_id:
            raise ValueError("doc_id is required")
        index = self._read_index(user_id)
        source = self._index_entry(index, clean_id)
        source_abs = self._resolve_doc_path(
            user_root=self._user_root(user_id),
            rel_path=str(source.get("path") or ""),
        )
        if not source_abs.exists():
            raise ValueError("doc_not_found")
        source_content = source_abs.read_text(encoding="utf-8")
        title = str(new_title or source.get("title") or "").strip() or str(source.get("title") or "Scratchpad")
        version = int(source.get("version") or 1) + 1
        created = self.create_doc(
            user_id=user_id,
            title=title,
            scope=str(source.get("scope") or "project"),
            tags=list(source.get("tags") or []),
            template=source_content,
            version=version,
        )
        return {
            "new_doc_id": str(created.get("doc_id") or ""),
            "source_doc_id": clean_id,
            "version": version,
        }

    def _render_read_content(self, *, raw: str, mode: str, max_chars: int, title: str) -> str:
        if mode == "head":
            return raw[:max_chars]
        if mode == "full":
            return raw[:max_chars]
        if mode == "summary":
            return _build_summary(raw=raw, title=title, max_chars=max_chars)
        return raw[-max_chars:]

    def _user_root(self, user_id: str) -> Path:
        clean = _safe_user_id(user_id)
        user_root = (self._root / clean).resolve()
        if not _is_subpath(user_root, self._root):
            raise ValueError("invalid_user_scope")
        user_root.mkdir(parents=True, exist_ok=True)
        return user_root

    def _index_path(self, user_id: str) -> Path:
        return self._user_root(user_id) / "index.json"

    def _read_index(self, user_id: str) -> dict[str, Any]:
        index_path = self._index_path(user_id)
        if not index_path.exists():
            payload: dict[str, Any] = {"docs": {}}
            self._write_json_atomic(index_path, payload)
            return payload
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"index_read_failed:{type(exc).__name__}") from exc
        if not isinstance(data, dict):
            raise ValueError("index_invalid")
        docs = data.get("docs")
        if not isinstance(docs, dict):
            data["docs"] = {}
        return data

    def _write_index(self, user_id: str, index: dict[str, Any]) -> None:
        index_path = self._index_path(user_id)
        self._write_json_atomic(index_path, index)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.parent / f".{path.name}.{secrets.token_hex(4)}.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _index_entry(self, index: dict[str, Any], doc_id: str) -> dict[str, Any]:
        docs = index.get("docs")
        if not isinstance(docs, dict):
            raise ValueError("index_invalid")
        entry = docs.get(doc_id)
        if not isinstance(entry, dict):
            raise ValueError("doc_not_found")
        return entry

    def _resolve_doc_path(self, *, user_root: Path, rel_path: str) -> Path:
        clean_rel = str(rel_path or "").strip().lstrip("/").lstrip("\\")
        if not clean_rel:
            raise ValueError("doc_path_missing")
        candidate = (user_root / clean_rel).resolve()
        if not _is_subpath(candidate, user_root):
            raise ValueError("path_traversal_blocked")
        return candidate

    def _build_relative_path(self, *, doc_id: str, title: str, created_at: str) -> str:
        dt = datetime.fromisoformat(created_at)
        slug = _slugify(title)
        return f"{dt.year:04d}/{dt.month:02d}/{doc_id}_{slug}.md"


def _safe_user_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value or "").strip())
    return cleaned or "default"


def _normalize_tags(values: list[str]) -> list[str]:
    tags: list[str] = []
    for item in values:
        tag = str(item or "").strip()
        if not tag:
            continue
        if tag not in tags:
            tags.append(tag)
    return tags


def _slugify(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:48] or "scratchpad"


def _new_doc_id() -> str:
    return f"sp_{secrets.token_hex(4)}"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _is_subpath(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except Exception:
        return False


def _default_scratchpad_root() -> Path:
    configured = str(os.getenv("ALPHONSE_SCRATCHPAD_ROOT") or "").strip()
    if configured:
        return Path(configured)
    # Keep scratchpads in the shared workdir sandbox when available.
    record = get_sandbox_alias("dumpster")
    if isinstance(record, dict) and bool(record.get("enabled")):
        base_path = str(record.get("base_path") or "").strip()
        if base_path:
            return Path(base_path) / "scratchpad"
    return Path("data/scratchpad")


def _normalize_limit(*, max_chars: int, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(max_chars)
    except Exception:
        return default
    return min(max(value, minimum), maximum)


def _build_snippet(*, content: str, needle: str, radius: int = 80) -> str:
    if not content:
        return ""
    text = str(content)
    lower = text.lower()
    idx = lower.find(needle)
    if idx < 0:
        return text[: radius * 2].strip()
    start = max(idx - radius, 0)
    end = min(idx + len(needle) + radius, len(text))
    return text[start:end].strip()


def _build_summary(*, raw: str, title: str, max_chars: int) -> str:
    lines = [f"Title: {title}".strip()]
    headers: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            headers.append(stripped)
        if len(headers) >= 10:
            break
    if headers:
        lines.append("Headers:")
        lines.extend(f"- {item}" for item in headers)
    marker = "\n\n---\n\n### "
    parts = raw.split(marker)
    if len(parts) > 1:
        recent = parts[-2:]
        lines.append("Recent entries:")
        for block in recent:
            block_text = block.strip()
            if not block_text:
                continue
            lines.append(f"- {block_text[:280]}")
    summary = "\n".join(lines).strip()
    return summary[:max_chars]
