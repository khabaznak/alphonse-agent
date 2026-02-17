from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.services.scratchpad_service import ScratchpadService


@dataclass(frozen=True)
class ToolServices:
    scratchpad: ScratchpadService


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    services: ToolServices


class ScratchpadCreateTool:
    def __init__(self, scratchpad: ScratchpadService) -> None:
        self._scratchpad = scratchpad

    def execute(
        self,
        *,
        title: str,
        scope: str = "project",
        tags: list[str] | None = None,
        template: str | None = None,
        ctx: ToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _resolve_ctx(scratchpad=self._scratchpad, ctx=ctx, user_id=user_id, state=state)
        try:
            result = resolved.services.scratchpad.create_doc(
                user_id=resolved.user_id,
                title=title,
                scope=scope,
                tags=tags or [],
                template=template,
            )
            return _ok(result=result, metadata={"tool": "scratchpad_create"})
        except Exception as exc:
            return _failed(code=_error_code(exc), message=str(exc), metadata={"tool": "scratchpad_create"})


class ScratchpadAppendTool:
    def __init__(self, scratchpad: ScratchpadService) -> None:
        self._scratchpad = scratchpad

    def execute(
        self,
        *,
        doc_id: str,
        text: str,
        ctx: ToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _resolve_ctx(scratchpad=self._scratchpad, ctx=ctx, user_id=user_id, state=state)
        try:
            result = resolved.services.scratchpad.append_doc(
                user_id=resolved.user_id,
                doc_id=doc_id,
                text=text,
            )
            return _ok(result=result, metadata={"tool": "scratchpad_append"})
        except Exception as exc:
            return _failed(code=_error_code(exc), message=str(exc), metadata={"tool": "scratchpad_append"})


class ScratchpadReadTool:
    def __init__(self, scratchpad: ScratchpadService) -> None:
        self._scratchpad = scratchpad

    def execute(
        self,
        *,
        doc_id: str,
        mode: str = "tail",
        max_chars: int = 6000,
        ctx: ToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _resolve_ctx(scratchpad=self._scratchpad, ctx=ctx, user_id=user_id, state=state)
        try:
            result = resolved.services.scratchpad.read_doc(
                user_id=resolved.user_id,
                doc_id=doc_id,
                mode=mode,
                max_chars=max_chars,
            )
            return _ok(result=result, metadata={"tool": "scratchpad_read"})
        except Exception as exc:
            return _failed(code=_error_code(exc), message=str(exc), metadata={"tool": "scratchpad_read"})


class ScratchpadListTool:
    def __init__(self, scratchpad: ScratchpadService) -> None:
        self._scratchpad = scratchpad

    def execute(
        self,
        *,
        scope: str | None = None,
        tag: str | None = None,
        limit: int = 25,
        ctx: ToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _resolve_ctx(scratchpad=self._scratchpad, ctx=ctx, user_id=user_id, state=state)
        try:
            result = resolved.services.scratchpad.list_docs(
                user_id=resolved.user_id,
                scope=scope,
                tag=tag,
                limit=limit,
            )
            return _ok(result=result, metadata={"tool": "scratchpad_list"})
        except Exception as exc:
            return _failed(code=_error_code(exc), message=str(exc), metadata={"tool": "scratchpad_list"})


class ScratchpadSearchTool:
    def __init__(self, scratchpad: ScratchpadService) -> None:
        self._scratchpad = scratchpad

    def execute(
        self,
        *,
        query: str,
        scope: str | None = None,
        tags_any: list[str] | None = None,
        limit: int = 10,
        ctx: ToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _resolve_ctx(scratchpad=self._scratchpad, ctx=ctx, user_id=user_id, state=state)
        try:
            result = resolved.services.scratchpad.search_docs(
                user_id=resolved.user_id,
                query=query,
                scope=scope,
                tags_any=tags_any or [],
                limit=limit,
            )
            return _ok(result=result, metadata={"tool": "scratchpad_search"})
        except Exception as exc:
            return _failed(code=_error_code(exc), message=str(exc), metadata={"tool": "scratchpad_search"})


class ScratchpadForkTool:
    def __init__(self, scratchpad: ScratchpadService) -> None:
        self._scratchpad = scratchpad

    def execute(
        self,
        *,
        doc_id: str,
        new_title: str | None = None,
        ctx: ToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _resolve_ctx(scratchpad=self._scratchpad, ctx=ctx, user_id=user_id, state=state)
        try:
            result = resolved.services.scratchpad.fork_doc(
                user_id=resolved.user_id,
                doc_id=doc_id,
                new_title=new_title,
            )
            return _ok(result=result, metadata={"tool": "scratchpad_fork"})
        except Exception as exc:
            return _failed(code=_error_code(exc), message=str(exc), metadata={"tool": "scratchpad_fork"})


def _resolve_ctx(
    *,
    scratchpad: ScratchpadService,
    ctx: ToolContext | None,
    user_id: str | None,
    state: dict[str, Any] | None,
) -> ToolContext:
    if isinstance(ctx, ToolContext):
        return ctx
    resolved_user = str(user_id or "").strip() or _user_id_from_state(state) or "default"
    return ToolContext(
        user_id=resolved_user,
        services=ToolServices(scratchpad=scratchpad),
    )


def _user_id_from_state(state: dict[str, Any] | None) -> str | None:
    if not isinstance(state, dict):
        return None
    for key in ("actor_person_id", "incoming_user_id", "channel_target", "chat_id"):
        value = str(state.get(key) or "").strip()
        if value:
            return value
    return None


def _error_code(exc: Exception) -> str:
    value = str(exc or "").strip()
    if not value:
        return "scratchpad_failed"
    return value.split(":", 1)[0]


def _ok(*, result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "metadata": metadata,
    }


def _failed(*, code: str, message: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": str(code or "scratchpad_failed"),
            "message": str(message or "scratchpad operation failed"),
        },
        "metadata": metadata,
    }
