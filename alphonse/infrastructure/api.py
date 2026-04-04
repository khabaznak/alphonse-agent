from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Body, File, Form, HTTPException, Header, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from alphonse.agent import identity
from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.nervous_system.onboarding_profiles import (
    delete_onboarding_profile,
    get_onboarding_profile,
    list_onboarding_profiles,
    upsert_onboarding_profile,
)
from alphonse.agent.nervous_system.location_profiles import (
    delete_location_profile,
    get_location_profile,
    insert_device_location,
    list_device_locations,
    list_location_profiles,
    upsert_location_profile,
)
from alphonse.agent.nervous_system.tool_configs import (
    delete_tool_config,
    get_tool_config,
    get_active_tool_config,
    list_tool_configs,
    upsert_tool_config,
)
from alphonse.agent.nervous_system.telegram_invites import (
    get_invite,
    list_invites,
    update_invite_status,
)
from alphonse.agent.cognition.prompt_templates_runtime import (
    get_prompt_seed_template,
    list_prompt_seed_templates,
)
from alphonse.agent.nervous_system.terminal_tools import (
    create_terminal_command,
    delete_terminal_sandbox,
    get_terminal_command,
    get_terminal_sandbox,
    get_terminal_session,
    ensure_terminal_session,
    list_terminal_commands,
    list_terminal_sandboxes,
    list_terminal_sessions,
    patch_terminal_sandbox,
    record_terminal_command_output,
    update_terminal_command_status,
    update_terminal_session_status,
    upsert_terminal_sandbox,
)
from alphonse.agent.tools.terminal import TerminalTool
from alphonse.agent.runtime import get_runtime
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_queue_metrics
from alphonse.agent.nervous_system.assets import register_uploaded_asset
from alphonse.infrastructure.api_gateway import gateway
from alphonse.infrastructure.web_event_hub import web_event_hub
from alphonse.agent.lan.api import router as lan_router

app = FastAPI(title="Alphonse API", version="0.1.0")
app.include_router(lan_router, prefix="/lan")


class OnboardingProfileUpsert(BaseModel):
    principal_id: str
    state: str = Field(default="not_started")
    primary_role: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    resume_token: str | None = None
    completed_at: str | None = None


class LocationProfileUpsert(BaseModel):
    location_id: str | None = None
    principal_id: str
    label: str = Field(default="other")
    address_text: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    source: str = "user"
    confidence: float | None = None
    is_active: bool = True


class DeviceLocationCreate(BaseModel):
    principal_id: str | None = None
    device_id: str
    latitude: float
    longitude: float
    accuracy_meters: float | None = None
    source: str = "device"
    observed_at: str | None = None
    metadata: dict[str, Any] | None = None


class ToolConfigUpsert(BaseModel):
    config_id: str | None = None
    tool_key: str
    name: str | None = None
    config: dict[str, Any]
    is_active: bool = True


class RoutingStrategyUpdate(BaseModel):
    strategy: str


class PromptTemplateCreate(BaseModel):
    key: str
    locale: str
    address_style: str
    tone: str
    channel: str
    variant: str
    policy_tier: str
    purpose: str = "general"
    template: str
    enabled: bool = True
    priority: int = 0
    changed_by: str
    reason: str | None = None


class PromptTemplatePatch(BaseModel):
    template: str | None = None
    enabled: bool | None = None
    priority: int | None = None
    purpose: str | None = None
    changed_by: str
    reason: str | None = None


class PromptTemplateRollback(BaseModel):
    version: int
    changed_by: str
    reason: str | None = None


class TerminalSandboxUpsert(BaseModel):
    sandbox_id: str | None = None
    owner_principal_id: str
    label: str
    path: str
    is_active: bool = True


class TerminalSandboxPatch(BaseModel):
    label: str | None = None
    path: str | None = None
    is_active: bool | None = None


class TerminalCommandCreate(BaseModel):
    session_id: str | None = None
    principal_id: str
    sandbox_id: str
    command: str
    cwd: str
    requested_by: str | None = None
    status: str | None = None
    timeout_seconds: float | None = None


class TerminalCommandApprove(BaseModel):
    approved_by: str | None = None


class TerminalCommandFinalize(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    status: str = "executed"


class TerminalCommandExecute(BaseModel):
    approved_by: str | None = None
    timeout_seconds: float | None = None


class UserCreate(BaseModel):
    user_id: str | None = None
    principal_id: str | None = None
    display_name: str
    role: str | None = None
    relationship: str | None = None
    is_admin: bool = False
    is_active: bool = True
    onboarded_at: str | None = None


class UserPatch(BaseModel):
    display_name: str | None = None
    role: str | None = None
    relationship: str | None = None
    is_admin: bool | None = None
    is_active: bool | None = None
    onboarded_at: str | None = None


class TelegramInviteStatusUpdate(BaseModel):
    status: str = "approved"

@app.get("/agent/status")
def agent_status(x_alphonse_api_token: str | None = Header(default=None)) -> dict[str, object]:
    _assert_api_token(x_alphonse_api_token)
    snapshot = get_runtime().snapshot()
    snapshot["pdca_metrics"] = get_pdca_queue_metrics()
    return snapshot


@app.post("/agent/message")
def agent_message(
    payload: dict[str, Any] = Body(...),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    text = _message_text_from_payload(payload)
    metadata_payload = payload.get("metadata")
    metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}
    content_payload = payload.get("content")
    content = dict(content_payload) if isinstance(content_payload, dict) else {"type": "text", "text": text}
    controls_payload = payload.get("controls")
    controls = dict(controls_payload) if isinstance(controls_payload, dict) else {}
    channel = str(payload.get("channel") or payload.get("provider") or "webui")
    target = str(payload.get("target") or payload.get("channel_target") or channel)
    correlation_id = _as_optional_str(payload.get("correlation_id"))
    message_id = str(payload.get("message_id") or payload.get("id") or correlation_id or f"{channel}:{target}:{int(time.time())}")
    envelope = build_incoming_message_envelope(
        message_id=message_id,
        channel_type=channel,
        channel_target=target,
        provider=str(payload.get("provider") or channel),
        text=text,
        occurred_at=str(payload.get("occurred_at") or datetime.now(timezone.utc).isoformat()),
        correlation_id=correlation_id,
        actor_external_user_id=_as_optional_str(payload.get("user_id")),
        actor_display_name=_as_optional_str(payload.get("user_name")),
        actor_person_id=_as_optional_str(payload.get("person_id")),
        controls=controls,
        metadata={
            "args": payload.get("args") or {},
            "raw_metadata": metadata,
            "content": content,
            "api_token_present": bool(x_alphonse_api_token),
        },
        locale=_as_optional_str(payload.get("locale")),
        timezone_name=_as_optional_str(payload.get("timezone")),
        reply_to_message_id=_as_optional_str(payload.get("reply_to_message_id")),
        session_hint=_as_optional_str(payload.get("session_hint")),
    )
    signal = gateway.build_signal(
        "sense.api.message.user.received",
        envelope,
        correlation_id,
    )
    response = gateway.emit_and_wait(signal, timeout=40.0)
    if response is None:
        raise HTTPException(status_code=503, detail="API gateway unavailable")
    return response


@app.post("/agent/assets")
async def create_agent_asset(
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),
    provider: str = Form(default="webui"),
    channel: str = Form(default="webui"),
    target: str | None = Form(default=None),
    kind: str = Form(default="audio"),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    blob = await file.read()
    try:
        record = register_uploaded_asset(
            content=blob,
            kind=str(kind or "audio"),
            mime_type=file.content_type,
            owner_user_id=user_id,
            provider=provider,
            channel_type=channel,
            channel_target=target or channel,
            original_filename=file.filename,
            metadata={"filename": file.filename or "", "uploaded_via": "api.agent.assets"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record


@app.get("/agent/events")
def agent_events(
    channel_target: str = "webui",
    x_alphonse_api_token: str | None = Header(default=None),
) -> StreamingResponse:
    _assert_api_token(x_alphonse_api_token)
    subscriber_id = web_event_hub.subscribe(channel_target)

    def _stream():
        try:
            while True:
                event = web_event_hub.next_event(subscriber_id, timeout=15.0)
                if event is None:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            web_event_hub.unsubscribe(subscriber_id)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/agent/prompts")
def list_agent_prompts(
    key: str | None = None,
    enabled_only: bool = False,
    purpose: str | None = None,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    prompts = list_prompt_seed_templates()
    if key:
        prompts = [row for row in prompts if str(row.get("key") or "") == key]
    return {"items": prompts}


@app.get("/agent/prompts/{template_id}")
def get_agent_prompt(
    template_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    row = get_prompt_seed_template(template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prompt template not found")
    return {"item": row}


@app.post("/agent/prompts", status_code=201)
def create_agent_prompt(
    payload: PromptTemplateCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    raise HTTPException(
        status_code=410,
        detail=(
            "Prompt DB templates are deprecated. Edit file seeds under "
            "alphonse/agent/cognition/prompt_seeds/ instead."
        ),
    )


@app.patch("/agent/prompts/{template_id}")
def patch_agent_prompt(
    template_id: str,
    payload: PromptTemplatePatch,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    raise HTTPException(
        status_code=410,
        detail=(
            "Prompt DB templates are deprecated. Edit file seeds under "
            "alphonse/agent/cognition/prompt_seeds/ instead."
        ),
    )


@app.delete("/agent/prompts/{template_id}")
def delete_agent_prompt(
    template_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    raise HTTPException(
        status_code=410,
        detail=(
            "Prompt DB templates are deprecated. Edit file seeds under "
            "alphonse/agent/cognition/prompt_seeds/ instead."
        ),
    )


@app.post("/agent/prompts/{template_id}/rollback")
def rollback_agent_prompt(
    template_id: str,
    payload: PromptTemplateRollback,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    raise HTTPException(
        status_code=410,
        detail=(
            "Prompt DB templates are deprecated. Edit file seeds under "
            "alphonse/agent/cognition/prompt_seeds/ instead."
        ),
    )


@app.get("/agent/onboarding/profiles")
def list_agent_onboarding_profiles(
    state: str | None = None,
    limit: int = 100,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {"items": list_onboarding_profiles(state=state, limit=limit)}


@app.get("/agent/onboarding/profiles/{principal_id}")
def get_agent_onboarding_profile(
    principal_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_onboarding_profile(principal_id)
    if not item:
        raise HTTPException(status_code=404, detail="Onboarding profile not found")
    return {"item": item}


@app.post("/agent/onboarding/profiles", status_code=201)
def upsert_agent_onboarding_profile(
    payload: OnboardingProfileUpsert,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    principal_id = upsert_onboarding_profile(payload.model_dump())
    return {"item": get_onboarding_profile(principal_id)}


@app.delete("/agent/onboarding/profiles/{principal_id}")
def delete_agent_onboarding_profile(
    principal_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = delete_onboarding_profile(principal_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Onboarding profile not found")
    return {"deleted": True, "principal_id": principal_id}


@app.get("/agent/locations")
def list_agent_locations(
    principal_id: str | None = None,
    label: str | None = None,
    active_only: bool = False,
    limit: int = 100,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {
        "items": list_location_profiles(
            principal_id=principal_id,
            label=label,
            active_only=active_only,
            limit=limit,
        )
    }


@app.get("/agent/locations/{location_id}")
def get_agent_location(
    location_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_location_profile(location_id)
    if not item:
        raise HTTPException(status_code=404, detail="Location profile not found")
    return {"item": item}


@app.post("/agent/locations", status_code=201)
def upsert_agent_location(
    payload: LocationProfileUpsert,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    location_id = upsert_location_profile(payload.model_dump())
    return {"item": get_location_profile(location_id)}


@app.delete("/agent/locations/{location_id}")
def delete_agent_location(
    location_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = delete_location_profile(location_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Location profile not found")
    return {"deleted": True, "location_id": location_id}


@app.get("/agent/device-locations")
def list_agent_device_locations(
    principal_id: str | None = None,
    device_id: str | None = None,
    limit: int = 100,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {
        "items": list_device_locations(
            principal_id=principal_id,
            device_id=device_id,
            limit=limit,
        )
    }


@app.post("/agent/device-locations", status_code=201)
def create_agent_device_location(
    payload: DeviceLocationCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    entry_id = insert_device_location(payload.model_dump())
    rows = list_device_locations(device_id=payload.device_id, limit=1)
    item = rows[0] if rows else {"id": entry_id}
    return {"item": item}


@app.get("/agent/tool-configs")
def list_agent_tool_configs(
    tool_key: str | None = None,
    active_only: bool = False,
    limit: int = 100,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {
        "items": list_tool_configs(
            tool_key=tool_key,
            active_only=active_only,
            limit=limit,
        )
    }


@app.get("/agent/tool-configs/{config_id}")
def get_agent_tool_config(
    config_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_tool_config(config_id)
    if not item:
        raise HTTPException(status_code=404, detail="Tool config not found")
    return {"item": item}


@app.post("/agent/tool-configs", status_code=201)
def upsert_agent_tool_config(
    payload: ToolConfigUpsert,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    config_id = upsert_tool_config(payload.model_dump())
    return {"item": get_tool_config(config_id)}


@app.delete("/agent/tool-configs/{config_id}")
def delete_agent_tool_config(
    config_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = delete_tool_config(config_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Tool config not found")
    return {"deleted": True, "config_id": config_id}


@app.get("/agent/routing/strategy")
def get_routing_strategy(
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    config = get_active_tool_config("routing_strategy")
    strategy = "multi_pass"
    if config and isinstance(config.get("config"), dict):
        raw = config.get("config") or {}
        strategy = str(raw.get("strategy") or strategy)
    return {"strategy": strategy, "config": config}


@app.post("/agent/routing/strategy")
def set_routing_strategy(
    payload: RoutingStrategyUpdate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    strategy = str(payload.strategy or "").strip()
    if strategy not in {"multi_pass", "single_pass"}:
        raise HTTPException(status_code=400, detail="Invalid strategy")
    config_id = upsert_tool_config(
        {
            "tool_key": "routing_strategy",
            "name": "Routing strategy",
            "config": {"strategy": strategy},
            "is_active": True,
        }
    )
    return {"item": get_tool_config(config_id)}


@app.get("/agent/users")
def list_agent_users(
    active_only: bool = False,
    limit: int = 200,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {"items": identity.list_users(active_only=active_only, limit=limit)}


@app.get("/agent/users/{user_id}")
def get_agent_user(
    user_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = identity.get_user(user_id)
    if not item:
        raise HTTPException(status_code=404, detail="User not found")
    return {"item": item}


@app.post("/agent/users", status_code=201)
def create_agent_user(
    payload: UserCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    user_id = identity.upsert_user(payload.model_dump())
    return {"item": identity.get_user(user_id)}


@app.patch("/agent/users/{user_id}")
def patch_agent_user(
    user_id: str,
    payload: UserPatch,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = identity.patch_user(user_id, payload.model_dump(exclude_unset=True))
    if not item:
        raise HTTPException(status_code=404, detail="User not found")
    return {"item": item}


@app.delete("/agent/users/{user_id}")
def delete_agent_user(
    user_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = identity.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": True, "user_id": user_id}


@app.get("/agent/terminal/sandboxes")
def list_agent_terminal_sandboxes(
    owner_principal_id: str | None = None,
    active_only: bool = False,
    limit: int = 200,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {
        "items": list_terminal_sandboxes(
            owner_principal_id=owner_principal_id,
            active_only=active_only,
            limit=limit,
        )
    }


@app.get("/agent/terminal/sandboxes/{sandbox_id}")
def get_agent_terminal_sandbox(
    sandbox_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_terminal_sandbox(sandbox_id)
    if not item:
        raise HTTPException(status_code=404, detail="Terminal sandbox not found")
    return {"item": item}


@app.post("/agent/terminal/sandboxes", status_code=201)
def upsert_agent_terminal_sandbox(
    payload: TerminalSandboxUpsert,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    sandbox_id = upsert_terminal_sandbox(payload.model_dump())
    return {"item": get_terminal_sandbox(sandbox_id)}


@app.patch("/agent/terminal/sandboxes/{sandbox_id}")
def patch_agent_terminal_sandbox(
    sandbox_id: str,
    payload: TerminalSandboxPatch,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = patch_terminal_sandbox(sandbox_id, payload.model_dump(exclude_unset=True))
    if not item:
        raise HTTPException(status_code=404, detail="Terminal sandbox not found")
    return {"item": item}


@app.delete("/agent/terminal/sandboxes/{sandbox_id}")
def delete_agent_terminal_sandbox(
    sandbox_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = delete_terminal_sandbox(sandbox_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Terminal sandbox not found")
    return {"deleted": True, "sandbox_id": sandbox_id}


@app.get("/agent/terminal/sessions")
def list_agent_terminal_sessions(
    principal_id: str | None = None,
    sandbox_id: str | None = None,
    status: str | None = None,
    limit: int = 200,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {
        "items": list_terminal_sessions(
            principal_id=principal_id,
            sandbox_id=sandbox_id,
            status=status,
            limit=limit,
        )
    }


@app.get("/agent/terminal/sessions/{session_id}")
def get_agent_terminal_session(
    session_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_terminal_session(session_id)
    if not item:
        raise HTTPException(status_code=404, detail="Terminal session not found")
    return {"item": item}


@app.post("/agent/terminal/commands", status_code=201)
def create_agent_terminal_command(
    payload: TerminalCommandCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    data = payload.model_dump()
    sandbox = get_terminal_sandbox(str(data.get("sandbox_id")))
    if not sandbox or not sandbox.get("is_active"):
        raise HTTPException(status_code=400, detail="Terminal sandbox not found or inactive")
    terminal_tool = TerminalTool()
    command_status = terminal_tool.classify_command(data.get("command") or "")
    session_id = data.get("session_id") or ensure_terminal_session(
        principal_id=data.get("principal_id"),
        sandbox_id=data.get("sandbox_id"),
    )
    command_id = create_terminal_command(
        {
            "session_id": session_id,
            "command": data.get("command"),
            "cwd": data.get("cwd"),
            "requested_by": data.get("requested_by"),
            "status": data.get("status") or ("approved" if command_status == "auto" else "pending"),
            "approved_by": data.get("requested_by") if command_status == "auto" else None,
            "timeout_seconds": data.get("timeout_seconds"),
        }
    )
    if command_status == "reject":
        record_terminal_command_output(
            command_id,
            stdout="",
            stderr="Command rejected by policy",
            exit_code=None,
            status="rejected",
        )
        update_terminal_session_status(session_id, "rejected")
    elif command_status == "auto":
        update_terminal_session_status(session_id, "approved")
    return {
        "item": get_terminal_command(command_id),
        "session": get_terminal_session(session_id),
    }


@app.get("/agent/terminal/commands")
def list_agent_terminal_commands(
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 200,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {
        "items": list_terminal_commands(
            session_id=session_id,
            status=status,
            limit=limit,
        )
    }


@app.get("/agent/terminal/commands/{command_id}")
def get_agent_terminal_command(
    command_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_terminal_command(command_id)
    if not item:
        raise HTTPException(status_code=404, detail="Terminal command not found")
    return {"item": item}


@app.get("/agent/telegram/invites")
def list_agent_telegram_invites(
    status: str | None = None,
    limit: int = 200,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {"items": list_invites(status=status, limit=limit)}


@app.get("/agent/telegram/invites/{chat_id}")
def get_agent_telegram_invite(
    chat_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_invite(chat_id)
    if not item:
        raise HTTPException(status_code=404, detail="Invite not found")
    return {"item": item}


@app.post("/agent/telegram/invites/{chat_id}/status")
def update_agent_telegram_invite_status(
    chat_id: str,
    payload: TelegramInviteStatusUpdate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = update_invite_status(chat_id, payload.status)
    if not item:
        raise HTTPException(status_code=404, detail="Invite not found")
    return {"item": item}


@app.post("/agent/terminal/commands/{command_id}/approve")
def approve_agent_terminal_command(
    command_id: str,
    payload: TerminalCommandApprove = Body(default=TerminalCommandApprove()),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = update_terminal_command_status(command_id, "approved", approved_by=payload.approved_by)
    if not item:
        raise HTTPException(status_code=404, detail="Terminal command not found")
    session = get_terminal_session(item.get("session_id"))
    if session:
        update_terminal_session_status(session["session_id"], "approved")
    return {"item": item, "session": session}


@app.post("/agent/terminal/commands/{command_id}/reject")
def reject_agent_terminal_command(
    command_id: str,
    payload: TerminalCommandApprove = Body(default=TerminalCommandApprove()),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = update_terminal_command_status(command_id, "rejected", approved_by=payload.approved_by)
    if not item:
        raise HTTPException(status_code=404, detail="Terminal command not found")
    session = get_terminal_session(item.get("session_id"))
    if session:
        update_terminal_session_status(session["session_id"], "rejected")
    return {"item": item, "session": session}


@app.post("/agent/terminal/commands/{command_id}/finalize")
def finalize_agent_terminal_command(
    command_id: str,
    payload: TerminalCommandFinalize,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = record_terminal_command_output(
        command_id,
        stdout=payload.stdout,
        stderr=payload.stderr,
        exit_code=payload.exit_code,
        status=payload.status,
    )
    if not item:
        raise HTTPException(status_code=404, detail="Terminal command not found")
    session = get_terminal_session(item.get("session_id"))
    if session:
        update_terminal_session_status(session["session_id"], payload.status)
    return {"item": item, "session": session}


@app.post("/agent/terminal/commands/{command_id}/execute", status_code=202)
def execute_agent_terminal_command(
    command_id: str,
    payload: TerminalCommandExecute = Body(default=TerminalCommandExecute()),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    command = get_terminal_command(command_id)
    if not command:
        raise HTTPException(status_code=404, detail="Terminal command not found")
    if command.get("status") not in {"approved", "auto", "pending"}:
        raise HTTPException(status_code=400, detail="Command cannot be executed in its current state")
    if command.get("status") == "pending":
        raise HTTPException(status_code=409, detail="Command approval required")
    update_terminal_command_status(command_id, "approved", approved_by=payload.approved_by)
    if command.get("session_id"):
        update_terminal_session_status(command["session_id"], "approved")
    return {"item": get_terminal_command(command_id), "queued": True}


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _message_text_from_payload(payload: dict[str, Any]) -> str:
    text = str(payload.get("text") or "").strip()
    if text:
        return text
    content = payload.get("content")
    if not isinstance(content, dict):
        return ""
    content_type = str(content.get("type") or "").strip().lower()
    if content_type == "text":
        return str(content.get("text") or "").strip()
    assets = content.get("assets")
    if content_type == "asset" and isinstance(assets, list):
        for item in assets:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            if kind == "audio":
                return "[audio asset message]"
    return ""


def _assert_api_token(provided: str | None) -> None:
    expected = os.getenv("ALPHONSE_API_TOKEN")
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API token")
