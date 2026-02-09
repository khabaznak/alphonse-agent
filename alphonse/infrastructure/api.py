from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from alphonse.agent.cognition.abilities.store import AbilitySpecStore
from alphonse.agent.cognition.intent_catalog import (
    IntentCatalogStore,
    IntentSpec,
    SlotSpec,
    reset_catalog_service,
)
from alphonse.agent.cognition.capability_gaps.coalescing import (
    coalesce_open_intent_gaps,
)
from alphonse.agent.cognition.capability_gaps.workflow import dispatch_gap_proposal
from alphonse.agent.nervous_system.gap_proposals import (
    delete_gap_proposal,
    get_gap_proposal,
    insert_gap_proposal,
    list_gap_proposals,
    update_gap_proposal_status,
)
from alphonse.agent.nervous_system.gap_tasks import (
    get_gap_task,
    list_gap_tasks,
    update_gap_task_status,
)
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
    list_tool_configs,
    upsert_tool_config,
)
from alphonse.agent.nervous_system.terminal_tools import (
    create_terminal_command,
    create_terminal_session,
    delete_terminal_sandbox,
    get_terminal_command,
    get_terminal_sandbox,
    get_terminal_session,
    list_terminal_commands,
    list_terminal_sandboxes,
    list_terminal_sessions,
    patch_terminal_sandbox,
    record_terminal_command_output,
    update_terminal_command_status,
    update_terminal_session_status,
    upsert_terminal_sandbox,
)
from alphonse.agent.nervous_system.users import (
    delete_user,
    get_user,
    list_users,
    patch_user,
    upsert_user,
)
from alphonse.infrastructure.api_gateway import gateway
from alphonse.infrastructure.web_event_hub import web_event_hub
from alphonse.agent.lan.api import router as lan_router

app = FastAPI(title="Alphonse API", version="0.1.0")
app.include_router(lan_router, prefix="/lan")


class GapProposalCreate(BaseModel):
    gap_id: str
    proposed_category: str = "intent_missing"
    proposed_next_action: str = "plan"
    proposed_intent_name: str | None = None
    confidence: float | None = None
    proposed_clarifying_question: str | None = None
    notes: str | None = None
    language: str | None = None
    status: str = "pending"


class GapProposalUpdate(BaseModel):
    status: str = Field(..., description="pending|approved|rejected|dispatched")
    reviewer: str | None = None
    notes: str | None = None


class GapProposalDispatch(BaseModel):
    task_type: str | None = Field(default=None, description="plan|investigate|fix_now")
    actor: str | None = None


class GapCoalesceRequest(BaseModel):
    limit: int = 300
    min_cluster_size: int = 2


class GapTaskUpdate(BaseModel):
    status: str = Field(..., description="open|done")


class AbilitySpecCreate(BaseModel):
    intent_name: str
    kind: str
    tools: list[str] = Field(default_factory=list)
    spec: dict[str, Any]
    enabled: bool = True
    source: str = "user"


class AbilitySpecPatch(BaseModel):
    kind: str | None = None
    tools: list[str] | None = None
    spec: dict[str, Any] | None = None
    enabled: bool | None = None
    source: str | None = None


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


class SlotPayload(BaseModel):
    name: str
    type: str
    required: bool = True
    prompt_key: str
    examples: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    critical: bool = True
    semantic_text: bool = False
    min_length: int | None = None
    reject_if_core_conversational: bool = False


class IntentCreate(BaseModel):
    intent_name: str
    category: str
    description: str
    examples: list[str] = Field(default_factory=list)
    required_slots: list[SlotPayload] = Field(default_factory=list)
    optional_slots: list[SlotPayload] = Field(default_factory=list)
    default_mode: str
    risk_level: str
    handler: str
    enabled: bool = True
    intent_version: str = "1.0.0"
    origin: str = "user"
    parent_intent: str | None = None


class IntentPatch(BaseModel):
    category: str | None = None
    description: str | None = None
    examples: list[str] | None = None
    required_slots: list[SlotPayload] | None = None
    optional_slots: list[SlotPayload] | None = None
    default_mode: str | None = None
    risk_level: str | None = None
    handler: str | None = None
    enabled: bool | None = None
    intent_version: str | None = None
    origin: str | None = None
    parent_intent: str | None = None


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


class TerminalCommandApprove(BaseModel):
    approved_by: str | None = None


class TerminalCommandFinalize(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    status: str = "executed"


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

@app.get("/agent/status")
def agent_status(x_alphonse_api_token: str | None = Header(default=None)) -> dict[str, object]:
    _assert_api_token(x_alphonse_api_token)
    signal = gateway.build_signal(
        "api.status_requested",
        {"api_token": x_alphonse_api_token},
        None,
    )
    response = gateway.emit_and_wait(signal, timeout=40.0)
    if response is None:
        raise HTTPException(status_code=503, detail="API gateway unavailable")
    return response


@app.post("/agent/message")
def agent_message(
    payload: dict[str, Any] = Body(...),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    text = str(payload.get("text", "")).strip()
    signal = gateway.build_signal(
        "api.message_received",
        {
            "text": text,
            "args": payload.get("args") or {},
            "user_id": payload.get("user_id"),
            "channel": payload.get("channel") or "webui",
            "metadata": payload.get("metadata") or {},
            "timestamp": payload.get("timestamp") or time.time(),
            "api_token": x_alphonse_api_token,
        },
        _as_optional_str(payload.get("correlation_id")),
    )
    response = gateway.emit_and_wait(signal, timeout=40.0)
    if response is None:
        raise HTTPException(status_code=503, detail="API gateway unavailable")
    return response


@app.get("/agent/timed-signals")
def timed_signals(limit: int = 200, x_alphonse_api_token: str | None = Header(default=None)) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    signal = gateway.build_signal(
        "api.timed_signals_requested",
        {"limit": limit, "api_token": x_alphonse_api_token},
        None,
    )
    response = gateway.emit_and_wait(signal, timeout=5.0)
    if response is None:
        raise HTTPException(status_code=503, detail="API gateway unavailable")
    return response


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


@app.post("/agent/gap-proposals/coalesce")
def coalesce_gap_proposals(
    payload: GapCoalesceRequest = Body(default=GapCoalesceRequest()),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    created = coalesce_open_intent_gaps(
        limit=payload.limit,
        min_cluster_size=payload.min_cluster_size,
    )
    return {"created_count": len(created), "proposal_ids": created}


@app.get("/agent/gap-proposals")
def list_agent_gap_proposals(
    status: str | None = None,
    limit: int = 50,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {"items": list_gap_proposals(status=status, limit=limit)}


@app.post("/agent/gap-proposals", status_code=201)
def create_agent_gap_proposal(
    payload: GapProposalCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    proposal_id = insert_gap_proposal(payload.model_dump())
    proposal = get_gap_proposal(proposal_id)
    return {"id": proposal_id, "item": proposal}


@app.get("/agent/gap-proposals/{proposal_id}")
def get_agent_gap_proposal(
    proposal_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    proposal = get_gap_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Gap proposal not found")
    return {"item": proposal}


@app.patch("/agent/gap-proposals/{proposal_id}")
def update_agent_gap_proposal(
    proposal_id: str,
    payload: GapProposalUpdate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = update_gap_proposal_status(
        proposal_id,
        payload.status,
        reviewer=payload.reviewer,
        notes=payload.notes,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Gap proposal not found")
    return {"item": get_gap_proposal(proposal_id)}


@app.delete("/agent/gap-proposals/{proposal_id}")
def delete_agent_gap_proposal(
    proposal_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = delete_gap_proposal(proposal_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Gap proposal not found")
    return {"deleted": True, "id": proposal_id}


@app.post("/agent/gap-proposals/{proposal_id}/dispatch")
def dispatch_agent_gap_proposal(
    proposal_id: str,
    payload: GapProposalDispatch = Body(default=GapProposalDispatch()),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    try:
        task_id = dispatch_gap_proposal(
            proposal_id,
            task_type=payload.task_type,
            actor=payload.actor,
        )
    except ValueError as exc:
        if str(exc) == "proposal_not_found":
            raise HTTPException(status_code=404, detail="Gap proposal not found") from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    task = get_gap_task(task_id)
    return {"task_id": task_id, "task": task, "proposal": get_gap_proposal(proposal_id)}


@app.get("/agent/gap-tasks")
def list_agent_gap_tasks(
    status: str | None = None,
    limit: int = 50,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {"items": list_gap_tasks(status=status, limit=limit)}


@app.get("/agent/gap-tasks/{task_id}")
def get_agent_gap_task(
    task_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    task = get_gap_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Gap task not found")
    return {"item": task}


@app.patch("/agent/gap-tasks/{task_id}")
def update_agent_gap_task(
    task_id: str,
    payload: GapTaskUpdate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = update_gap_task_status(task_id, payload.status)
    if not ok:
        raise HTTPException(status_code=404, detail="Gap task not found")
    return {"item": get_gap_task(task_id)}


@app.get("/agent/abilities")
def list_agent_abilities(
    enabled_only: bool = False,
    limit: int = 100,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = AbilitySpecStore()
    return {"items": store.list_specs(enabled_only=enabled_only, limit=limit)}


@app.get("/agent/intents")
def list_agent_intents(
    enabled_only: bool = False,
    limit: int = 200,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = IntentCatalogStore()
    intents = store.list_enabled() if enabled_only else store.list_all()
    return {"items": [_spec_to_dict(spec) for spec in intents[:limit]]}


@app.get("/agent/intents/{intent_name}")
def get_agent_intent(
    intent_name: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = IntentCatalogStore()
    spec = store.get(intent_name)
    if not spec:
        raise HTTPException(status_code=404, detail="Intent not found")
    return {"item": _spec_to_dict(spec)}


@app.post("/agent/intents", status_code=201)
def create_agent_intent(
    payload: IntentCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = IntentCatalogStore()
    spec = _spec_from_payload(payload)
    store.upsert(spec)
    reset_catalog_service()
    return {"item": _spec_to_dict(store.get(payload.intent_name))}


@app.patch("/agent/intents/{intent_name}")
def patch_agent_intent(
    intent_name: str,
    payload: IntentPatch,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = IntentCatalogStore()
    current = store.get(intent_name)
    if not current:
        raise HTTPException(status_code=404, detail="Intent not found")
    updated = IntentSpec(
        intent_name=intent_name,
        category=payload.category if payload.category is not None else current.category,
        description=payload.description if payload.description is not None else current.description,
        examples=payload.examples if payload.examples is not None else current.examples,
        required_slots=_slots_from_payload(payload.required_slots)
        if payload.required_slots is not None
        else current.required_slots,
        optional_slots=_slots_from_payload(payload.optional_slots)
        if payload.optional_slots is not None
        else current.optional_slots,
        default_mode=payload.default_mode if payload.default_mode is not None else current.default_mode,
        risk_level=payload.risk_level if payload.risk_level is not None else current.risk_level,
        handler=payload.handler if payload.handler is not None else current.handler,
        enabled=payload.enabled if payload.enabled is not None else current.enabled,
        intent_version=payload.intent_version if payload.intent_version is not None else current.intent_version,
        origin=payload.origin if payload.origin is not None else current.origin,
        parent_intent=payload.parent_intent if payload.parent_intent is not None else current.parent_intent,
        created_at=current.created_at,
        updated_at=current.updated_at,
    )
    store.upsert(updated)
    reset_catalog_service()
    return {"item": _spec_to_dict(store.get(intent_name))}


@app.delete("/agent/intents/{intent_name}")
def delete_agent_intent(
    intent_name: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = IntentCatalogStore()
    current = store.get(intent_name)
    if not current:
        raise HTTPException(status_code=404, detail="Intent not found")
    disabled = IntentSpec(
        intent_name=current.intent_name,
        category=current.category,
        description=current.description,
        examples=current.examples,
        required_slots=current.required_slots,
        optional_slots=current.optional_slots,
        default_mode=current.default_mode,
        risk_level=current.risk_level,
        handler=current.handler,
        enabled=False,
        intent_version=current.intent_version,
        origin=current.origin,
        parent_intent=current.parent_intent,
        created_at=current.created_at,
        updated_at=current.updated_at,
    )
    store.upsert(disabled)
    reset_catalog_service()
    return {"deleted": True, "intent_name": intent_name}


@app.get("/agent/abilities/{intent_name}")
def get_agent_ability(
    intent_name: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = AbilitySpecStore()
    item = store.get_spec(intent_name)
    if not item:
        raise HTTPException(status_code=404, detail="Ability spec not found")
    return {"item": item}


@app.post("/agent/abilities", status_code=201)
def create_agent_ability(
    payload: AbilitySpecCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    if payload.spec.get("intent_name") and payload.spec.get("intent_name") != payload.intent_name:
        raise HTTPException(status_code=400, detail="spec.intent_name must match intent_name")
    store = AbilitySpecStore()
    spec = dict(payload.spec)
    spec["intent_name"] = payload.intent_name
    spec["kind"] = payload.kind
    spec["tools"] = list(payload.tools)
    store.upsert_spec(
        payload.intent_name,
        spec,
        enabled=payload.enabled,
        source=payload.source,
    )
    return {"item": store.get_spec(payload.intent_name)}


@app.patch("/agent/abilities/{intent_name}")
def patch_agent_ability(
    intent_name: str,
    payload: AbilitySpecPatch,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = AbilitySpecStore()
    current = store.get_spec(intent_name)
    if not current:
        raise HTTPException(status_code=404, detail="Ability spec not found")
    current_spec = current.get("spec") if isinstance(current.get("spec"), dict) else {}
    spec = dict(current_spec)
    if payload.spec is not None:
        spec.update(payload.spec)
    if payload.kind is not None:
        spec["kind"] = payload.kind
    if payload.tools is not None:
        spec["tools"] = list(payload.tools)
    spec["intent_name"] = intent_name
    enabled = current.get("enabled") if payload.enabled is None else payload.enabled
    source = str(current.get("source") or "user") if payload.source is None else payload.source
    store.upsert_spec(
        intent_name,
        spec,
        enabled=bool(enabled),
        source=source,
    )
    return {"item": store.get_spec(intent_name)}


@app.delete("/agent/abilities/{intent_name}")
def delete_agent_ability(
    intent_name: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    store = AbilitySpecStore()
    ok = store.delete_spec(intent_name)
    if not ok:
        raise HTTPException(status_code=404, detail="Ability spec not found")
    return {"deleted": True, "intent_name": intent_name}


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


@app.get("/agent/users")
def list_agent_users(
    active_only: bool = False,
    limit: int = 200,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    return {"items": list_users(active_only=active_only, limit=limit)}


@app.get("/agent/users/{user_id}")
def get_agent_user(
    user_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = get_user(user_id)
    if not item:
        raise HTTPException(status_code=404, detail="User not found")
    return {"item": item}


@app.post("/agent/users", status_code=201)
def create_agent_user(
    payload: UserCreate,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    user_id = upsert_user(payload.model_dump())
    return {"item": get_user(user_id)}


@app.patch("/agent/users/{user_id}")
def patch_agent_user(
    user_id: str,
    payload: UserPatch,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    item = patch_user(user_id, payload.model_dump(exclude_unset=True))
    if not item:
        raise HTTPException(status_code=404, detail="User not found")
    return {"item": item}


@app.delete("/agent/users/{user_id}")
def delete_agent_user(
    user_id: str,
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    ok = delete_user(user_id)
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
    session_id = data.get("session_id")
    if not session_id:
        session_id = create_terminal_session(
            {
                "principal_id": data.get("principal_id"),
                "sandbox_id": data.get("sandbox_id"),
                "status": "pending",
            }
        )
    command_id = create_terminal_command(
        {
            "session_id": session_id,
            "command": data.get("command"),
            "cwd": data.get("cwd"),
            "requested_by": data.get("requested_by"),
            "status": data.get("status") or "pending",
        }
    )
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


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _slots_from_payload(slots: list[SlotPayload]) -> list[SlotSpec]:
    return [
        SlotSpec(
            name=slot.name,
            type=slot.type,
            required=slot.required,
            prompt_key=slot.prompt_key,
            examples=list(slot.examples),
            constraints=dict(slot.constraints),
            critical=slot.critical,
            semantic_text=slot.semantic_text,
            min_length=slot.min_length,
            reject_if_core_conversational=slot.reject_if_core_conversational,
        )
        for slot in slots
    ]


def _spec_from_payload(payload: IntentCreate) -> IntentSpec:
    return IntentSpec(
        intent_name=payload.intent_name,
        category=payload.category,
        description=payload.description,
        examples=list(payload.examples),
        required_slots=_slots_from_payload(payload.required_slots),
        optional_slots=_slots_from_payload(payload.optional_slots),
        default_mode=payload.default_mode,
        risk_level=payload.risk_level,
        handler=payload.handler,
        enabled=payload.enabled,
        intent_version=payload.intent_version,
        origin=payload.origin,
        parent_intent=payload.parent_intent,
    )


def _spec_to_dict(spec: IntentSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {
        "intent_name": spec.intent_name,
        "category": spec.category,
        "description": spec.description,
        "examples": list(spec.examples),
        "required_slots": [_slot_to_dict(slot) for slot in spec.required_slots],
        "optional_slots": [_slot_to_dict(slot) for slot in spec.optional_slots],
        "default_mode": spec.default_mode,
        "risk_level": spec.risk_level,
        "handler": spec.handler,
        "enabled": spec.enabled,
        "intent_version": spec.intent_version,
        "origin": spec.origin,
        "parent_intent": spec.parent_intent,
        "created_at": spec.created_at,
        "updated_at": spec.updated_at,
    }


def _slot_to_dict(slot: SlotSpec) -> dict[str, Any]:
    return {
        "name": slot.name,
        "type": slot.type,
        "required": slot.required,
        "prompt_key": slot.prompt_key,
        "examples": list(slot.examples),
        "constraints": dict(slot.constraints),
        "critical": slot.critical,
        "semantic_text": slot.semantic_text,
        "min_length": slot.min_length,
        "reject_if_core_conversational": slot.reject_if_core_conversational,
    }


def _assert_api_token(provided: str | None) -> None:
    expected = os.getenv("ALPHONSE_API_TOKEN")
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API token")
