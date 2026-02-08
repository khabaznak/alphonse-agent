from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from alphonse.agent.cognition.abilities.store import AbilitySpecStore
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


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _assert_api_token(provided: str | None) -> None:
    expected = os.getenv("ALPHONSE_API_TOKEN")
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API token")
