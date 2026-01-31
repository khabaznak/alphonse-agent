from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from urllib import request

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.repositories.family import (
    create_family_member,
    get_family_member,
    list_family_members,
    update_family_member,
)
from core.repositories.push_devices import list_push_devices
from core.settings_store import (
    create_setting,
    delete_setting,
    get_setting,
    get_timezone,
    init_db,
    list_settings,
    update_setting,
)
from core.nerve_store import (
    create_signal,
    create_state,
    create_transition,
    delete_signal,
    delete_state,
    delete_transition,
    get_signal,
    get_state,
    get_transition,
    list_signal_queue,
    list_signals,
    list_states,
    list_transitions,
    update_signal,
    update_state,
    update_transition,
)
from interfaces.http.routes.api import router as api_router, trigger_router
from alphonse.agent.cognition.provider_selector import get_provider_info
from alphonse.config import load_alphonse_config

load_dotenv()
init_db()

app = FastAPI(
    title="Atrium",
    description="Domestic presence interface",
    version="0.1.0"
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR.parent / "static")),
    name="static",
)


app.include_router(api_router)
app.include_router(trigger_router)


def _top_nav_links(active: str) -> list[dict[str, str | bool]]:
    return [
        {"label": "Alphonse", "href": "/", "active": active == "alphonse"},
        {"label": "Timed Signals", "href": "/timed-signals", "active": active == "timed-signals"},
        {"label": "Family", "href": "/family", "active": active == "family"},
        {"label": "Push Test", "href": "/push-test", "active": active == "push-test"},
        {"label": "Settings", "href": "/settings", "active": active == "settings"},
        {"label": "Nerve DB", "href": "/nerve/signals", "active": active.startswith("nerve")},
        {"label": "Status JSON", "href": "/status"},
    ]


def _side_nav_links(page: str) -> list[dict[str, str | bool]]:
    if page == "timed-signals":
        return [
            {"label": "Scheduled", "href": "#timed-signals"},
            {"label": "Create", "href": "#create-timed-signal"},
        ]
    if page == "family":
        return [
            {"label": "Family List", "href": "#family-list"},
            {"label": "Edit Member", "href": "#edit-member"},
            {"label": "Create Member", "href": "#create-member"},
        ]
    if page == "push-test":
        return [
            {"label": "Setup", "href": "#push-setup"},
            {"label": "Register", "href": "#push-register"},
        ]
    if page == "settings":
        return [
            {"label": "Settings List", "href": "#settings-list"},
            {"label": "Create Setting", "href": "#create-setting"},
        ]
    if page.startswith("nerve"):
        return [
            {"label": "Signals", "href": "/nerve/signals"},
            {"label": "States", "href": "/nerve/states"},
            {"label": "Transitions", "href": "/nerve/transitions"},
            {"label": "Signal Queue", "href": "/nerve/queue"},
        ]
    return [
        {"label": "Current State", "href": "#current-state", "active": True},
        {"label": "Timed Signals", "href": "/timed-signals"},
        {"label": "Family", "href": "/family"},
        {"label": "Push Test", "href": "/push-test"},
        {"label": "Settings", "href": "/settings"},
        {"label": "Nerve DB", "href": "/nerve/signals"},
        {"label": "Diagnostics", "href": "#diagnostics"},
        {"label": "Presence Log", "href": "#presence-log"},
    ]


def _model_status() -> str:
    info = get_provider_info(load_alphonse_config())
    return f"{info['mode']} · {info['provider']} / {info['model']}"


def _alphonse_api_base() -> str:
    base = os.getenv("ALPHONSE_API_BASE_URL", "http://localhost:8001")
    return base.rstrip("/")


def _fetch_alphonse_status() -> dict[str, object]:
    url = f"{_alphonse_api_base()}/agent/status"
    try:
        with request.urlopen(url, timeout=3) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except Exception:
        return {"runtime": None, "error": "status_unavailable"}


def _fetch_alphonse_message(text: str) -> dict[str, object]:
    url = f"{_alphonse_api_base()}/agent/message"
    body = json.dumps(
        {
            "text": text,
            "channel": "webui",
            "timestamp": time.time(),
        }
    ).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except Exception:
        return {"response": "Alphonse is unavailable.", "decision": None}


def _fetch_alphonse_timed_signals() -> list[dict[str, object]]:
    url = f"{_alphonse_api_base()}/agent/timed-signals"
    try:
        with request.urlopen(url, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            return data.get("timed_signals", [])
    except Exception:
        return []


def _post_alphonse_message(text: str, args: dict[str, object] | None = None) -> dict[str, object]:
    url = f"{_alphonse_api_base()}/agent/message"
    body = json.dumps(
        {
            "text": text,
            "args": args or {},
            "channel": "webui",
            "timestamp": time.time(),
            "metadata": {
                "user_name": "Alex",
            },
        }
    ).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except Exception:
        return {"response": "Alphonse is unavailable.", "decision": None}


def _base_context(request: Request, page: str) -> dict:
    return {
        "request": request,
        "top_nav_links": _top_nav_links(page),
        "side_nav_links": _side_nav_links(page),
        "model_status": _model_status(),
    }


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_datetime(value: str) -> str:
    parsed = _parse_iso_datetime(value)
    if not parsed:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_local_timezone())
    return parsed.isoformat()


def _format_datetime_local(value: str | None) -> str:
    parsed = _parse_iso_datetime(value)
    if not parsed:
        return ""
    parsed = _to_local(parsed)
    return parsed.strftime("%Y-%m-%dT%H:%M")


def _format_date_local(value: str | None) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed:
        return _to_local(parsed).strftime("%Y-%m-%d")
    if value:
        return value[:10]
    return ""


def _local_timezone():
    tz_name = get_timezone()
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tz_name)
    except Exception:
        return datetime.now().astimezone().tzinfo


def _to_local(parsed: datetime) -> datetime:
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_local_timezone())


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_bool(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    return 1 if value.lower() in {"1", "true", "on", "yes"} else 0


def _select_target_notification(
    notifications: list[dict[str, str | None]],
    now: datetime,
) -> tuple[dict[str, str | None] | None, list[dict[str, str | None]]]:
    target = None
    target_delta = None
    for notification in notifications:
        event_datetime = _parse_iso_datetime(
            notification.get("event_datetime")  # type: ignore[arg-type]
        )
        if not event_datetime:
            continue
        delta = abs((event_datetime - now).total_seconds())
        if target_delta is None or delta < target_delta:
            target = notification
            target_delta = delta

    pending = []
    for notification in notifications:
        if target and notification.get("id") == target.get("id"):
            continue
        pending.append(notification)
    return target, pending


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "alphonse.html",
        _base_context(request, "alphonse"),
    )


@app.get("/status/fragment", response_class=HTMLResponse)
def get_status_fragment(request: Request):
    result = _fetch_alphonse_message("status")
    snapshot = _fetch_alphonse_status()
    message = str(result.get("response") or "Alphonse is unavailable.")
    return templates.TemplateResponse(
        "partials/status.html",
        {
            "request": request,
            "message": message.strip(),
            "snapshot": {"runtime": snapshot},
        },
    )


@app.get("/status")
def get_status():
    result = _fetch_alphonse_message("status")
    snapshot = _fetch_alphonse_status()
    return {"message": result.get("response"), "runtime": snapshot}


@app.get("/notifications")
def notifications_redirect():
    return RedirectResponse("/timed-signals", status_code=303)


@app.get("/timed-signals", response_class=HTMLResponse)
def timed_signals(request: Request):
    timed_signals = _fetch_alphonse_timed_signals()
    return templates.TemplateResponse(
        "timed_signals.html",
        {
            **_base_context(request, "timed-signals"),
            "timed_signals": timed_signals,
            "format_dt_local": _format_datetime_local,
        },
    )


@app.post("/timed-signals")
def create_timed_signal(
    signal_type: str = Form(...),
    trigger_at: str = Form(...),
    rrule: str = Form(""),
    timezone: str = Form(""),
    target: str = Form(""),
    origin: str = Form("webui"),
    correlation_id: str = Form(""),
    payload: str = Form("{}"),
):
    try:
        payload_data = json.loads(payload) if payload.strip() else {}
    except json.JSONDecodeError:
        payload_data = {}
    args = {
        "signal_type": signal_type,
        "payload": payload_data,
        "trigger_at": trigger_at,
        "rrule": rrule or None,
        "timezone": timezone or None,
        "target": target or None,
        "origin": origin or None,
        "correlation_id": correlation_id or None,
    }
    _post_alphonse_message("schedule", args)
    return RedirectResponse("/timed-signals", status_code=303)


@app.get("/family", response_class=HTMLResponse)
def family(request: Request, edit_id: str | None = None):
    members = list_family_members(limit=200)
    edit_member = get_family_member(edit_id) if edit_id else None

    return templates.TemplateResponse(
        "family.html",
        {
            **_base_context(request, "family"),
            "members": members,
            "edit_member": edit_member,
            "format_date_local": _format_date_local,
        },
    )


@app.post("/family")
def create_family(
    name: str = Form(...),
    role: str = Form(...),
    birthday: str = Form(""),
):
    payload = {
        "name": name,
        "role": role,
        "birthday": birthday or None,
    }
    created = create_family_member(payload)
    member_id = created.get("id")
    location = f"/family?edit_id={member_id}" if member_id else "/family"
    return RedirectResponse(location, status_code=303)


@app.post("/family/{member_id}")
def update_family(
    member_id: str,
    name: str = Form(...),
    role: str = Form(...),
    birthday: str = Form(""),
):
    payload = {
        "name": name,
        "role": role,
        "birthday": birthday or None,
    }
    update_family_member(member_id, payload)
    return RedirectResponse(f"/family?edit_id={member_id}", status_code=303)


@app.get("/push-test", response_class=HTMLResponse)
def push_test(request: Request):
    public_key = os.getenv("VAPID_PUBLIC_KEY", "")
    family_members = list_family_members(limit=200)
    devices = list_push_devices(limit=200)
    return templates.TemplateResponse(
        "push_test.html",
        {
            **_base_context(request, "push-test"),
            "vapid_public_key": public_key,
            "family_members": family_members,
            "devices": devices,
        },
    )


@app.get("/webpush-sw.js")
def webpush_service_worker():
    sw_path = BASE_DIR.parent / "static" / "webpush-sw.js"
    return FileResponse(sw_path)


@app.get("/settings", response_class=HTMLResponse)
def settings(request: Request):
    return templates.TemplateResponse(
        "settings.html",
        {
            **_base_context(request, "settings"),
        },
    )


@app.get("/settings/table", response_class=HTMLResponse)
def settings_table(request: Request):
    return templates.TemplateResponse(
        "partials/settings_table.html",
        {
            "request": request,
            "settings": list_settings(),
        },
    )


@app.get("/settings/form", response_class=HTMLResponse)
def settings_form(request: Request):
    return templates.TemplateResponse(
        "partials/settings_form.html",
        {
            "request": request,
            "setting": None,
        },
    )


@app.get("/settings/{setting_id}/form", response_class=HTMLResponse)
def settings_form_edit(request: Request, setting_id: int):
    return templates.TemplateResponse(
        "partials/settings_form.html",
        {
            "request": request,
            "setting": get_setting(setting_id),
        },
    )


@app.post("/settings", response_class=HTMLResponse)
def create_settings(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    schema: str = Form(""),
    config: str = Form("{}"),
):
    create_setting(
        {
            "name": name.strip(),
            "description": description.strip() or None,
            "schema": schema.strip() or None,
            "config": config.strip() or "{}",
        }
    )
    return templates.TemplateResponse(
        "partials/settings_table.html",
        {
            "request": request,
            "settings": list_settings(),
        },
        headers={"HX-Trigger": "settings-updated"},
    )


@app.post("/settings/{setting_id}", response_class=HTMLResponse)
def update_settings(
    request: Request,
    setting_id: int,
    name: str = Form(...),
    description: str = Form(""),
    schema: str = Form(""),
    config: str = Form("{}"),
):
    update_setting(
        setting_id,
        {
            "name": name.strip(),
            "description": description.strip() or None,
            "schema": schema.strip() or None,
            "config": config.strip() or "{}",
        },
    )
    return templates.TemplateResponse(
        "partials/settings_table.html",
        {
            "request": request,
            "settings": list_settings(),
        },
        headers={"HX-Trigger": "settings-updated"},
    )


@app.post("/settings/{setting_id}/delete", response_class=HTMLResponse)
def delete_settings(request: Request, setting_id: int):
    delete_setting(setting_id)
    return templates.TemplateResponse(
        "partials/settings_table.html",
        {
            "request": request,
            "settings": list_settings(),
        },
        headers={"HX-Trigger": "settings-updated"},
    )


@app.get("/nerve/signals", response_class=HTMLResponse)
def nerve_signals(request: Request):
    return templates.TemplateResponse(
        "nerve_signals.html",
        {
            **_base_context(request, "nerve-signals"),
        },
    )


@app.get("/nerve/signals/table", response_class=HTMLResponse)
def nerve_signals_table(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_signals_table.html",
        {
            "request": request,
            "signals": list_signals(),
        },
    )


@app.get("/nerve/signals/form", response_class=HTMLResponse)
def nerve_signals_form(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_signals_form.html",
        {
            "request": request,
            "signal": None,
        },
    )


@app.get("/nerve/signals/{signal_id}/form", response_class=HTMLResponse)
def nerve_signals_form_edit(request: Request, signal_id: int):
    return templates.TemplateResponse(
        "partials/nerve_signals_form.html",
        {
            "request": request,
            "signal": get_signal(signal_id),
        },
    )


@app.post("/nerve/signals", response_class=HTMLResponse)
def create_nerve_signal(
    request: Request,
    key: str = Form(...),
    name: str = Form(...),
    source: str = Form("system"),
    description: str = Form(""),
    is_enabled: str | None = Form(None),
):
    create_signal(
        {
            "key": key.strip(),
            "name": name.strip(),
            "source": source.strip() or "system",
            "description": description.strip() or None,
            "is_enabled": _as_bool(is_enabled, 1),
        }
    )
    return templates.TemplateResponse(
        "partials/nerve_signals_table.html",
        {
            "request": request,
            "signals": list_signals(),
        },
        headers={"HX-Trigger": "nerve-signals-updated"},
    )


@app.post("/nerve/signals/{signal_id}", response_class=HTMLResponse)
def update_nerve_signal(
    request: Request,
    signal_id: int,
    key: str = Form(...),
    name: str = Form(...),
    source: str = Form("system"),
    description: str = Form(""),
    is_enabled: str | None = Form(None),
):
    update_signal(
        signal_id,
        {
            "key": key.strip(),
            "name": name.strip(),
            "source": source.strip() or "system",
            "description": description.strip() or None,
            "is_enabled": _as_bool(is_enabled, 1),
        },
    )
    return templates.TemplateResponse(
        "partials/nerve_signals_table.html",
        {
            "request": request,
            "signals": list_signals(),
        },
        headers={"HX-Trigger": "nerve-signals-updated"},
    )


@app.post("/nerve/signals/{signal_id}/delete", response_class=HTMLResponse)
def delete_nerve_signal(request: Request, signal_id: int):
    delete_signal(signal_id)
    return templates.TemplateResponse(
        "partials/nerve_signals_table.html",
        {
            "request": request,
            "signals": list_signals(),
        },
        headers={"HX-Trigger": "nerve-signals-updated"},
    )


@app.get("/nerve/states", response_class=HTMLResponse)
def nerve_states(request: Request):
    return templates.TemplateResponse(
        "nerve_states.html",
        {
            **_base_context(request, "nerve-states"),
        },
    )


@app.get("/nerve/states/table", response_class=HTMLResponse)
def nerve_states_table(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_states_table.html",
        {
            "request": request,
            "states": list_states(),
        },
    )


@app.get("/nerve/states/form", response_class=HTMLResponse)
def nerve_states_form(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_states_form.html",
        {
            "request": request,
            "state": None,
        },
    )


@app.get("/nerve/states/{state_id}/form", response_class=HTMLResponse)
def nerve_states_form_edit(request: Request, state_id: int):
    return templates.TemplateResponse(
        "partials/nerve_states_form.html",
        {
            "request": request,
            "state": get_state(state_id),
        },
    )


@app.post("/nerve/states", response_class=HTMLResponse)
def create_nerve_state(
    request: Request,
    key: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    is_terminal: str | None = Form(None),
    is_enabled: str | None = Form(None),
):
    create_state(
        {
            "key": key.strip(),
            "name": name.strip(),
            "description": description.strip() or None,
            "is_terminal": _as_bool(is_terminal, 0),
            "is_enabled": _as_bool(is_enabled, 1),
        }
    )
    return templates.TemplateResponse(
        "partials/nerve_states_table.html",
        {
            "request": request,
            "states": list_states(),
        },
        headers={"HX-Trigger": "nerve-states-updated"},
    )


@app.post("/nerve/states/{state_id}", response_class=HTMLResponse)
def update_nerve_state(
    request: Request,
    state_id: int,
    key: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    is_terminal: str | None = Form(None),
    is_enabled: str | None = Form(None),
):
    update_state(
        state_id,
        {
            "key": key.strip(),
            "name": name.strip(),
            "description": description.strip() or None,
            "is_terminal": _as_bool(is_terminal, 0),
            "is_enabled": _as_bool(is_enabled, 1),
        },
    )
    return templates.TemplateResponse(
        "partials/nerve_states_table.html",
        {
            "request": request,
            "states": list_states(),
        },
        headers={"HX-Trigger": "nerve-states-updated"},
    )


@app.post("/nerve/states/{state_id}/delete", response_class=HTMLResponse)
def delete_nerve_state(request: Request, state_id: int):
    delete_state(state_id)
    return templates.TemplateResponse(
        "partials/nerve_states_table.html",
        {
            "request": request,
            "states": list_states(),
        },
        headers={"HX-Trigger": "nerve-states-updated"},
    )


@app.get("/nerve/transitions", response_class=HTMLResponse)
def nerve_transitions(request: Request):
    return templates.TemplateResponse(
        "nerve_transitions.html",
        {
            **_base_context(request, "nerve-transitions"),
        },
    )


@app.get("/nerve/transitions/table", response_class=HTMLResponse)
def nerve_transitions_table(request: Request):
    signals = list_signals()
    states = list_states()
    signal_lookup = {signal["id"]: signal for signal in signals}
    state_lookup = {state["id"]: state for state in states}
    return templates.TemplateResponse(
        "partials/nerve_transitions_table.html",
        {
            "request": request,
            "transitions": list_transitions(),
            "signal_lookup": signal_lookup,
            "state_lookup": state_lookup,
        },
    )


@app.get("/nerve/transitions/form", response_class=HTMLResponse)
def nerve_transitions_form(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_transitions_form.html",
        {
            "request": request,
            "transition": None,
            "signals": list_signals(),
            "states": list_states(),
        },
    )


@app.get("/nerve/transitions/{transition_id}/form", response_class=HTMLResponse)
def nerve_transitions_form_edit(request: Request, transition_id: int):
    return templates.TemplateResponse(
        "partials/nerve_transitions_form.html",
        {
            "request": request,
            "transition": get_transition(transition_id),
            "signals": list_signals(),
            "states": list_states(),
        },
    )


@app.post("/nerve/transitions", response_class=HTMLResponse)
def create_nerve_transition(
    request: Request,
    state_id: str = Form(...),
    signal_id: str = Form(...),
    next_state_id: str = Form(...),
    priority: str = Form("100"),
    is_enabled: str | None = Form(None),
    guard_key: str = Form(""),
    action_key: str = Form(""),
    match_any_state: str | None = Form(None),
    notes: str = Form(""),
):
    create_transition(
        {
            "state_id": _as_int(state_id),
            "signal_id": _as_int(signal_id),
            "next_state_id": _as_int(next_state_id),
            "priority": _as_int(priority, 100),
            "is_enabled": _as_bool(is_enabled, 1),
            "guard_key": guard_key.strip() or None,
            "action_key": action_key.strip() or None,
            "match_any_state": _as_bool(match_any_state, 0),
            "notes": notes.strip() or None,
        }
    )
    return nerve_transitions_table(request)


@app.post("/nerve/transitions/{transition_id}", response_class=HTMLResponse)
def update_nerve_transition(
    request: Request,
    transition_id: int,
    state_id: str = Form(...),
    signal_id: str = Form(...),
    next_state_id: str = Form(...),
    priority: str = Form("100"),
    is_enabled: str | None = Form(None),
    guard_key: str = Form(""),
    action_key: str = Form(""),
    match_any_state: str | None = Form(None),
    notes: str = Form(""),
):
    update_transition(
        transition_id,
        {
            "state_id": _as_int(state_id),
            "signal_id": _as_int(signal_id),
            "next_state_id": _as_int(next_state_id),
            "priority": _as_int(priority, 100),
            "is_enabled": _as_bool(is_enabled, 1),
            "guard_key": guard_key.strip() or None,
            "action_key": action_key.strip() or None,
            "match_any_state": _as_bool(match_any_state, 0),
            "notes": notes.strip() or None,
        },
    )
    return nerve_transitions_table(request)


@app.post("/nerve/transitions/{transition_id}/delete", response_class=HTMLResponse)
def delete_nerve_transition(request: Request, transition_id: int):
    delete_transition(transition_id)
    return nerve_transitions_table(request)


@app.get("/nerve/queue", response_class=HTMLResponse)
def nerve_queue(request: Request):
    return templates.TemplateResponse(
        "nerve_queue.html",
        {
            **_base_context(request, "nerve-queue"),
        },
    )


@app.get("/nerve/queue/table", response_class=HTMLResponse)
def nerve_queue_table(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_queue_table.html",
        {
            "request": request,
            "queue": list_signal_queue(),
        },
    )
