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
from core.identity_store import (
    create_channel,
    create_group,
    create_person,
    create_prefs,
    delete_channel,
    delete_group,
    delete_person,
    delete_prefs,
    get_channel,
    get_group,
    get_person,
    get_prefs,
    get_presence,
    list_channels,
    list_groups,
    list_persons,
    list_prefs,
    list_presence,
    update_channel,
    update_group,
    update_person,
    update_prefs,
    upsert_presence,
)
from core.nerve_store import (
    create_plan_executor,
    create_plan_kind,
    create_plan_kind_version,
    create_signal,
    create_sense,
    create_state,
    create_transition,
    delete_plan_executor,
    delete_plan_kind,
    delete_plan_kind_version,
    delete_signal,
    delete_sense,
    delete_state,
    delete_transition,
    get_plan_executor,
    get_plan_kind,
    get_plan_kind_version,
    get_signal,
    get_sense,
    get_state,
    get_transition,
    list_plan_executors,
    list_plan_instances,
    list_plan_kind_versions,
    list_plan_kinds,
    list_signal_queue,
    list_signals,
    list_senses,
    list_states,
    list_trace,
    list_transitions,
    resolve_transition,
    update_plan_executor,
    update_plan_kind,
    update_plan_kind_version,
    update_signal,
    update_sense,
    update_state,
    update_transition,
)
from interfaces.http.routes.api import router as api_router, trigger_router
from alphonse.agent.cognition.provider_selector import get_provider_info
from alphonse.config import load_alphonse_config
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

load_dotenv()
init_db()
apply_schema(resolve_nervous_system_db_path())

app = FastAPI(
    title="Atrium", description="Domestic presence interface", version="0.1.0"
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
        {
            "label": "Timed Signals",
            "href": "/timed-signals",
            "active": active == "timed-signals",
        },
        {"label": "Family", "href": "/family", "active": active == "family"},
        {"label": "Push Test", "href": "/push-test", "active": active == "push-test"},
        {"label": "Settings", "href": "/settings", "active": active == "settings"},
        {
            "label": "Nerve DB",
            "href": "/nerve/signals",
            "active": active.startswith("nerve"),
        },
        {"label": "Plans", "href": "/admin/plans", "active": active == "admin-plans"},
        {
            "label": "Identity",
            "href": "/identity/persons",
            "active": active.startswith("identity"),
        },
        {"label": "LAN", "href": "/lan", "active": active == "lan"},
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
            {"label": "Senses", "href": "/nerve/senses"},
            {"label": "FSM Inspector", "href": "/nerve/inspector"},
            {"label": "Trace", "href": "/nerve/trace"},
            {"label": "Signal Queue", "href": "/nerve/queue"},
        ]
    if page.startswith("identity"):
        return [
            {"label": "Persons", "href": "/identity/persons"},
            {"label": "Groups", "href": "/identity/groups"},
            {"label": "Channels", "href": "/identity/channels"},
            {"label": "Prefs", "href": "/identity/prefs"},
            {"label": "Presence", "href": "/identity/presence"},
        ]
    if page == "admin-plans":
        return [
            {"label": "Plan Kinds", "href": "#plan-kinds"},
            {"label": "Plan Versions", "href": "#plan-versions"},
            {"label": "Plan Executors", "href": "#plan-executors"},
            {"label": "Plan Instances", "href": "#plan-instances"},
        ]
    if page == "lan":
        return [
            {"label": "Pairing", "href": "#lan-pairing"},
            {"label": "Devices", "href": "#lan-devices"},
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


def _alphonse_api_token() -> str | None:
    token = os.getenv("ALPHONSE_API_TOKEN")
    return token.strip() if isinstance(token, str) and token.strip() else None


def _fetch_alphonse_status() -> dict[str, object]:
    url = f"{_alphonse_api_base()}/agent/status"
    req = request.Request(url, method="GET")
    token = _alphonse_api_token()
    if token:
        req.add_header("x-alphonse-api-token", token)
    try:
        with request.urlopen(req, timeout=3) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, dict) and "data" in data:
                return data.get("data") or {}
            return data
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
    token = _alphonse_api_token()
    if token:
        req.add_header("x-alphonse-api-token", token)
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            return data if isinstance(data, dict) else {"message": str(data)}
    except Exception:
        return {"message": "Alphonse is unavailable."}


def _fetch_alphonse_timed_signals() -> list[dict[str, object]]:
    url = f"{_alphonse_api_base()}/agent/timed-signals"
    req = request.Request(url, method="GET")
    token = _alphonse_api_token()
    if token:
        req.add_header("x-alphonse-api-token", token)
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {}).get("timed_signals", [])
            return data.get("timed_signals", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _fetch_alphonse_pairing_code() -> dict[str, object]:
    url = f"{_alphonse_api_base()}/lan/pairing-codes/qr"
    req = request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    token = _alphonse_api_token()
    if token:
        req.add_header("x-alphonse-api-token", token)
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fetch_alphonse_paired_devices() -> list[dict[str, object]]:
    url = f"{_alphonse_api_base()}/lan/devices"
    req = request.Request(url, method="GET")
    token = _alphonse_api_token()
    if token:
        req.add_header("x-alphonse-api-token", token)
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, dict):
                return data.get("devices", []) if isinstance(data.get("devices"), list) else []
            return []
    except Exception:
        return []

def _post_alphonse_message(
    text: str, args: dict[str, object] | None = None
) -> dict[str, object]:
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
    token = _alphonse_api_token()
    if token:
        req.add_header("x-alphonse-api-token", token)
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            return data if isinstance(data, dict) else {"message": str(data)}
    except Exception:
        return {"message": "Alphonse is unavailable."}


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
    message = str(result.get("message") or "Alphonse is unavailable.")
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
    return {"message": result.get("message"), "runtime": snapshot}


@app.get("/lan", response_class=HTMLResponse)
def lan_home(request: Request):
    devices = _fetch_alphonse_paired_devices()
    return templates.TemplateResponse(
        "lan.html",
        {
            **_base_context(request, "lan"),
            "devices": devices,
        },
    )


@app.post("/lan/pairing-code", response_class=HTMLResponse)
def lan_pairing_code(request: Request):
    data = _fetch_alphonse_pairing_code()
    pair_code = data.get("pair_code") if isinstance(data, dict) else None
    expires_at = data.get("expires_at") if isinstance(data, dict) else None
    qr_svg = data.get("qr_svg") if isinstance(data, dict) else None
    if not pair_code:
        pair_code = "Pairing unavailable"
        expires_at = "n/a"
    return templates.TemplateResponse(
        "partials/lan_pairing_code.html",
        {
            "request": request,
            "pair_code": pair_code,
            "expires_at": expires_at,
            "qr_svg": qr_svg,
        },
    )


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
    schema_: str = Form("", alias="schema"),
    config: str = Form("{}"),
):
    create_setting(
        {
            "name": name.strip(),
            "description": description.strip() or None,
            "schema": schema_.strip() or None,
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
    schema_: str = Form("", alias="schema"),
    config: str = Form("{}"),
):
    update_setting(
        setting_id,
        {
            "name": name.strip(),
            "description": description.strip() or None,
            "schema": schema_.strip() or None,
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


@app.get("/nerve/senses", response_class=HTMLResponse)
def nerve_senses(request: Request):
    return templates.TemplateResponse(
        "nerve_senses.html",
        {
            **_base_context(request, "nerve-senses"),
        },
    )


@app.get("/nerve/senses/table", response_class=HTMLResponse)
def nerve_senses_table(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_senses_table.html",
        {
            "request": request,
            "senses": list_senses(),
        },
    )


@app.get("/nerve/senses/form", response_class=HTMLResponse)
def nerve_senses_form(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_senses_form.html",
        {
            "request": request,
            "sense": None,
        },
    )


@app.get("/nerve/senses/{sense_id}/form", response_class=HTMLResponse)
def nerve_senses_form_edit(request: Request, sense_id: int):
    return templates.TemplateResponse(
        "partials/nerve_senses_form.html",
        {
            "request": request,
            "sense": get_sense(sense_id),
        },
    )


@app.post("/nerve/senses", response_class=HTMLResponse)
def create_nerve_sense(
    request: Request,
    key: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    source_type: str = Form("system"),
    enabled: str | None = Form(None),
    owner: str = Form(""),
):
    create_sense(
        {
            "key": key.strip(),
            "name": name.strip(),
            "description": description.strip() or None,
            "source_type": source_type.strip() or "system",
            "enabled": _as_bool(enabled, 1),
            "owner": owner.strip() or None,
        }
    )
    return nerve_senses_table(request)


@app.post("/nerve/senses/{sense_id}", response_class=HTMLResponse)
def update_nerve_sense(
    request: Request,
    sense_id: int,
    key: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    source_type: str = Form("system"),
    enabled: str | None = Form(None),
    owner: str = Form(""),
):
    update_sense(
        sense_id,
        {
            "key": key.strip(),
            "name": name.strip(),
            "description": description.strip() or None,
            "source_type": source_type.strip() or "system",
            "enabled": _as_bool(enabled, 1),
            "owner": owner.strip() or None,
        },
    )
    return nerve_senses_table(request)


@app.post("/nerve/senses/{sense_id}/delete", response_class=HTMLResponse)
def delete_nerve_sense(request: Request, sense_id: int):
    delete_sense(sense_id)
    return nerve_senses_table(request)


@app.get("/nerve/trace", response_class=HTMLResponse)
def nerve_trace(request: Request):
    return templates.TemplateResponse(
        "nerve_trace.html",
        {
            **_base_context(request, "nerve-trace"),
        },
    )


@app.get("/nerve/trace/table", response_class=HTMLResponse)
def nerve_trace_table(request: Request):
    return templates.TemplateResponse(
        "partials/nerve_trace_table.html",
        {
            "request": request,
            "trace": list_trace(),
        },
    )


@app.get("/nerve/inspector", response_class=HTMLResponse)
def nerve_inspector(request: Request):
    return templates.TemplateResponse(
        "nerve_inspector.html",
        {
            **_base_context(request, "nerve-inspector"),
            "signals": list_signals(),
            "states": list_states(),
            "resolved": None,
        },
    )


@app.post("/nerve/inspector", response_class=HTMLResponse)
def nerve_inspector_resolve(
    request: Request,
    state_id: int = Form(...),
    signal_id: int = Form(...),
):
    resolved = resolve_transition(state_id, signal_id)
    return templates.TemplateResponse(
        "nerve_inspector.html",
        {
            **_base_context(request, "nerve-inspector"),
            "signals": list_signals(),
            "states": list_states(),
            "resolved": resolved,
        },
    )


@app.get("/admin/plans", response_class=HTMLResponse)
def admin_plans(request: Request):
    return templates.TemplateResponse(
        "admin_plans.html",
        {
            **_base_context(request, "admin-plans"),
        },
    )


@app.get("/admin/plans/kinds/table", response_class=HTMLResponse)
def plan_kinds_table(request: Request):
    return templates.TemplateResponse(
        "partials/plan_kinds_table.html",
        {
            "request": request,
            "plan_kinds": list_plan_kinds(),
        },
    )


@app.get("/admin/plans/kinds/form", response_class=HTMLResponse)
def plan_kinds_form(request: Request):
    return templates.TemplateResponse(
        "partials/plan_kinds_form.html",
        {
            "request": request,
            "plan_kind": None,
        },
    )


@app.get("/admin/plans/kinds/{plan_kind}/form", response_class=HTMLResponse)
def plan_kinds_form_edit(request: Request, plan_kind: str):
    return templates.TemplateResponse(
        "partials/plan_kinds_form.html",
        {
            "request": request,
            "plan_kind": get_plan_kind(plan_kind),
        },
    )


@app.post("/admin/plans/kinds", response_class=HTMLResponse)
def create_plan_kind_view(
    request: Request,
    plan_kind: str = Form(...),
    description: str = Form(""),
    is_enabled: str | None = Form(None),
):
    create_plan_kind(
        {
            "plan_kind": plan_kind.strip(),
            "description": description.strip() or None,
            "is_enabled": _as_bool(is_enabled, 1),
        }
    )
    return plan_kinds_table(request)


@app.post("/admin/plans/kinds/{plan_kind}", response_class=HTMLResponse)
def update_plan_kind_view(
    request: Request,
    plan_kind: str,
    description: str = Form(""),
    is_enabled: str | None = Form(None),
):
    update_plan_kind(
        plan_kind,
        {
            "description": description.strip() or None,
            "is_enabled": _as_bool(is_enabled, 1),
        },
    )
    return plan_kinds_table(request)


@app.post("/admin/plans/kinds/{plan_kind}/delete", response_class=HTMLResponse)
def delete_plan_kind_view(request: Request, plan_kind: str):
    delete_plan_kind(plan_kind)
    return plan_kinds_table(request)


@app.get("/admin/plans/versions/table", response_class=HTMLResponse)
def plan_versions_table(request: Request, plan_kind: str | None = None):
    return templates.TemplateResponse(
        "partials/plan_versions_table.html",
        {
            "request": request,
            "plan_versions": list_plan_kind_versions(plan_kind=plan_kind),
        },
    )


@app.get("/admin/plans/versions/form", response_class=HTMLResponse)
def plan_versions_form(request: Request):
    return templates.TemplateResponse(
        "partials/plan_versions_form.html",
        {
            "request": request,
            "plan_version": None,
        },
    )


@app.get(
    "/admin/plans/versions/{plan_kind}/{plan_version}/form", response_class=HTMLResponse
)
def plan_versions_form_edit(request: Request, plan_kind: str, plan_version: int):
    return templates.TemplateResponse(
        "partials/plan_versions_form.html",
        {
            "request": request,
            "plan_version": get_plan_kind_version(plan_kind, plan_version),
        },
    )


@app.post("/admin/plans/versions", response_class=HTMLResponse)
def create_plan_version_view(
    request: Request,
    plan_kind: str = Form(...),
    plan_version: int = Form(...),
    json_schema: str = Form(...),
    example: str = Form(""),
    is_deprecated: str | None = Form(None),
):
    create_plan_kind_version(
        {
            "plan_kind": plan_kind.strip(),
            "plan_version": plan_version,
            "json_schema": json_schema.strip(),
            "example": example.strip() or None,
            "is_deprecated": _as_bool(is_deprecated, 0),
        }
    )
    return plan_versions_table(request)


@app.post(
    "/admin/plans/versions/{plan_kind}/{plan_version}", response_class=HTMLResponse
)
def update_plan_version_view(
    request: Request,
    plan_kind: str,
    plan_version: int,
    json_schema: str = Form(...),
    example: str = Form(""),
    is_deprecated: str | None = Form(None),
):
    update_plan_kind_version(
        plan_kind,
        plan_version,
        {
            "json_schema": json_schema.strip(),
            "example": example.strip() or None,
            "is_deprecated": _as_bool(is_deprecated, 0),
        },
    )
    return plan_versions_table(request)


@app.post(
    "/admin/plans/versions/{plan_kind}/{plan_version}/delete",
    response_class=HTMLResponse,
)
def delete_plan_version_view(request: Request, plan_kind: str, plan_version: int):
    delete_plan_kind_version(plan_kind, plan_version)
    return plan_versions_table(request)


@app.get("/admin/plans/executors/table", response_class=HTMLResponse)
def plan_executors_table(request: Request):
    return templates.TemplateResponse(
        "partials/plan_executors_table.html",
        {
            "request": request,
            "plan_executors": list_plan_executors(),
        },
    )


@app.get("/admin/plans/executors/form", response_class=HTMLResponse)
def plan_executors_form(request: Request):
    return templates.TemplateResponse(
        "partials/plan_executors_form.html",
        {
            "request": request,
            "plan_executor": None,
        },
    )


@app.get(
    "/admin/plans/executors/{plan_kind}/{plan_version}/form",
    response_class=HTMLResponse,
)
def plan_executors_form_edit(request: Request, plan_kind: str, plan_version: int):
    return templates.TemplateResponse(
        "partials/plan_executors_form.html",
        {
            "request": request,
            "plan_executor": get_plan_executor(plan_kind, plan_version),
        },
    )


@app.post("/admin/plans/executors", response_class=HTMLResponse)
def create_plan_executor_view(
    request: Request,
    plan_kind: str = Form(...),
    plan_version: int = Form(...),
    executor_key: str = Form(...),
    min_agent_version: str = Form(""),
):
    create_plan_executor(
        {
            "plan_kind": plan_kind.strip(),
            "plan_version": plan_version,
            "executor_key": executor_key.strip(),
            "min_agent_version": min_agent_version.strip() or None,
        }
    )
    return plan_executors_table(request)


@app.post(
    "/admin/plans/executors/{plan_kind}/{plan_version}", response_class=HTMLResponse
)
def update_plan_executor_view(
    request: Request,
    plan_kind: str,
    plan_version: int,
    executor_key: str = Form(...),
    min_agent_version: str = Form(""),
):
    update_plan_executor(
        plan_kind,
        plan_version,
        {
            "executor_key": executor_key.strip(),
            "min_agent_version": min_agent_version.strip() or None,
        },
    )
    return plan_executors_table(request)


@app.post(
    "/admin/plans/executors/{plan_kind}/{plan_version}/delete",
    response_class=HTMLResponse,
)
def delete_plan_executor_view(request: Request, plan_kind: str, plan_version: int):
    delete_plan_executor(plan_kind, plan_version)
    return plan_executors_table(request)


@app.get("/admin/plans/instances/table", response_class=HTMLResponse)
def plan_instances_table(request: Request, correlation_id: str | None = None):
    return templates.TemplateResponse(
        "partials/plan_instances_table.html",
        {
            "request": request,
            "plan_instances": list_plan_instances(correlation_id=correlation_id),
            "correlation_id": correlation_id or "",
        },
    )


@app.get("/identity/persons", response_class=HTMLResponse)
def identity_persons(request: Request):
    return templates.TemplateResponse(
        "identity_persons.html",
        {
            **_base_context(request, "identity-persons"),
        },
    )


@app.get("/identity/persons/table", response_class=HTMLResponse)
def identity_persons_table(request: Request):
    return templates.TemplateResponse(
        "partials/identity_persons_table.html",
        {
            "request": request,
            "persons": list_persons(),
        },
    )


@app.get("/identity/persons/form", response_class=HTMLResponse)
def identity_persons_form(request: Request):
    return templates.TemplateResponse(
        "partials/identity_persons_form.html",
        {
            "request": request,
            "person": None,
        },
    )


@app.get("/identity/persons/{person_id}/form", response_class=HTMLResponse)
def identity_persons_form_edit(request: Request, person_id: str):
    return templates.TemplateResponse(
        "partials/identity_persons_form.html",
        {
            "request": request,
            "person": get_person(person_id),
        },
    )


@app.post("/identity/persons", response_class=HTMLResponse)
def create_identity_person(
    request: Request,
    person_id: str = Form(...),
    display_name: str = Form(...),
    relationship: str = Form(""),
    timezone: str = Form(""),
    is_active: str | None = Form(None),
):
    create_person(
        {
            "person_id": person_id.strip(),
            "display_name": display_name.strip(),
            "relationship": relationship.strip() or None,
            "timezone": timezone.strip() or None,
            "is_active": _as_bool(is_active, 1),
        }
    )
    return identity_persons_table(request)


@app.post("/identity/persons/{person_id}", response_class=HTMLResponse)
def update_identity_person(
    request: Request,
    person_id: str,
    display_name: str = Form(...),
    relationship: str = Form(""),
    timezone: str = Form(""),
    is_active: str | None = Form(None),
):
    update_person(
        person_id,
        {
            "display_name": display_name.strip(),
            "relationship": relationship.strip() or None,
            "timezone": timezone.strip() or None,
            "is_active": _as_bool(is_active, 1),
        },
    )
    return identity_persons_table(request)


@app.post("/identity/persons/{person_id}/delete", response_class=HTMLResponse)
def delete_identity_person(request: Request, person_id: str):
    delete_person(person_id)
    return identity_persons_table(request)


@app.get("/identity/groups", response_class=HTMLResponse)
def identity_groups(request: Request):
    return templates.TemplateResponse(
        "identity_groups.html",
        {
            **_base_context(request, "identity-groups"),
        },
    )


@app.get("/identity/groups/table", response_class=HTMLResponse)
def identity_groups_table(request: Request):
    return templates.TemplateResponse(
        "partials/identity_groups_table.html",
        {
            "request": request,
            "groups": list_groups(),
        },
    )


@app.get("/identity/groups/form", response_class=HTMLResponse)
def identity_groups_form(request: Request):
    return templates.TemplateResponse(
        "partials/identity_groups_form.html",
        {
            "request": request,
            "group": None,
        },
    )


@app.get("/identity/groups/{group_id}/form", response_class=HTMLResponse)
def identity_groups_form_edit(request: Request, group_id: str):
    return templates.TemplateResponse(
        "partials/identity_groups_form.html",
        {
            "request": request,
            "group": get_group(group_id),
        },
    )


@app.post("/identity/groups", response_class=HTMLResponse)
def create_identity_group(
    request: Request,
    group_id: str = Form(...),
    name: str = Form(...),
    is_active: str | None = Form(None),
):
    create_group(
        {
            "group_id": group_id.strip(),
            "name": name.strip(),
            "is_active": _as_bool(is_active, 1),
        }
    )
    return identity_groups_table(request)


@app.post("/identity/groups/{group_id}", response_class=HTMLResponse)
def update_identity_group(
    request: Request,
    group_id: str,
    name: str = Form(...),
    is_active: str | None = Form(None),
):
    update_group(
        group_id,
        {
            "name": name.strip(),
            "is_active": _as_bool(is_active, 1),
        },
    )
    return identity_groups_table(request)


@app.post("/identity/groups/{group_id}/delete", response_class=HTMLResponse)
def delete_identity_group(request: Request, group_id: str):
    delete_group(group_id)
    return identity_groups_table(request)


@app.get("/identity/channels", response_class=HTMLResponse)
def identity_channels(request: Request):
    return templates.TemplateResponse(
        "identity_channels.html",
        {
            **_base_context(request, "identity-channels"),
        },
    )


@app.get("/identity/channels/table", response_class=HTMLResponse)
def identity_channels_table(request: Request):
    return templates.TemplateResponse(
        "partials/identity_channels_table.html",
        {
            "request": request,
            "channels": list_channels(),
        },
    )


@app.get("/identity/channels/form", response_class=HTMLResponse)
def identity_channels_form(request: Request):
    return templates.TemplateResponse(
        "partials/identity_channels_form.html",
        {
            "request": request,
            "channel": None,
        },
    )


@app.get("/identity/channels/{channel_id}/form", response_class=HTMLResponse)
def identity_channels_form_edit(request: Request, channel_id: str):
    return templates.TemplateResponse(
        "partials/identity_channels_form.html",
        {
            "request": request,
            "channel": get_channel(channel_id),
        },
    )


@app.post("/identity/channels", response_class=HTMLResponse)
def create_identity_channel(
    request: Request,
    channel_id: str = Form(...),
    channel_type: str = Form(...),
    person_id: str = Form(""),
    address: str = Form(...),
    priority: int = Form(100),
    is_enabled: str | None = Form(None),
):
    create_channel(
        {
            "channel_id": channel_id.strip(),
            "channel_type": channel_type.strip(),
            "person_id": person_id.strip() or None,
            "address": address.strip(),
            "priority": priority,
            "is_enabled": _as_bool(is_enabled, 1),
        }
    )
    return identity_channels_table(request)


@app.post("/identity/channels/{channel_id}", response_class=HTMLResponse)
def update_identity_channel(
    request: Request,
    channel_id: str,
    channel_type: str = Form(...),
    person_id: str = Form(""),
    address: str = Form(...),
    priority: int = Form(100),
    is_enabled: str | None = Form(None),
):
    update_channel(
        channel_id,
        {
            "channel_type": channel_type.strip(),
            "person_id": person_id.strip() or None,
            "address": address.strip(),
            "priority": priority,
            "is_enabled": _as_bool(is_enabled, 1),
        },
    )
    return identity_channels_table(request)


@app.post("/identity/channels/{channel_id}/delete", response_class=HTMLResponse)
def delete_identity_channel(request: Request, channel_id: str):
    delete_channel(channel_id)
    return identity_channels_table(request)


@app.get("/identity/prefs", response_class=HTMLResponse)
def identity_prefs(request: Request):
    return templates.TemplateResponse(
        "identity_prefs.html",
        {
            **_base_context(request, "identity-prefs"),
        },
    )


@app.get("/identity/prefs/table", response_class=HTMLResponse)
def identity_prefs_table(request: Request):
    return templates.TemplateResponse(
        "partials/identity_prefs_table.html",
        {
            "request": request,
            "prefs": list_prefs(),
        },
    )


@app.get("/identity/prefs/form", response_class=HTMLResponse)
def identity_prefs_form(request: Request):
    return templates.TemplateResponse(
        "partials/identity_prefs_form.html",
        {
            "request": request,
            "prefs": None,
        },
    )


@app.get("/identity/prefs/{prefs_id}/form", response_class=HTMLResponse)
def identity_prefs_form_edit(request: Request, prefs_id: str):
    return templates.TemplateResponse(
        "partials/identity_prefs_form.html",
        {
            "request": request,
            "prefs": get_prefs(prefs_id),
        },
    )


@app.post("/identity/prefs", response_class=HTMLResponse)
def create_identity_prefs(
    request: Request,
    prefs_id: str = Form(...),
    scope_type: str = Form("person"),
    scope_id: str = Form(...),
    language_preference: str = Form(""),
    tone: str = Form(""),
    formality: str = Form(""),
    emoji: str = Form(""),
    verbosity_cap: str = Form(""),
    quiet_hours_start: str = Form(""),
    quiet_hours_end: str = Form(""),
    allow_push: str | None = Form(None),
    allow_telegram: str | None = Form(None),
    allow_web: str | None = Form(None),
    allow_cli: str | None = Form(None),
    model_budget_policy: str = Form(""),
):
    create_prefs(
        {
            "prefs_id": prefs_id.strip(),
            "scope_type": scope_type.strip(),
            "scope_id": scope_id.strip(),
            "language_preference": language_preference.strip() or None,
            "tone": tone.strip() or None,
            "formality": formality.strip() or None,
            "emoji": emoji.strip() or None,
            "verbosity_cap": verbosity_cap.strip() or None,
            "quiet_hours_start": int(quiet_hours_start) if quiet_hours_start else None,
            "quiet_hours_end": int(quiet_hours_end) if quiet_hours_end else None,
            "allow_push": _as_bool(allow_push, 1),
            "allow_telegram": _as_bool(allow_telegram, 1),
            "allow_web": _as_bool(allow_web, 1),
            "allow_cli": _as_bool(allow_cli, 1),
            "model_budget_policy": model_budget_policy.strip() or None,
        }
    )
    return identity_prefs_table(request)


@app.post("/identity/prefs/{prefs_id}", response_class=HTMLResponse)
def update_identity_prefs(
    request: Request,
    prefs_id: str,
    scope_type: str = Form("person"),
    scope_id: str = Form(...),
    language_preference: str = Form(""),
    tone: str = Form(""),
    formality: str = Form(""),
    emoji: str = Form(""),
    verbosity_cap: str = Form(""),
    quiet_hours_start: str = Form(""),
    quiet_hours_end: str = Form(""),
    allow_push: str | None = Form(None),
    allow_telegram: str | None = Form(None),
    allow_web: str | None = Form(None),
    allow_cli: str | None = Form(None),
    model_budget_policy: str = Form(""),
):
    update_prefs(
        prefs_id,
        {
            "scope_type": scope_type.strip(),
            "scope_id": scope_id.strip(),
            "language_preference": language_preference.strip() or None,
            "tone": tone.strip() or None,
            "formality": formality.strip() or None,
            "emoji": emoji.strip() or None,
            "verbosity_cap": verbosity_cap.strip() or None,
            "quiet_hours_start": int(quiet_hours_start) if quiet_hours_start else None,
            "quiet_hours_end": int(quiet_hours_end) if quiet_hours_end else None,
            "allow_push": _as_bool(allow_push, 1),
            "allow_telegram": _as_bool(allow_telegram, 1),
            "allow_web": _as_bool(allow_web, 1),
            "allow_cli": _as_bool(allow_cli, 1),
            "model_budget_policy": model_budget_policy.strip() or None,
        },
    )
    return identity_prefs_table(request)


@app.post("/identity/prefs/{prefs_id}/delete", response_class=HTMLResponse)
def delete_identity_prefs(request: Request, prefs_id: str):
    delete_prefs(prefs_id)
    return identity_prefs_table(request)


@app.get("/identity/presence", response_class=HTMLResponse)
def identity_presence(request: Request):
    return templates.TemplateResponse(
        "identity_presence.html",
        {
            **_base_context(request, "identity-presence"),
        },
    )


@app.get("/identity/presence/table", response_class=HTMLResponse)
def identity_presence_table(request: Request):
    return templates.TemplateResponse(
        "partials/identity_presence_table.html",
        {
            "request": request,
            "presence": list_presence(),
        },
    )


@app.get("/identity/presence/form", response_class=HTMLResponse)
def identity_presence_form(request: Request):
    return templates.TemplateResponse(
        "partials/identity_presence_form.html",
        {
            "request": request,
            "presence": None,
        },
    )


@app.get("/identity/presence/{person_id}/form", response_class=HTMLResponse)
def identity_presence_form_edit(request: Request, person_id: str):
    return templates.TemplateResponse(
        "partials/identity_presence_form.html",
        {
            "request": request,
            "presence": get_presence(person_id),
        },
    )


@app.post("/identity/presence", response_class=HTMLResponse)
def upsert_identity_presence(
    request: Request,
    person_id: str = Form(...),
    in_meeting: str | None = Form(None),
    location_hint: str = Form(""),
):
    upsert_presence(
        {
            "person_id": person_id.strip(),
            "in_meeting": _as_bool(in_meeting, 0),
            "location_hint": location_hint.strip() or None,
        }
    )
    return identity_presence_table(request)
