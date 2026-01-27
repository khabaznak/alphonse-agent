from datetime import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.repositories.agentic_notifications import (
    create_notification,
    get_notification,
    list_notifications,
    list_unexecuted_notifications,
    update_notification,
)
from core.repositories.family import (
    create_family_member,
    get_family_member,
    list_family_members,
    update_family_member,
)
from core.repositories.push_devices import list_push_devices
from interfaces.http.routes.api import router as api_router, trigger_router
from rex.cognition.notification_reasoning import reason_about_execution_target
from rex.cognition.provider_selector import get_provider_info
from rex.cognition.status_reasoning import reason_about_status
from rex.config import load_rex_config

load_dotenv()

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
        {"label": "Rex", "href": "/", "active": active == "rex"},
        {"label": "Notifications", "href": "/notifications", "active": active == "notifications"},
        {"label": "Family", "href": "/family", "active": active == "family"},
        {"label": "Push Test", "href": "/push-test", "active": active == "push-test"},
        {"label": "Status JSON", "href": "/status"},
    ]


def _side_nav_links(page: str) -> list[dict[str, str | bool]]:
    if page == "notifications":
        return [
            {"label": "Execution Target", "href": "#execution-target"},
            {"label": "Pending Queue", "href": "#pending-queue"},
            {"label": "All Notifications", "href": "#all-notifications"},
            {"label": "Edit Notification", "href": "#edit-notification"},
            {"label": "Create Notification", "href": "#create-notification"},
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
    return [
        {"label": "Current State", "href": "#current-state", "active": True},
        {"label": "Notifications", "href": "/notifications"},
        {"label": "Family", "href": "/family"},
        {"label": "Push Test", "href": "/push-test"},
        {"label": "Diagnostics", "href": "#diagnostics"},
        {"label": "Presence Log", "href": "#presence-log"},
    ]


def _model_status() -> str:
    info = get_provider_info(load_rex_config())
    return f"{info['mode']} · {info['provider']} / {info['model']}"


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
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.isoformat()


def _format_datetime_local(value: str | None) -> str:
    parsed = _parse_iso_datetime(value)
    if not parsed:
        return ""
    if parsed.tzinfo:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y-%m-%dT%H:%M")


def _format_date_local(value: str | None) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    if value:
        return value[:10]
    return ""


def _select_due_notification(
    notifications: list[dict[str, str | None]],
    now: datetime,
) -> tuple[dict[str, str | None] | None, list[dict[str, str | None]]]:
    due = None
    due_time = None
    for notification in notifications:
        event_datetime = _parse_iso_datetime(
            notification.get("event_datetime")  # type: ignore[arg-type]
        )
        if not event_datetime:
            continue
        if event_datetime <= now and (due_time is None or event_datetime > due_time):
            due = notification
            due_time = event_datetime
    pending = []
    for notification in notifications:
        if due and notification.get("id") == due.get("id"):
            continue
        pending.append(notification)
    return due, pending


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "rex.html",
        _base_context(request, "rex"),
    )


@app.get("/status/fragment", response_class=HTMLResponse)
def get_status_fragment(request: Request):
    message, snapshot = reason_about_status()
    return templates.TemplateResponse(
        "partials/status.html",
        {
            "request": request,
            "message": message.strip(),
            "snapshot": snapshot,
        },
    )


@app.get("/status")
def get_status():
    message, snapshot = reason_about_status()

    return {
        "message": message,
        "awareness": snapshot
    }


@app.get("/notifications", response_class=HTMLResponse)
def notifications(request: Request, edit_id: str | None = None):
    now = datetime.now().astimezone()
    unexecuted = list_unexecuted_notifications(limit=200)
    due_notification, pending_notifications = _select_due_notification(unexecuted, now)
    all_notifications = list_notifications(limit=200)
    edit_notification = get_notification(edit_id) if edit_id else None
    family_members = list_family_members(limit=200)

    owner_lookup = {
        member.get("id"): f"{member.get('name', 'Unknown')} ({member.get('role', 'member')})"
        for member in family_members
    }

    execution_interpretation = reason_about_execution_target(
        due_notification,
        owner_lookup.get(due_notification.get("owner_id")) if due_notification else None,
    )

    return templates.TemplateResponse(
        "notifications.html",
        {
            **_base_context(request, "notifications"),
            "due_notification": due_notification,
            "execution_interpretation": execution_interpretation,
            "pending_notifications": pending_notifications,
            "all_notifications": all_notifications,
            "edit_notification": edit_notification,
            "family_members": family_members,
            "owner_lookup": owner_lookup,
            "format_dt_local": _format_datetime_local,
            "now_iso": now.isoformat(),
        },
    )


@app.post("/notifications")
def create_notifications(
    owner_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    event_datetime: str = Form(...),
    target_group: str = Form("all"),
    recurrence: str = Form(""),
):
    payload = {
        "owner_id": owner_id,
        "title": title,
        "description": description or None,
        "event_datetime": _normalize_datetime(event_datetime),
        "target_group": target_group or "all",
        "recurrence": recurrence or None,
        "execution_status": "pending",
    }
    created = create_notification(payload)
    notification_id = created.get("id")
    location = f"/notifications?edit_id={notification_id}" if notification_id else "/notifications"
    return RedirectResponse(location, status_code=303)


@app.post("/notifications/{notification_id}")
def update_notifications(
    notification_id: str,
    owner_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    event_datetime: str = Form(...),
    target_group: str = Form("all"),
    recurrence: str = Form(""),
):
    payload = {
        "owner_id": owner_id,
        "title": title,
        "description": description or None,
        "event_datetime": _normalize_datetime(event_datetime),
        "target_group": target_group or "all",
        "recurrence": recurrence or None,
    }
    update_notification(notification_id, payload)
    return RedirectResponse(f"/notifications?edit_id={notification_id}", status_code=303)


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
