from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from interfaces.http.routes.api import router as api_router
from rex.cognition.status_reasoning import reason_about_status

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

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    top_nav_links = [
        {"label": "Rex", "href": "/", "active": True},
        {"label": "Status JSON", "href": "/status"},
    ]
    side_nav_links = [
        {"label": "Current State", "href": "#current-state", "active": True},
        {"label": "Diagnostics", "href": "#diagnostics"},
        {"label": "Presence Log", "href": "#presence-log"},
    ]
    return templates.TemplateResponse(
        "rex.html",
        {
            "request": request,
            "top_nav_links": top_nav_links,
            "side_nav_links": side_nav_links,
        },
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
