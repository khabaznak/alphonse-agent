from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "rex.html",
        {"request": request},
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
