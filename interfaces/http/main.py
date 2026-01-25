from fastapi import FastAPI

from rex.cognition.status_reasoning import reason_about_status

app = FastAPI(
    title="Atrium",
    description="Domestic presence interface",
    version="0.1.0"
)

@app.get("/status")
def get_status():
    message, snapshot = reason_about_status()

    return {
        "message": message,
        "awareness": snapshot
    }
