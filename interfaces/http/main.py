from fastapi import FastAPI
from core.context.awareness import get_awareness_snapshot

app = FastAPI(
    title="Atrium",
    description="Domestic presence interface",
    version="0.1.0"
)

@app.get("/status")
def get_status():
    snapshot = get_awareness_snapshot()

    time_ctx = snapshot["time"]
    system_ctx = snapshot["system"]

    message = (
        f"Atrium is online. "
        f"It is {time_ctx['day_period']}, "
        f"and the system has been running steadily."
    )

    return {
        "message": message,
        "awareness": snapshot
    }
