from fastapi import FastAPI

from interfaces.http.routes.api import router as api_router, trigger_router

app = FastAPI(title="Alphonse Agent API", version="0.1.0")

app.include_router(api_router)
app.include_router(trigger_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
