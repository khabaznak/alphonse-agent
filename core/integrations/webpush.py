import json
import os
from pathlib import Path

from pywebpush import WebPushException, webpush


def _load_dotenv_if_missing() -> None:
    if os.getenv("VAPID_PRIVATE_KEY") and os.getenv("VAPID_PUBLIC_KEY") and os.getenv("VAPID_EMAIL"):
        return
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def send_web_push(subscription: dict, title: str, body: str, data: dict | None = None) -> None:
    _load_dotenv_if_missing()
    private_key = os.getenv("VAPID_PRIVATE_KEY")
    public_key = os.getenv("VAPID_PUBLIC_KEY")
    email = os.getenv("VAPID_EMAIL")

    if not private_key or not public_key or not email:
        raise ValueError("Missing VAPID configuration (VAPID_PRIVATE_KEY/PUBLIC_KEY/EMAIL).")

    payload = {
        "title": title,
        "body": body,
        "data": data or {},
    }

    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=private_key,
            vapid_claims={"sub": f"mailto:{email}"},
        )
    except WebPushException as exc:  # noqa: BLE001
        raise RuntimeError(str(exc)) from exc
