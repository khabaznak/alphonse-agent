import json
import os

from pywebpush import WebPushException, webpush


def send_web_push(subscription: dict, title: str, body: str, data: dict | None = None) -> None:
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
