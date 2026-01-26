import json
import os

import firebase_admin
from firebase_admin import credentials, messaging


def get_fcm_app():
    if firebase_admin._apps:
        return firebase_admin.get_app()

    raw_json = os.getenv("FCM_CREDENTIALS_JSON")
    if not raw_json:
        raise ValueError("Missing FCM_CREDENTIALS_JSON env var.")

    service_account = json.loads(raw_json)
    cred = credentials.Certificate(service_account)
    return firebase_admin.initialize_app(cred)


def send_push_notification(tokens: list[str], title: str, body: str, data: dict | None = None) -> int:
    if not tokens:
        return 0

    get_fcm_app()

    message = messaging.MulticastMessage(
        notification=messaging.Notification(title=title, body=body),
        data={str(key): str(value) for key, value in (data or {}).items()},
        tokens=tokens,
    )
    response = messaging.send_multicast(message)
    return response.success_count
