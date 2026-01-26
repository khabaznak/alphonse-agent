from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = {
    "mode": "test",
    "providers": {
        "test": {
            "type": "ollama",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "mistral:7b-instruct",
                "timeout": 120,
            },
        },
        "production": {
            "type": "openai",
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "timeout": 60,
            },
        },
    },
    "notifications": {
        "overdue_minutes": 30,
    },
    "push": {
        "provider": "fcm",
        "fcm": {
            "credentials_env": "FCM_CREDENTIALS_JSON",
        },
        "webpush": {
            "vapid_private_key_env": "VAPID_PRIVATE_KEY",
            "vapid_public_key_env": "VAPID_PUBLIC_KEY",
            "vapid_email_env": "VAPID_EMAIL",
        },
    },
}


def load_rex_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or PROJECT_ROOT / "config" / "rex.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _merge_dicts(DEFAULT_CONFIG, data)


def _merge_dicts(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in defaults.items():
        if key not in override:
            merged[key] = value
            continue
        if isinstance(value, dict) and isinstance(override[key], dict):
            merged[key] = _merge_dicts(value, override[key])
        else:
            merged[key] = override[key]

    for key, value in override.items():
        if key not in merged:
            merged[key] = value

    return merged
