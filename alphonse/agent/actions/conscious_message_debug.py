from __future__ import annotations

import json
import os


def pack_raw_provider_event_markdown(*, channel_type: str, payload: dict[str, object], correlation_id: str) -> str:
    render_mode = str(os.getenv("ALPHONSE_PROVIDER_EVENT_RENDER_MODE") or "json").strip().lower()
    if render_mode == "markdown":
        return pack_raw_provider_event_as_markdown(
            channel_type=channel_type,
            payload=payload,
            correlation_id=correlation_id,
        )
    return (
        "## RAW MESSAGE\n"
        "\n"
        f"- channel: {channel_type}\n"
        f"- correlation_id: {correlation_id}\n\n"
        "## RAW JSON\n"
        "\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )


def pack_raw_provider_event_as_markdown(
    *,
    channel_type: str,
    payload: dict[str, object],
    correlation_id: str,
) -> str:
    lines = [
        "## RAW MESSAGE",
        "",
        f"- channel: {channel_type}",
        f"- correlation_id: {correlation_id}",
        "",
        "## RAW MESSAGE FIELDS",
    ]
    lines.extend(render_json_as_markdown(payload, level=0))
    return "\n".join(lines).rstrip() + "\n"


def render_json_as_markdown(value: object, *, level: int) -> list[str]:
    indent = "  " * level
    if isinstance(value, dict):
        lines: list[str] = []
        if not value:
            return [f"{indent}- {{}}"]
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{indent}- {key_text}:")
                lines.extend(render_json_as_markdown(item, level=level + 1))
            else:
                lines.append(f"{indent}- {key_text}: {render_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{indent}- []"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{indent}-")
                lines.extend(render_json_as_markdown(item, level=level + 1))
            else:
                lines.append(f"{indent}- {render_scalar(item)}")
        return lines
    return [f"{indent}- {render_scalar(value)}"]


def render_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
