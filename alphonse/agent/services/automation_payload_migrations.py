from __future__ import annotations

import json
import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.services.automation_tool_call_contract import is_canonical_tool_call, to_canonical_tool_call


def migrate_timed_signal_tool_call_payloads(*, dry_run: bool = False, sample_limit: int = 20) -> dict[str, Any]:
    scanned = 0
    updated = 0
    invalid = 0
    samples: list[str] = []
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, payload FROM timed_signals ORDER BY updated_at DESC").fetchall()
        for row in rows:
            signal_id = str(row[0] or "").strip()
            payload = _parse_payload(row[1])
            if not isinstance(payload, dict):
                continue
            if not _looks_like_tool_call_payload(payload):
                continue
            scanned += 1
            if is_canonical_tool_call(payload):
                continue
            try:
                canonical = to_canonical_tool_call(payload, allow_legacy=True)
            except ValueError:
                invalid += 1
                if len(samples) < max(1, int(sample_limit)):
                    samples.append(signal_id)
                continue
            updated += 1
            if len(samples) < max(1, int(sample_limit)):
                samples.append(signal_id)
            if not dry_run:
                conn.execute(
                    "UPDATE timed_signals SET payload = ?, updated_at = datetime('now') WHERE id = ?",
                    (json.dumps(canonical, ensure_ascii=False, separators=(",", ":")), signal_id),
                )
        if not dry_run:
            conn.commit()
    return {
        "scanned": scanned,
        "updated": updated,
        "invalid": invalid,
        "sample_ids": samples,
        "dry_run": bool(dry_run),
    }


def _looks_like_tool_call_payload(payload: dict[str, Any]) -> bool:
    if is_canonical_tool_call(payload):
        return True
    if str(payload.get("payload_type") or "").strip().lower() == "tool_call":
        return True
    return bool(str(payload.get("tool_key") or payload.get("tool_name") or payload.get("tool") or "").strip())


def _parse_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}
