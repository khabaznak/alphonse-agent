from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Any

from alphonse.agent.tools.registry import build_default_tool_registry


INSTRUCTIONS = """Create this helper in Home Assistant before running the smoke test:

Option A (UI):
- Settings -> Devices & Services -> Helpers -> Create Helper -> Toggle
- Name: Alphonse Test Toggle
- Entity ID should be: input_boolean.alphonse_test_toggle

Option B (configuration.yaml):
input_boolean:
  alphonse_test_toggle:
    name: Alphonse Test Toggle

Then restart Home Assistant and run this smoke command.
"""


def run_toggle_smoke(*, entity_id: str, duration_seconds: float = 10.0) -> dict[str, Any]:
    registry = build_default_tool_registry()
    query_tool = registry.get("domotics.query")
    subscribe_tool = registry.get("domotics.subscribe")
    execute_tool = registry.get("domotics.execute")

    if not query_tool or not subscribe_tool or not execute_tool:
        return {
            "status": "failed",
            "result": None,
            "error": {
                "code": "domotics_tools_not_registered",
                "message": "domotics.query/domotics.execute/domotics.subscribe must be registered",
                "retryable": False,
                "details": {},
            },
            "metadata": {"tool": "domotics.smoke_toggle"},
        }

    before = query_tool.execute(kind="state", entity_id=entity_id)

    subscribe_result: dict[str, Any] = {}

    def _run_subscribe() -> None:
        nonlocal subscribe_result
        subscribe_result = subscribe_tool.execute(
            event_type="state_changed",
            duration_seconds=duration_seconds,
            max_events=200,
        )

    sub_thread = threading.Thread(target=_run_subscribe, daemon=True)
    sub_thread.start()

    time.sleep(1.0)
    turned_on = execute_tool.execute(
        domain="input_boolean",
        service="turn_on",
        target={"entity_id": entity_id},
        readback=True,
        readback_entity_id=entity_id,
        expected_state="on",
    )
    time.sleep(1.0)
    turned_off = execute_tool.execute(
        domain="input_boolean",
        service="turn_off",
        target={"entity_id": entity_id},
        readback=True,
        readback_entity_id=entity_id,
        expected_state="off",
    )

    sub_thread.join(timeout=max(1.0, duration_seconds + 2.0))
    after = query_tool.execute(kind="state", entity_id=entity_id)

    return {
        "status": "ok",
        "result": {
            "entity_id": entity_id,
            "before": before,
            "subscribe": subscribe_result,
            "execute_on": turned_on,
            "execute_off": turned_off,
            "after": after,
            "instructions": INSTRUCTIONS,
        },
        "error": None,
        "metadata": {"tool": "domotics.smoke_toggle"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Home Assistant domotics smoke test")
    parser.add_argument(
        "--entity-id",
        default="input_boolean.alphonse_test_toggle",
        help="Entity ID to toggle during smoke test",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
        help="Subscription capture window in seconds",
    )
    parser.add_argument(
        "--print-instructions",
        action="store_true",
        help="Print helper setup instructions and exit",
    )
    args = parser.parse_args()

    if args.print_instructions:
        print(INSTRUCTIONS.strip())
        return

    result = run_toggle_smoke(
        entity_id=str(args.entity_id),
        duration_seconds=float(args.duration_seconds),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
