from __future__ import annotations

from alphonse.agent_v2.cli import format_status
from alphonse.agent_v2.cli import build_parser


def test_cli_exposes_daemon_lifecycle_commands() -> None:
    assert build_parser().parse_args(["start"]).command == "start"
    assert build_parser().parse_args(["status"]).command == "status"
    assert build_parser().parse_args(["stop"]).command == "stop"


def test_format_status_renders_daemon_and_scheduler_health() -> None:
    rendered = format_status(
        {
            "daemon_id": "daemon-test",
            "queue_size": 2,
            "outbound_queue_size": 1,
            "due_schedules": 3,
            "scheduler": {"ticks": 10, "claimed": 4, "queued": 3, "retried": 1},
        }
    )

    assert "Alphonse daemon: running" in rendered
    assert "daemon id: daemon-test" in rendered
    assert "inbound queue: 2" in rendered
    assert "outbound queue: 1" in rendered
    assert "scheduled retries: 1" in rendered
