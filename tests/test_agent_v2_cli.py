from __future__ import annotations

from alphonse.agent_v2.cli import format_status
from alphonse.agent_v2.cli import format_active_work
from alphonse.agent_v2.cli import format_unavailable_status
from alphonse.agent_v2.cli import build_parser
from alphonse.agent_v2.cli import _wait_for_shutdown


def test_cli_exposes_daemon_lifecycle_commands() -> None:
    assert build_parser().parse_args(["start"]).command == "start"
    assert build_parser().parse_args(["start", "tui"]).start_target == "tui"
    assert build_parser().parse_args(["status"]).command == "status"
    assert build_parser().parse_args(["stop"]).command == "stop"


def test_format_status_renders_daemon_and_scheduler_health() -> None:
    rendered = format_status(
        {
            "daemon_id": "daemon-test",
            "queue_size": 2,
            "outbound_queue_size": 1,
            "inbound_counts": {"processing": 1, "retry_wait": 2},
            "outbound_counts": {"claimed": 1, "retry_wait": 0},
            "active_work": {"user": "alex", "prompt": "Reply to the active request."},
            "due_schedules": 3,
            "scheduler": {"ticks": 10, "claimed": 4, "queued": 3, "retried": 1},
        }
    )

    assert "Alphonse daemon: running" in rendered
    assert "daemon id: daemon-test" in rendered
    assert "inbound queue: 2 ready, 1 processing, 2 retrying, 0 failed" in rendered
    assert "outbound queue: 1 ready, 1 delivering, 0 retrying" in rendered
    assert "active work: alex: Reply to the active request." in rendered
    assert "scheduled retries: 1" in rendered


def test_wait_for_shutdown_detects_unreachable_daemon() -> None:
    class UnreachableClient:
        def ping(self) -> None:
            raise OSError("socket unavailable")

    assert _wait_for_shutdown(UnreachableClient(), timeout_sec=0.5) is True


def test_unavailable_status_explains_that_daemon_is_stopped() -> None:
    assert format_unavailable_status(FileNotFoundError("socket missing")) == (
        "Alphonse daemon: stopped. Run 'alphonse start' to launch it."
    )


def test_active_work_formatting_collapses_whitespace_and_truncates() -> None:
    assert format_active_work({"user": "alex", "prompt": "  Reply\nnow  "}) == "alex: Reply now"
