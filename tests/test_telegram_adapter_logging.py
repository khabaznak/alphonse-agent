from __future__ import annotations

import logging

from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter


def test_telegram_empty_poll_logs_debug_not_info(caplog) -> None:
    adapter = TelegramAdapter(
        {"bot_token": "fake-token", "poll_interval_sec": 0.0, "poll_summary_interval_sec": 9999.0}
    )
    adapter._running = True

    def _fetch_updates(_offset: int) -> list[dict]:
        adapter._running = False
        return []

    adapter._fetch_updates = _fetch_updates  # type: ignore[method-assign]
    adapter._handle_update = lambda _u: None  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG):
        adapter._run_polling()

    debug_lines = [r for r in caplog.records if r.levelno == logging.DEBUG and "getUpdates" in r.message]
    info_empty_lines = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "getUpdates" in r.message and "returned=0" in r.message
    ]
    assert debug_lines
    assert not info_empty_lines


def test_telegram_poll_summary_logs_at_info(caplog) -> None:
    adapter = TelegramAdapter(
        {"bot_token": "fake-token", "poll_interval_sec": 0.0, "poll_summary_interval_sec": 0.0}
    )
    adapter._running = True

    def _fetch_updates(_offset: int) -> list[dict]:
        adapter._running = False
        return []

    adapter._fetch_updates = _fetch_updates  # type: ignore[method-assign]
    adapter._handle_update = lambda _u: None  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO):
        adapter._run_polling()

    assert any("poll summary" in r.message for r in caplog.records if r.levelno == logging.INFO)
