from __future__ import annotations

import logging
from urllib.error import HTTPError

from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.extremities.interfaces.integrations.telegram import telegram_adapter as telegram_module


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


def test_telegram_http_409_conflict_logs_warning_once_and_backs_off(caplog, monkeypatch) -> None:
    adapter = TelegramAdapter(
        {
            "bot_token": "fake-token",
            "poll_interval_sec": 0.0,
            "poll_conflict_backoff_sec": 2.0,
            "poll_conflict_log_interval_sec": 9999.0,
        }
    )

    def _raise_conflict(*_args, **_kwargs):
        raise HTTPError(
            url="https://api.telegram.org/botfake-token/getUpdates",
            code=409,
            msg="Conflict",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(telegram_module.request, "urlopen", _raise_conflict)

    with caplog.at_level(logging.WARNING):
        first = adapter._fetch_updates(offset=1)
        second = adapter._fetch_updates(offset=1)

    assert first is None
    assert second is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "conflict (409)" in r.message]
    assert len(warnings) == 1
    assert adapter._poll_conflict_backoff_until > 0
