# Telegram Adapter

## Config

Required:
- `bot_token` (string)

Optional:
- `allowed_chat_ids` (list[int])
- `poll_interval_sec` (float, default 1.0)

## Basic test

Set env vars:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_CHAT_ID` (optional)

Run with the main entrypoint:
- `ALPHONSE_ENABLE_TELEGRAM=true python -m alphonse.agent.main`

Legacy test harness:
- `ALPHONSE_ENABLE_TELEGRAM=true python scripts/run_telegram_echo_heart.py`

Send a message to the bot and it will echo back.
