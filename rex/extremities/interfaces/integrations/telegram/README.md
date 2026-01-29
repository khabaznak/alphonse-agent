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

Run:
- `python -m rex.run_telegram_echo_heart`

Send a message to the bot and it will echo back.
