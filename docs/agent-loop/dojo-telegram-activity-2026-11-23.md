# Telegram activity Dojo

Run `python -m alphonse.agent_v2.evaluation.dojo_telegram_activity` in the project environment. The exercise injects an authorized synthetic Telegram owner message into the actual owner listener and worker. A deterministic fake model is held for six seconds, then returns a synthetic reply. A fake Telegram client records reactions, typing actions and replies; no real Telegram message or household data is used.

The exercise checks for a reaction bound to the incoming message or a typing action in the same chat while the model is held, before the one reply. A missing reply or an unheld model is inconclusive. This exercise does not verify real Telegram delivery, real-model behavior, exact timing guarantees, or reliability over multiple runs.

The initial observation on 2026-11-23 was a **failure**: the model remained held for six seconds, then the reply was sent; no reaction or typing action was recorded. The legacy `TelegramPresenceAdapter` supports reactions and typing, but the current owner-only Telegram listener and worker do not use its presence lifecycle. This observation should be revisited as an objective regression when the integration is connected.
