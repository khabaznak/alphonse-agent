# Telegram activity Dojo

Run `python -m alphonse.agent_v2.evaluation.dojo_telegram_activity` in the project environment. The exercise injects an authorized synthetic Telegram owner message into the actual owner listener and worker. A deterministic fake model is held for six seconds, then returns a synthetic reply. A fake Telegram client records reactions, typing actions and replies; no real Telegram message or household data is used.

The exercise checks for a reaction bound to the incoming message or a typing action in the same chat while the model is held, before the one reply. A missing reply or an unheld model is inconclusive. This exercise does not verify real Telegram delivery, real-model behavior, exact timing guarantees, or reliability over multiple runs.

The initial observation on 2026-11-23 was a **failure**: the model remained held for six seconds, then the reply was sent; no reaction or typing action was recorded. That historical observation predates the owner listener's feedback implementation. Focused tests now check the separate receipt reaction and typing refresh, including failure and cancellation.

On 2026-11-25, the focused synthetic tests also held a feedback API call in
flight while the owner worker completed or was cancelled. In these runs, a
blocked typing call did not delay the reply or replay the model, and no new
typing call started after the outcome. When released, however, the already
started call produced its fake typing event after the reply or cancellation.
A separately blocked receipt likewise did not block the reply and appeared
only after release. These observations expose a real ordering limit: stopping
local refresh cannot retract a request already in flight. They do not prove
when or whether Telegram would display an action, how long it would remain
visible, or how a real provider behaves under delay. The tests use disposable
data and an in-process fake, and do not assess real-model performance.

A scripted clarification case also checks that typing stops and one question
is delivered when the worker needs input. Its clarification judgment is a
fixture; it does not assess a live decision model or Telegram delivery.
