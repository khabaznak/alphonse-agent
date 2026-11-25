# Conversation UX: focused Dojo observation, 2026-11-25

Two disposable `dojo_held_followups` exercises were run against the owner IPC and simpler loop. Both used the scripted provider; neither assessed real Jev, semantic understanding, the Desktop product path, or external effects. The first sandboxed attempt could not bind its local Unix socket. The exercises then ran with local socket access.

- `scenario="steer"`: the update was accepted and subsequently included in planning; after the original answer was ready, a late update was not accepted. Two scripted model calls were made. Restart and cancellation were not assessed. Inclusion did not establish the semantic task outcome.
- `scenario="normal"`: three follow-ups were queued, one removed, and the remaining two were held until their predecessors' answers were acknowledged, then released and completed in order. Three scripted model calls were made. Semantic behavior, steering, restart, and uncertain delivery were not assessed in this case.

The observations are plumbing evidence, not proof of general reliability or authority. The new suggestion component and routing decisions have separate simulated tests. No live Jev evaluation or multilingual behavioral qualification was performed.
