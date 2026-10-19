# Review sheet

Agent inspection only. Independent human reviewer: **pending**.

These are the latest targeted diagnostics for each responsibility, not a single matched five-case run. Raw reports preserve all earlier failures and source revisions. No success rate is inferred.

## Historical recall

Report: [response-stage-anniversary](household-response-stage-anniversary.json.gz). Task status: `completed`. Original mechanical score: **pass**.

Observed response:

> Last month, you went to Casa del Lago in Chapultepec for dinner for your anniversary.

Calls: 3 System Two, 21 Jev. Total elapsed: 45.7 seconds.

Failed mechanical checks: none.

## Reminder for another household member

Report: [final-mother-medication-reminder](household-final-mother-medication-reminder.json.gz). Task status: `completed`. Original mechanical score: **pass**.

Observed response:

> Elena’s daily medicine reminder has been successfully scheduled to go directly to her Telegram, starting October 19 at 8:00 a.m. Mexico City time. It says to take one prescribed tablet of synthetic Medicine A, as directed by her clinician. This confirms the reminder is scheduled; it does not establish that it has been delivered.

Calls: 5 System Two, 29 Jev. Total elapsed: 55.6 seconds.

Failed mechanical checks: none.

Fixture effect: audience `fixture-mother`, destination `fixture-mother-chat`, outbox status `pending`. Body: Reminder: Elena, please take one prescribed tablet of synthetic Medicine A.

## Recovery through a different route

Report: [final-sensor-recovery](household-final-sensor-recovery.json.gz). Task status: `completed`. Original mechanical score: **pass**.

Observed response:

> The current study temperature is 24 °C. The primary route was unavailable, so I obtained this fresh reading using the backup route.

Calls: 13 System Two, 33 Jev. Total elapsed: 124.4 seconds.

Failed mechanical checks: none.

## Visible failure

Report: [scoped-sensor-visible-failure](household-scoped-sensor-visible-failure.json.gz). Task status: `completed`. Original mechanical score: **pass**.

Observed response:

> I couldn’t get a fresh study temperature. I tried both live sensor alternatives: the primary and backup routes each reported that the route was unavailable and returned no measurement. I don’t have a current reading to report.

Calls: 9 System Two, 24 Jev. Total elapsed: 114.2 seconds.

Failed mechanical checks: none.

## Reusable skill

Report: [learning-v2-instruction-skill](household-learning-v2-instruction-skill.json.gz). Task status: `failed`. Original mechanical score: **fail**.

Observed response:

> I couldn't complete this task because a required step failed. The requested result has not been fully verified. Any completed work has been preserved.

Calls: 24 System Two, 69 Jev. Total elapsed: 283.3 seconds.

Failed mechanical checks: `answer_fragments_present`, `task_completed`, `repair_reused_before_failed_route`.

The approved definition was installed and selected by a fresh task. That repeat task still confused a skill name with an artifact ID and reached the shared call budget. The Bash schema was subsequently clarified; end-to-end reuse remains unverified.

The recall source says “wedding celebration”; the final response says “anniversary.” Human review must assess that contextual link. The venue check alone does not settle it.

## Human review still required

- Does each answer match its source evidence and verified effects?
- Are limitations understandable and actionable?
- Did the agent perform the requested responsibility, rather than merely describe it?
- Are the number of calls and elapsed time acceptable for everyday delegation?

Artifact implementation repair is a separate next case: use the relevant skill expertise to improve an existing disposable artifact in its registered storage and verify the requested result. The instruction-only scenario does not establish that behavior.
