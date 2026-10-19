# Household delegation pilot — 2026-10-18

This exploratory pilot tests whether memory and recovery help Alphonse carry out
household responsibilities. All people, medicines, sensor readings and destinations
are synthetic. No real reminder, message, skill or household data was changed.

The real V3 processor and native tools ran with live System Two and Jev providers,
real disposable stores, an allowlisted sensor fixture, and a fixture outbox. A queued
outbox entry proves routing and message composition, not provider delivery.

## Evidence and interpretation

[summary.json](summary.json) indexes every completed attempt, its raw/compressed
SHA-256, checks, answer, call counts, reported usage, latency and disposition. Each
`.json.gz` contains the complete original report, including call-time inputs and
outputs, task history, effects and available source/corpus hashes. Missing usage or
source metadata is unavailable, not zero. Decompress using Python's `gzip` module or
`gzip -dc`.

Selected mechanical passes include historical anniversary recall, a verified
recurring medication reminder targeting the mother, fresh sensor recovery, and an
explicit limitation when every sensor route fails. Failures exposed stale replies,
missing recipient support, insufficient effect readback, transport failures, repeated
empty searches, lost approval provenance, and completion before approved work was
done. Those failed attempts remain alongside the fixes; none was replaced silently.

Trials span code, corpus and fixture-adapter revisions. Do not pool them into a
success rate or infer a paired performance improvement. The first network-restricted
anniversary attempt was aborted without a report; it is recorded separately in the
index. V2 explicitly states the learning route preference that V1's oracle expected
but its user request did not state. The frozen memory rollout benchmark is unchanged.

[review.md](review.md) records selected answers and effects for human review.
Independent human review remains pending. [verification.json](verification.json)
records the test command, counts and all baseline/final failure identities. The
suite retains 36 baseline failures; there are no additional failure identities.

## Limits and next decisions

This is a functional diagnostic, not rollout qualification or Issue #18 closure.
No feature default changed. Durable memory extraction is not evaluated. Real channel
delivery, broader held-out responsibilities, repeated paired trials, latency,
interrupted-task rollback and independent review remain to be established. The
successful exploratory cases took roughly 100–140 seconds, already too slow for a
polished everyday interaction. See the [design and reproduction notes](../../16-household-delegation.md).

The latest targeted checks pass mechanically for recall, recipient reminders,
read recovery and explicit failure. Skill reuse still fails end to end: the
approved definition was installed and selected, but the new task supplied a skill
name as an artifact ID and exhausted its call budget. The tool schema was clarified
subsequently; that correction has not yet demonstrated successful reuse.

For recall, the final response names the venue directly and describes the occasion
as an anniversary; the source calls it a wedding celebration. Whether that contextual
link is justified belongs in human review. The mechanical venue check alone does
not establish that semantic judgment. A progress update described this final reply
as qualified, but the retained actual response does not express that uncertainty.

The design now explicitly includes improving an existing artifact through relevant
skill expertise. That full implementation-repair loop has not been exercised here.
