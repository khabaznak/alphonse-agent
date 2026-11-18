# Owner obstacle gym: synthetic development fixtures

The frozen development corpus in `tests/fixtures/owner_obstacles/` covers a
one-time reminder, recurring event, event trigger, timer, timebox, voice,
Home Assistant mutation and composition of two synthetic domain skills. These
fixtures do not contact household services or contain user packages. Each
case's target remains task completion. A safe unavailable result is graded
separately for boundary behavior and has an unknown task pass, not success.

Run the offline candidate wiring exercise with a new disposable output path:

```
python -m alphonse.agent_v2.evaluation.owner_obstacle_gym \
  tests/fixtures/owner_obstacles/development_v1.json /tmp/owner-obstacles.json \
  --adapter candidate-scripted
```

The adapter exercises the actual simple-loop candidate with scripted decisions
and synthetic skill reads. It supports the two-skill case only; all other cases
are reported unavailable. Running without `--adapter` reports all cases as
unavailable. The corpus also has simulated-success cases for schedules,
triggers, timers, timeboxes, voice and home control, plus separate unavailable
and denied cases. Positive requests specify the dates, times, timezone or spoken
text needed for their fixture effects; ambiguous variants do not invent details.
Grading expectations are outside scenario input, but the
in-process world is mutable and is not independent of adversarial adapter code.
A later adapter can exercise the same input and world through `ObstacleAdapter.run`.
The adapter's offline declaration is not a network sandbox. Fixture passes do not establish
autonomous behavior, real provider authorization, external effects, delivery,
or production readiness. No live model, Jev or household integration is called.
The scripted two-skill case checks wiring and reads only; its task pass remains
unknown because its answer has no independent semantic assessment.

The separate `clock_development_v1.json` corpus can exercise the current
candidate's read-only tool path with a frozen UTC clock observation:

```
python -m alphonse.agent_v2.evaluation.owner_obstacle_gym \
  tests/fixtures/owner_obstacles/clock_development_v1.json /tmp/owner-clock.json \
  --adapter clock-scripted
```

Its scripted chooser reads the synthetic clock and assembles a JSON timestamp
from that observation. A separate fixture oracle checks the exact JSON value,
the clock-read event and absence of effects. Unavailable and denied clock cases
block rather than fabricate a value. This is a candidate wiring exercise, not a
real owner clock, a model assessment, an elapsed-time test or evidence of a
real reminder or external delivery.

`relative_reminder_development_v1.json` records the owner's Spanish request
to be reminded in one minute to go downstairs for water. It fixes a synthetic
UTC observation and checks for one matching reminder due exactly one minute
later, with a clock read and no duplicate effect. The expected reminder text is
an exact fixture representation, so valid paraphrases need a future semantic
grader. Response meaning remains unassessed; even a scripted fixture effect is
not an autonomous task pass. No adapter currently exercises this case against
the real owner processor. Without one it is reported unavailable, not passed;
the existing English-only reminder gate is still a known product limitation.

Before using this for broader qualification, add independent semantic and
evidence-quality grading, independently designed held-out cases, multilingual
examples, realistic clocks and cancellation/restart faults, execution provenance,
and adapters for the real owner worker with strictly fake transport. Preserve
all failed, unavailable and interrupted outcomes rather than retrying them away.
The current runner writes its report only after the full run: an actual process
termination can lose every in-memory row, including earlier attempts. Its
`FixtureInterrupted` handling represents an explicitly raised synthetic
interruption only. Contradictory observed effects fail; an expected partial
effect leaves completion unknown. Durable per-case journaling and incomplete-run markers remain
future work; do not treat this report as an interruption-safe journal.
