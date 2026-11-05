# Treatments and care widgets — #75

The declared blockers #8 and #69 were verified closed and merged before work.
This extends their existing At hand drawer, authorized poll snapshots, member
layout, pins/locks and conversation-draft routing. Doctor implementation remains
in #6, which is still open. **No production Doctor adapter is installed by this
change. Live care records correctly remain unavailable until #6 supplies it.**

## Resource and protected adapter contract

An installed Doctor skill may declare this bounded resource in `widgets.json`:

```json
{"version":1,"widgets":[{"key":"care","title":"Treatments & care","binding":{"type":"care"}}]}
```

Only this fixed binding is accepted. No subject override, path, database,
artifact ID, script, arbitrary UI or executable action is allowed. Skills and
artifacts are never materialized as new runtime tools. The catalog's Treatments
& care card requests this binding even outside an active task. Contextual cards
reuse the existing Jev skill selection; no extra classification call is added.

`V2RuntimeHost.care_records` is an optional **trusted native** `CareRecords`
adapter, default `None`. #6 must install its implementation through the harness,
not by evaluating a skill resource. `read_for_member(actor_user_id, skill_id)`
must authorize the current actor and installed skill on **every read**, query
only actor-owned health records, and return no other subjects. A skill selection,
retained widget ID, layout pin or Doctor appearance grants no access. This first
projection deliberately supports the current member only; broader caregiver or
shared-subject access requires an explicit authorized extension under #6.

The adapter returns a versionless internal projection with exactly:

- `member`: authenticated actor, checked again before projection;
- `source`: a bounded plain-text source description;
- `recordedAt`, `expiresAt`: timezone-aware ISO timestamps, with a current lifetime
  no greater than one hour;
- `items`: at most 50 records, each with stable `id`, source-defined `definition`,
  nullable `dueAt`, `adherence` (`recorded`, `missed`, `unknown`), nullable
  `recordedAt`, and `writeState` (`none`, `verified`, `failed`, `unconfirmed`).

The adapter must read actual definitions and explicit adherence records. It must
never infer missed care from absent data or a passed due date. A known adherence
status requires its record timestamp. `writeState=verified` requires persisted
effect evidence from the existing write workflow, not an attempted call or a
model assertion. This module validates structure and provenance requirements;
it cannot replace the adapter's protected database authorization or verification.

The daemon reads on every requested poll without retaining health data in a
cache. Missing adapters/configuration, denied/revoked access, query errors,
wrong-member payloads, invalid data and expired sources emit an unavailable row
with no previous payload. Removed skills disappear. Existing poll-generation
and member guards reject delayed responses. Local storage holds layout metadata
only; disconnection suppresses record contents. The renderer checks source expiry
on the drawer's regular refresh as well as each new poll.

## Display and requests

Cards show member, source and source timestamp; each record shows its definition,
record ID, recorded due time or unknown, and explicit adherence status/date.
Unknown and empty records never become missed adherence. The widget supplies no
medication, dosage or diagnosis. Source text is escaped React text. Source-confirmed
verified writes are labeled separately from failed and unconfirmed writes.

Open record / ask, Request logging and Request correction prepare conversation
drafts using the existing App routing. They do not execute a mutation or claim
completion. The draft carries the member, source revision and record ID, asks for
missing details, and requires fresh authorization, source retrieval and persisted
verification. Existing question/task/skill flows perform the operation and report
failures. The next protected poll reflects only the source's confirmed records.
There is no new health database, generated program, artifact catalog dependency
or product implementation of the Doctor skill.

## Validation and limits

Disposable synthetic adapters exercise the actual daemon widget polling path:
separate member records, missing capability/configuration, source revocation,
uninstallation, failed writes without optimistic adherence, verified receipts,
malformed/expired sources, explicit missed records without dates, and rejected
path/member/script/artifact overrides. UI tests cover provenance, unknown versus
missed, write receipt states, escaped definitions, source expiry, disconnection,
member changes, missing capability and metadata-only persistence.

90 targeted Python tests pass (care, context bridge, coordination, IPC, questions
and settings). All 114 Desktop tests and the TypeScript/Vite production build
pass; Vite reports its existing bundle-size advisory. Browser checks use a
synthetic React fixture, not real health records or a packaged native release.
Real Doctor database retrieval and write verification remain #6; real household
acceptance remains #78.

Browser smoke verification: Request logging populated a draft containing the
member/source/record ID and verification requirements. Revocation removed the
care payload and showed the truthful unavailable state; locked removal/movement
remained disabled. Visually reviewed the ready card and the revoked card at
760 × 600. Local screenshots are in `output/playwright/care-ready.png` and
`output/playwright/care-unavailable-760.png`; temporary fixtures are not committed.
Initial fixture-only font lookup errors were resolved by serving the bundled fonts.
