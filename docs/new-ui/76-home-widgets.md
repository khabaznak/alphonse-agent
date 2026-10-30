# Home sensor and emergency-contact widgets — #76

Blockers #8 and #69 were verified completed and merged into
`codex/v3-hierarchical-capd` before implementation. This extends their existing
member-scoped polling, fixed renderers, contextual resurfacing, pins and locks.

## Behavior and prerequisites

Home sensors and Emergency contacts are available in Add widget. A selected skill
can also surface its specific source. Catalog cards aggregate only sources
already present in the current authorized poll; adding a catalog card does not
read unselected resources. Ask Alphonse to use the owning skill and pin its
surfaced source card to retain that source across conversations/restarts.

There is no bundled Home Assistant/device provider or emergency-contact directory
in this repository. Existing home automation programs are user-owned skill
resources. To supply these widgets, an installed skill must have an enabled,
registered artifact with a configured provider/account or chosen contact data,
permission to read it, and a workflow that publishes the bounded snapshot below
in its registered storage. Provider setup, credentials, refresh scheduling and
configuration updates remain the owning skill's responsibility. No provider
credentials or generated programs are added to product source. The empty catalog
explains setup and can create a conversation draft. Sending that draft uses the
ordinary authorization and skill workflow; it does not immediately configure or
execute anything. A valid empty source is distinguished from an unavailable one.

Sensors show the source's catalog name/owner, provider label, units, measurement
time and source update time in the member's timezone. “Current reading” is a
recent successful sample, not a streaming guarantee. Its bounded freshness window
is independently checked in Python on every poll and in Desktop every second,
including when pinned or locked. Stale, disconnected and error readings explicitly
say “not live” and label retained values “Last known”. Missing values say “No
reading available”; a source that has never received a reading says so. An expired
sensor snapshot may display last-known values only, always stale. A malformed,
missing or unauthorized snapshot displays unavailable without retained readings.

Emergency contacts show configured names, roles and optional phone/email. Copy
phone/email are manual clipboard actions; the member chooses their phone/mail app.
Copy failures show a fallback for selecting the displayed text. No contact action
is offered when a field is absent, and no fallback numbers are invented. Expired
contact sources hide all destinations/actions. Review/edit opens a conversation
draft with the resource/revision; subsequent source updates replace the display.
Opening, pinning, polling and copying never call anyone, dispatch help, control a
device, execute an artifact, or create runtime tools. This issue grants no new
device or communication authority.

## Declaration and source contract

An installed skill's `widgets.json` uses the existing version-1 declaration:

```json
{"version":1,"widgets":[{"key":"room-readings","title":"Home sensors","binding":{"type":"home","family":"sensors","artifact_id":"artifact.example","path":"widget-sensors.json"}}]}
```

`family` is `sensors` or `emergency_contacts`. The artifact must belong to that
skill. The relative snapshot path must resolve inside its registered storage root,
including traversal and symlink checks. Catalog owner, explicit sharing and admin
rules are checked on every read, independently of relevance, project or layout.
Missing skills, disabled artifacts, revoked sharing, unavailable providers/files
and invalid data return unavailable. Layout persistence stores identities and
placement metadata, never readings or contacts. Source replacement is read fresh
on the next accepted poll; no private payload cache survives revocation.

The source JSON has exactly `version` (integer 1), `updated_at`, `expires_at`, and
`items`. Times must include an offset. Updates cannot be in the future; expiry
must follow update by at most 24 hours. The file is at most 64 KB and contains at
most 50 uniquely identified items. Unknown fields are rejected. The skill should
publish atomically after a verified provider read or confirmed configuration
change; publication is the skill's responsibility, not performed by this reader.

A sensor item has exactly these fields (synthetic schema example):

```json
{"id":"room-temperature","name":"Room","source":"Configured provider","unit":"°C","value":23.5,"observed_at":"2026-10-29T18:00:00-06:00","stale_after_seconds":60,"state":"connected"}
```

`state` is `connected`, `stale`, `disconnected`, or `error`. The value must be a
finite number or null. The reading time cannot follow the source update time.
`connected`/`stale` require a value and reading time; disconnected/error may use
null for both when no sample exists. Retained values always require a timestamp.
`stale_after_seconds` is an integer from 1 to 3600: repeated snapshot publication
cannot make an old measurement current. The provider chooses a suitable window
within that bound. Unit is a nonempty display string (for dimensionless readings,
use a meaningful label). Names/provider labels are at most 200 characters, units
40, and IDs 128. Provider errors should be published as error/disconnected rather
than represented as a successful fresh measurement.

An emergency-contact item has exactly these fields:

```json
{"id":"chosen-contact","name":"Configured name","role":"Configured role","phone":null,"email":null}
```

Names and roles are required and at most 200 characters; IDs are at most 128.
Phone/email may be null. Phones accept an optional leading `+`, digits, spaces,
parentheses and hyphens (at most 40 characters). Email must be a bounded address
without whitespace or angle brackets. Neither field accepts an action URL or
executable content. React renders all labels as escaped text. This is a schema
example, not an installed contact or household record.

## Validation

Disposable Python fixtures exercise actual Desktop polling, authorization and
catalog sharing/revocation, owning-skill checks, source replacement, missing
files/skills, no writes on reads, stale/disconnected/error readings, missing
samples, empty contacts, expired destinations, malformed fields, unsafe action
fields, duplicate IDs, size limits, timestamps and traversal/symlink escape.
Desktop fixtures exercise renderers, escaping, units/provenance, aging independent
of locks, contact actions, missing configuration, member changes, disconnection,
revocation and metadata-only persistence. Browser results and final validation
counts are recorded on the implementation PR. Real provider qualification,
packaged native verification and household daily-use acceptance remain #78.

Initial validation passes: **129 targeted Python tests**, all **114 Desktop tests**,
and the TypeScript/Vite production build. The existing bundle-size advisory remains.
The browser fixture uses the actual drawer/renderers with synthetic sources; it
verifies configured clipboard payloads and failure feedback (clipboard stubbed),
contact replacement, conversation drafts, stale/disconnected/error states, expired
contact actions, revocation while locked, empty contacts, deduplication and
metadata-only storage. Light 1120×760 and dark 760×600 screenshots were inspected;
the card list and card bodies scroll independently with no horizontal overflow.
Screenshots remain local under `output/playwright/issue-76-home-widgets*.png`.

Integrated the newly merged care widgets (#75), preserving their explicit catalog
query and protected member-scoped adapter. Final combined validation passes:
**141 targeted Python tests**, all **118 Desktop tests**, production build, and
the repeated browser checks above. The integration tests caught a lost care
catalog condition during conflict resolution; it was restored before committing.
GitHub's native dependency relationship endpoint also confirms that #8 and #69
are the only blockers and both are closed.
