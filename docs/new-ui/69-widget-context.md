# Conversational widget bridge — #69

The only blocker (#8) is complete, merged in PR #93. This extends its existing
At hand drawer, member layout storage and explicit pin/lock/remove controls.

## Resource and transport contract

An installed skill may contain `widgets.json`, a declarative version 1 resource:

```json
{
  "version": 1,
  "widgets": [
    {"key": "calendar", "title": "Calendar", "binding": {"type": "agenda"}},
    {"key": "past-session", "title": "Exercise progress", "binding": {
      "type": "memory", "source_id": "conversation:<authorized-record-id>"
    }, "suggestion": "show"}
  ]
}
```

These examples are a schema illustration, not installed household records.
A skill author can use the existing authorized skill-file editor/installation
flow. No generated program or real household data belongs in this repository.

- Identity is `skill-widget:` plus a digest of member, owning skill ID and
  declaration key. Keep keys stable when updating a binding/title. All current
  bindings are member-scoped. Context ID identifies the active task; it does
  not confer permission. Skill IDs remain the existing installed skill IDs.
- At #69 completion, only the fixed Agenda navigation and historical-memory
  text renderers were supported. #73 adds the authorized fixed list snapshot
  binding described in [daily coordination widgets](73-coordination-widgets.md). Agenda opens the existing authorized workspace; it does not claim
  to provide a calendar data implementation (#71/#73). Historical text uses the
  existing native memory-source reader, with the polling actor as requester,
  no neighbors and at most 4000 characters. It is labeled historical, never a
  current measurement or verified external effect. Future widget families add
  reviewed native bindings; arbitrary URLs, paths, scripts, HTML, custom actions
  and extra binding fields are rejected. Text renders as React text.
- `desktop_poll.widget_context` is an authoritative versioned snapshot, tagged
  with member. It includes owning skill, task context, binding type, current
  availability, plain text, relevance, suggestion and absolute expiry. Client capabilities carry at most 128 retained widget IDs; these are lookup
  hints, never access grants. Only selected skills and retained IDs have their
  sources read, avoiding reads of unrelated skill data. Every
  poll rereads installed declarations and current source authorization. Missing,
  revoked, failed or unavailable sources emit no old text. Missing/uninstalled
  skills disappear from the snapshot. Pins/locks preserve layout only.
- Skill declarations are bounded to 32 KB, 16 rows per skill and 64 installed
  skills. A declaration must resolve inside its skill package; symlink escape,
  malformed schemas, unknown renderers and duplicate keys fail closed.

## Selection, lifecycle and ordering

The daemon reuses `v3_skill_selection.selected_skill_ids` from the active task's
existing Jev decision, only when that task belongs to the polling member and is
not cancelled/terminal. It does not classify the question with keywords or an
extra LLM call. Waiting-user tasks may retain relevant context. System Two
continues composing language/tool calls. There is no new tool or script execution
path, artifact registry entry, permission grant or mutation capability.

Supported declarations suggest `show` (default), `update`, `pin`, `remove` or
`expire`. Show and pin proposals can append a ready, relevant contextual card;
pinning itself requires the explicit member control. Update replaces the fresh
payload only. Pin/remove/expire suggestions are explained on the existing card
and use its explicit controls. They cannot remove or change a locked position,
change saved modes/order, reopen a deliberately closed drawer or undo dismissal.
Contextual cards receive a 30-minute suggested lifetime. Existing placement
expiry is never refreshed by polling; pinned/locked placements do not expire.
Expiry shows the existing placeholder. Removal affects layout only.

The App's existing monotonically increasing poll-generation guard rejects late
responses before committing any widgets. Member-tagged snapshots additionally
prevent cross-member rendering or layout reconciliation. Every accepted snapshot
replaces payload state, including an empty/absent snapshot; there is no merge of
old data. A disconnected client hides payloads. Reconnects and daemon restart
replay stable identities, so reconciliation never duplicates cards. Browser
storage persists layout metadata only, never skill text/bindings/source IDs.

## Validation

Disposable Python fixtures exercise real IPC polling, installed skill resources,
conversation-source authorization, dinner/trip planning selection, past-session
exercise context, repeated reconnect snapshots, other-actor denial, missing
sources, changed/revoked binding, uninstallation, unsafe renderer/binding
rejection and symlink escape. No real household records/programs are used.
Desktop fixtures exercise both contextual examples, preservation of an unrelated
locked card, stable replay, dismissal, revoked/missing/disconnected/member-scoped
data suppression and lifecycle suggestions. They verify arbitrary text is escaped.

Validation totals and browser smoke evidence are recorded on the implementation
PR. Real household acceptance and individual widget-family qualification remain
in #78 and their respective implementation issues.

Verified October 29, 2026 (member timezone): 80 targeted Python tests passed
(widget bridge, IPC, skill appearances and memory recall), all 92 Desktop tests passed, and the
TypeScript/Vite production build passed (existing bundle-size advisory only).
Disposable browser IPC fixtures exercised the actual App poll path with a delayed
ready response released after a newer unavailable response: old text did not
return. Explicit locks, disabled locked removal, unrelated locked placement,
calendar dismissal, reload, reconnect deduplication, missing-skill placeholders,
and metadata-only storage passed. Cards were visually checked at 1120 × 760;
the drawer was checked at the 760 × 600 native minimum. Screenshots remain local
in `output/playwright/issue-69-context-cards.png` and
`output/playwright/issue-69-widget-context.png`; no fixture data is committed.
The browser fixture mocks Tauri/IPC; it does not qualify real household data or
a packaged native release. The fixture's initial missing Tauri event-cleanup
mock was corrected and the complete browser scenarios rerun successfully.

Integrated PR #94 (#9 task activity) from the current V3 branch before delivery.
The combined build passes, all 98 Desktop tests pass, and 85 targeted Python tests
pass. Repeated the complete delayed-response/revocation, lock/dismissal/reload,
missing-skill and visual browser scenarios against that combined implementation;
all passed. The existing poll-generation guard and widget reconciliation both
remain connected after integration.
