# At hand widget home — #8

Extends the responsive drawer shipped in #65; it does not replace the shell.
The existing Pending questions and Household agenda cards are the initial
catalog. Agenda remains a shortcut to the authorized workspace, not a calendar
implementation. No fabricated household widget records are introduced.

## Layout and member controls

The member can open/close the drawer, add missing catalog cards, move unlocked
cards up/down, pin/unpin, lock/unlock, and remove unlocked cards. Locked cards
cannot be moved, removed, or crossed by the move buttons until unlocked.
Unlocking returns a card to pinned. Native buttons, labels, a native catalog
disclosure, and visible focus support keyboard operation. Opening focuses Close;
Escape/Close restores the toolbar toggle. The card list scrolls independently
of the main workspace; each card body can scroll without moving its controls.
The existing 280 px third column/overlay and 760 × 600 native minimum remain.

Drawer metadata is versioned in member-specific localStorage keys:
`alphonse.desktop.atHand.v1.<encoded-member-id>`. It persists visibility, stable
identities, kind, mode, expiry, ordering, and dismissed identities across workspace
navigation, client reload/restart and daemon reconnect/restart. It stores no
question bodies, agenda records, program paths or other widget payloads. A missing
member disables the toggle and prevents persistence. Identity-tagged state avoids
writing the previous member's layout into the next member's key. Failed writes
leave the current session usable and show a persistence notice; malformed storage
falls back safely, and duplicate saved identities are discarded.

## Lifecycle and bridge contract

- Contextual cards have an explicit absolute expiry. Expiry leaves a labeled
  placeholder until the member dismisses or pins it, avoiding automatic layout
  movement. Unpinning a card makes it contextual for 30 minutes. Pins and locks
  survive contextual expiry; locking implies keeping the card.
- Removing a card records a layout dismissal only. It never calls a task,
  question, schedule or artifact cancellation/deletion endpoint. Closing the
  drawer does not dismiss cards. Catalog Add restores a removed catalog card.
- `reconcileContext` is the deterministic automatic-update entry point for #69.
  It only appends eligible, unexpired identities. It preserves every existing
  identity, order, mode and expiry; it cannot downgrade locks, move/remove cards,
  reopen the drawer, undo dismissal, or duplicate cards on repeated snapshots.
  A source that needs a new context lifetime must issue a new stable identity.
- `changeDrawer` is exclusively for explicit member controls. Do not route agent
  or bridge updates through it. #69 owns context selection and the authorized
  data bridge; neither is connected in #8. Future widget families extend the
  catalog and their own renderers in their implementation issues.
- Saved IDs and relevance are never permission. Unknown kinds show an unavailable
  placeholder without source details. No member shows loading; a lost connection
  shows unavailable, without stale question counts. Question counts are exposed
  only after a fresh authorized poll for the current member. Agenda delegates
  data access to the existing authorized workspace. #69 must supply only fresh,
  authorized payloads and suppress revoked data even for pinned/locked cards:
  a lock preserves placement, never access.

## Validation — October 29, 2026

- 89 desktop tests pass after integrating the merged #77 mascot changes,
  including seven new model/render tests covering member
  storage isolation, round trips, payload exclusion, lock enforcement, ordering,
  expiry, dismissal, reconnect deduplication, unavailable/loading states,
  stale-member count suppression and failed persistence.
- TypeScript and Vite production build pass. The existing bundle-size advisory
  remains. No Python/daemon code changed.
- Disposable browser Tauri/IPC fixtures verified reorder → lock → workspace
  navigation → reload preserves order, visibility and locks; locked removal is
  disabled; remove/catalog-add produces exactly two cards; a saved closed drawer
  stays closed after reload. Unpin survives reload and keyboard Enter pins again.
- At widths 760, 960, 1120 and 1440 (600 px high), no horizontal overflow; the
  drawer stays in bounds, card-list scrolling is independent, and Escape restores
  toolbar focus at every size. Light and system Dark were visually checked,
  including the native minimum. No real household data or mutations were used.
- Browser reload/model fixtures validate restart-independent layout behavior;
  this does not claim packaged native or real household workflow qualification.
  Broader New UI daily-use acceptance remains #78.

The #77 integration conflict in the poll handler retains both the fresh-member
question marker and the mascot context update. The combined build/tests pass;
browser reload at the minimum size retains the locked layout and base mascot.
