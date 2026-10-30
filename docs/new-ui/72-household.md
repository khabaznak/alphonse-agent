# Household — #72

Household now presents coordination cards rather than member administration:
Today at home, shared efforts, questions needing input, routines and everyday
work. It follows Design Study 01's warm surfaces, primary today card and quieter
supporting cards. Administrators can still open member management in Settings.

## Existing authorization and API gap

Existing project-list Desktop methods normalize to the administrator. They are
unsuitable for this projection. The read-only `household_overview` IPC method
requires an explicit, active member and calls the existing membership-scoped
project store without an administrative bypass. It returns active shared project
IDs, names and states, excluding system home projects. It omits descriptions,
paths, raw context, summaries, membership lists and cross-owner task counts.
The current member and audience are explicit in the UI.

Shared is the default. Selecting **Only you** adds that member's today plans,
repeating schedules, active task goals/states and pending questions. The daemon
does not query these sources in Shared mode. The IPC personal flag accepts only
JSON `true`, not truthy strings. No owner-selection parameter is supported.
Administrative rights do not expand this household projection.

Schedules and checkpoints have owner-based permissions, not household sharing.
A shared project's membership does not authorize another person's schedule or
checkpoint. Therefore the shared today card explains the boundary and offers
personal disclosure; it does not manufacture a shared calendar from private
records. This work adds no sharing model or generated household artifact. Today
uses the member's configured timezone, including for schedules with another zone.
Schedules are a bounded first 100 active records in the existing next-run order;
tasks use the first 50 active checkpoints. Each section shows at most 12 cards,
with explicit scope text and routes to the full workspaces. No hidden/inaccessible
records contribute displayed counts or overflow hints. Treatment, shopping,
sensor and finance artifact contents are never read by this projection.

Personal task/schedule project references must still be visible; missing,
archived or revoked projects are omitted. Delegated questions continue to use
respondent authorization independently of project membership. The existing
respondent-owned child task is preserved. No ownership, delegation, scheduling,
execution or completion semantics change.

## Source navigation and freshness

Cards use stable source IDs. Shared efforts open the project editor; scheduled
plans/routines open the existing Agenda record and controls/history. Task cards
open the existing creation-day conversation, and questions open the existing
Needs your attention answer workspace. Household has no duplicate editors,
answers, task completion controls or new execution path.

Every card activation re-reads the projection and confirms the record is still
present. Revoked or terminal records disappear instead of navigating. Each source
is isolated; unavailable sources clear their cards and show a generic error.
Raw exception text is withheld because it can include private paths/content.
Missing sources never claim an empty, successful result.

The component keeps no persistent cache. It clears its in-memory snapshot on
refresh, disconnect, focus loss, visibility loss, navigation and member change;
late responses are discarded using a request generation and audience key. Private
disclosure resets on navigation. A single 30-second refresh timer runs only while
Household is active and connected; hidden/unfocused windows do not query. Access
is a fresh snapshot, not a revocation push channel: a change while focused is
reflected on refresh (within 30 seconds) or before opening a card. This does not
claim instantaneous revocation of already rendered pixels. Native destination
APIs retain their existing authorization; #51 owns broader legacy Desktop
project-administration reconciliation.

## Validation — October 29, 2026

- Eight disposable Python/SQLite tests cover admin/member/unrelated scopes,
  explicit shared membership, personal source opt-in, private health/finance
  exclusion, allowlisted payloads, missing/archived/revoked references, inactive
  actors, delegated respondents/child tasks, cancellation, unavailable sources,
  stable IDs, timezone-based today, empty users and bounded cards.
- Desktop tests cover member/audience snapshot rejection, Shared default,
  administrator-only management, member context and disconnected guidance.
- Twenty-two disposable Playwright checks exercise the integrated shell with
  synthetic data: hierarchy, personal opt-in, project/Agenda/conversation/question
  links, reset on navigation, access recheck on activation, failed refresh and
  private error suppression, empty state, loading, out-of-order audience responses,
  blur/focus clearing and recovery, 760 × 600 width and independent scrolling.
  Light/dark and minimum-window screenshots were inspected. No household daemon
  or real household records are involved.
- Source failures, membership and ownership are tested against real stores;
  browser fixtures model those responses. Real household daily-use acceptance
  remains #78. No browser/mobile product client or CLI removal is introduced.

Final checks: 106 targeted Python tests, all 68 Desktop tests and the
TypeScript/Vite production build pass. Vite retains its existing bundle-size
advisory. The Playwright checks were rerun after the final lifecycle refinement.
