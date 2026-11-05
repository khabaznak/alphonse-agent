# Desktop shell — #65

Builds the navigation foundation from #63 on the #64 design system. The left
navigation remains available around the central workspace and collapsible
**At hand** drawer. Conversation is the default and has no project selector.
The sidebar includes Household, Projects & Tasks, Agenda, Needs your attention,
Settings, and active projects visible through the existing authorized project
API. Current member and connection/activity status remain in the toolbar when
the sidebar reduces to an icon rail.

Settings changes the left sidebar to section navigation, including Skills, with
Back to workspace returning to the prior workspace. Existing Settings, Skills,
project, member and schedule management controls are reused. Mandatory initial
setup retains its blocking dialog. This foundation does not complete the deeper
Settings organization (#66), standalone task integration (#70), Household
workflows (#72), Agenda calendar views (#71), or persistent widget pins/locks
(#8). The drawer currently provides real pending-question counts and routes to
attention/agenda, without fabricating household widget data.

Workspaces mount on first visit and remain mounted with `hidden` afterward.
This preserves local forms, selections, conversation/question drafts and scroll
positions without creating another App, polling controller or agent client ID.
The existing poll dependencies are independent of workspace navigation. Hidden
conversation updates do not auto-scroll its timeline. Needs your attention
lists the existing pending questions and links to their conversation content;
answer controls and daemon routing remain owned by the existing conversation.
Question interaction improvements remain #68.

Each column scrolls independently. At 1200 px and above the drawer is a 280 px
third column. Below that it overlays the right edge on explicit request, so
editors keep their width. At 960 px and below navigation becomes an icon rail
with named buttons/tooltips; the member and activity remain in the top toolbar.
The supported native minimum remains 760 × 600. Navigation focuses the active
workspace; hidden panels leave the keyboard/accessibility traversal. Opening
the drawer focuses Close, and Escape/Close restores its toggle's focus.

## Validation — October 29, 2026

- All 63 existing desktop tests pass. TypeScript and Vite production build pass;
  Vite retains its existing main-bundle size advisory.
- macOS ARM Tauri application packaging passes. This uses the existing
  development daemon sidecar and is a UI verification package.
- Browser checks used disposable in-memory Tauri/daemon responses, with no
  household data or mutation. Conversation text, textarea selection (5–9),
  timeline scrollTop (120), Settings scrollTop (250), new-project draft,
  new-member draft and a pending
  open-text question draft survived navigation. Settings section navigation
  and return navigation worked. Hidden panels were absent from accessibility
  snapshots, and the newly active workspace received focus. Repeated pending
  question polls did not move a manually positioned timeline (scrollTop 100).
- The fixture's `ensure_daemon` count stayed at its settled startup count (4,
  including development StrictMode and the existing startup dependencies)
  across workspace changes. No additional workspace polling or session was
  added. This is a navigation invariant, not a claim to redesign daemon startup.
- Browser resizing at 760, 960, 1120 and 1440 px: no document horizontal overflow;
  drawer stayed inside the window. Narrow sizes used an overlay; 1440 used
  three columns. Drawer open focused Close; Escape hid it and returned focus
  to the toggle at every size.
- Packaged WebKit checks used `ALPHONSE_DESKTOP_DIAGNOSTIC_MODE=static` and
  `ALPHONSE_V2_SOCKET_PATH=/tmp/alphonse-65-unavailable.sock`, without contacting
  the household daemon. Verified native draft retention across Conversation →
  Needs your attention → Conversation, contextual Settings navigation, Light
  and system Dark, visible keyboard focus, drawer open/close and Escape focus
  restoration. Resized the native window from 1120 × 760 to 760 × 600 and
  visually checked the icon rail, toolbar status, date controls, composer and
  drawer. Settings content and sidebar scroll independently at the minimum.

The browser fixture is UI evidence, not end-to-end daemon qualification. Real
household workflows and daily-use acceptance remain #78.
