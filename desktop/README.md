# Alphonse Desktop

The macOS-first native client for the Alphonse v2 daemon. The interface is a
Tauri 2 shell with a React/TypeScript frontend; all agent state and processing
remain in the Python daemon.

## Development

```bash
npm install
npm run tauri dev
```

Desktop first connects to the configured local v2 socket. If no daemon is
available, development builds launch the repository's `.venv/bin/python -m
alphonse.agent_v2.daemon` from the project root, falling back to `python3` when
no project virtual environment exists, then wait for it to become healthy.
Install project dependencies first (`.venv/bin/python -m pip install -r
requirements.txt`), including the event-worker JSON Schema dependency.

## Release sidecar

Tauri expects a target-specific executable at
`src-tauri/binaries/alphonse-daemon-<target-triple>`. The committed macOS ARM
wrapper starts the daemon from the project virtual environment for development. Replace it with the
standalone executable before packaging:

```bash
python -m pip install pyinstaller
./scripts/build-daemon-sidecar.sh aarch64-apple-darwin
npm run tauri build
```

The packaged executable is launched only when the app cannot connect to an
already-running daemon. Closing Desktop leaves the daemon running.

## WebKit memory diagnostics

Set `ALPHONSE_DESKTOP_DIAGNOSTIC_MODE` before launching the desktop executable
to isolate recurring work in the WebKit renderer. Supported values are
`static`, `ping-only`, `poll-no-commit`, `render-only`, `history-static`,
`history-render`, `history-render-plain`, `history-render-memo`,
`history-render-timeline-memo`, and `normal`.
Unknown or missing values use `normal`.

`poll-no-commit` performs real Desktop polls but deliberately does not render or
acknowledge their contents. Use it only with an idle daemon that has no queued
messages, active work, questions, or scheduled-task notifications.

The four `history-*` modes load up to 100 real Home conversation messages once.
They respectively leave the history static, rerender its Markdown, rerender it
as plain text, or rerender the parent while memoizing the Markdown bubbles.
Set `ALPHONSE_DESKTOP_DIAGNOSTIC_PROJECT_ID` to load a specific project's
timeline instead of Home.

`history-render-timeline-memo` exercises the production fix: the parent keeps
rerendering while the complete unchanged timeline and its message bubbles retain
their existing React element identities.

After launching a fresh app process, identify its new WebKit WebContent PID and
record the standard one-minute warm-up plus five-minute sample:

```bash
./scripts/sample-webkit-memory.sh WEBKIT_PID MODE
```

The script accepts optional warm-up, duration, interval, and output-file
arguments after the mode. It writes CSV measurements from macOS `vmmap`; the
terminal may request permission to inspect the WebKit process.

## Everyday use and recovery

Start in **Conversation** for an everyday question or request; a project is
optional. **Household** shows shared efforts you are allowed to see.
**Projects & Tasks** separates active, blocked/needs-input and historical work;
open the item to inspect its current state and related conversation.

**Needs your attention** keeps pending questions reachable across navigation.
Answer the specific question card once. Text and choice answers resume its own
task; a question answer does not grant new tool permissions. If submission fails,
read the displayed error and check the question's current state before retrying.

Open **At hand** for contextual or saved cards. A pin keeps a card after context
changes; a lock also prevents moving or removing it until you unlock it. Compact
header icons have labels/tooltips and support keyboard navigation. The header's
plus opens the catalog at the bottom of the list. Escape closes the drawer and
returns focus to its toggle. Removing a card changes the drawer only: it does not
cancel a task, remove a schedule or delete its program/data. Long cards and the
card list scroll separately. Source/details disclosures retain provenance and
secondary actions without crowding out the content.

The Agenda card previews Month, Week or Today; use Detailed Agenda for the full
workspace. Select a schedule occurrence to inspect/edit it, pause/resume it or
review execution history. A projected occurrence is not evidence that it ran.
Other open Agenda views refresh within 30 seconds. Source status stays separate
from event time and recurrence. Unconnected calendar feeds are identified under
Calendar sources.

Widget requests such as list edits or timer pauses may prepare a **conversation
draft**. Review and send it; a prepared request is not a confirmed write. Only
source-confirmed effects update the record. An elapsed timer does not prove a
completed task. Check Source & details or the underlying item when a value is
unexpected. Unavailable health/exercise/sensor data requires its owning skill and
authorized source; missing data is not a missed treatment or a live measurement.

On disconnection, wait for Connected locally before assuming fresh data. Check
pending questions and the item's current state after reconnect; avoid blindly
resending a mutation that may already have run. Closing Desktop leaves the daemon
running. On an app or daemon restart, verify the same member/workspace, history,
questions, schedules and pins/locks. Work logs are a bounded replay, not an
unlimited persistent audit; conversation and source records remain authoritative.
Unsent settings edits and drafts should be copied before quitting if needed.

For a failed/blocked task, expand its work log, read the failed action and required
input, and request a retry only after resolving the blocker. Use the visible stop
control to request cancellation; inspect the terminal state and verified partial
effects before retrying. Do not delete databases, the Workspace, skill resources
or artifact files to repair a display, reconnect or migration problem.

Settings contains appearance, model/System One validation, integrations, members,
skills and artifact management, subject to existing access rules. Layout and skin
preferences are per member on this Mac. Motion follows macOS Reduce Motion.

The current validation evidence, known baseline failures, observation record and
pending owner acceptance are in
[the daily-use acceptance record](../docs/new-ui/78-daily-use-acceptance.md).
