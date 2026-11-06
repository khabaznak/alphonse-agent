# Household workflow and daily-use acceptance — #78

Status: **not accepted**. On October 30, 2026 (America/Mexico_City), the owner
reported that At hand remained cramped and did not respect Design Study 01,
then supplied side-by-side screenshots. Passing component tests did not catch
that visual mismatch. The layout correction below still needs owner review;
this record does not authorize closing #78 or unblocking CLI removal (#80).

## Revision and configuration

All 17 native blocked-by relationships were verified closed before this run.
The initial integrated revision was `13ebaea987d42b9748eb5e5131d4009b717f0f40`.
The tested frontend correction is commit `3271125` (its code was built/tested
immediately before committing). This document and README guidance are subsequent
documentation changes. Python source is unchanged from the integrated revision.

Environment: macOS 26.6.2 / ARM64, Python 3.12.11, Node 26.0.0, npm 11.14.1,
Rust 1.92.0; installed lockfiles/dependencies, member timezone America/Mexico_City.
Tauri 2 packages its WKWebView frontend and standalone PyInstaller daemon.

## Acceptance finding and correction

The old 280 px drawer differed from the accepted 324 px reference. A persistent
catalog/help block and full-width action rows dominated the cards. Agenda reused
workspace controls inside a 220 px scrolling body; its Month/Week views were
vertical stacks of full day cards. These were product presentation defects.

The correction restores the 324 px drawer, quiet header/count, compact labeled
icon controls, content spacing, and bottom Add widget catalog from the mockup.
Help stays in the expandable catalog. The card list scrolls independently; long
card contents retain a larger bounded scrolling area. Locks, pins, dismissal,
expiry, identities, ownership and persisted layout semantics are unchanged.

Agenda now has dedicated previews: segmented Month/Week/Today controls, a compact
seven-column month picker with occurrence markers, week rows, and time/event
rows. Date navigation, schedule details, source availability, actual execution
failure/status and routes into the full workspaces remain available. The full
Agenda workspace is unchanged. Progress cards lead with a numeric ratio and
percentage; list cards lead with rows and confirmed source status. Source IDs,
provenance, workflow requests and secondary actions remain inspectable in details.
A check mark still reflects the source's completed state; opening a row never
completes work. No mockup sample records become production defaults.

## Automated validation

From the repository root unless a directory is shown:

```sh
.venv/bin/python -m pytest tests -q --tb=short
.venv/bin/python -m pytest tests/test_agent_v2_ipc.py tests/test_desktop_questions.py tests/test_agent_v2_ask_question_ui.py tests/test_desktop_settings_workspace.py tests/test_agent_v2_artifacts.py tests/test_agent_v3_artifact_resources.py tests/test_widget_context.py tests/test_coordination_widgets.py tests/test_activity_widgets.py tests/test_care_widgets.py tests/test_home_widgets.py tests/test_skill_appearances.py tests/test_agent_v2_native_scheduled_task_tool.py -q --tb=short
npm --prefix desktop test
npm --prefix desktop run build
npm --prefix desktop run tauri:build
```

- Full Python suite on the initial integration: **1,307 passed, 3 failed, 0 skipped**.
- Focused Python suite above: **225 passed**. Covers real store/IPC/question
  answer and continuation, schedule ownership/lifecycle, widget revocation,
  containment, settings/skill lifecycle and preservation of artifact files.
- Corrected frontend: **126 passed in 29 files**. Includes new preview coverage
  for selected-day Month content, all seven Week rows, failed actual executions
  versus unexecuted schedules, and truthful empty Today.
- TypeScript/Vite production build and macOS ARM native package: **passed**.
  Vite retains its bundle-size advisory (597.13 kB JS before gzip). No warning
  threshold was raised. PyInstaller also emits dependency/platform advisories;
  packaging success does not establish every optional provider's availability.

### Python failures and baseline comparison

The exact failing IDs are:

- `tests/test_project_provenance_waiting.py::test_new_conversation_creates_protected_home_project`
- `tests/test_project_provenance_waiting.py::test_independent_message_receives_exact_turns_ahead_notice`
- `tests/test_memory_knowledge.py::test_runtime_wires_extraction_to_current_router_and_jev_without_project`

All three reproduce on an untouched export of
`05a054ae0a4717447052b9255f6721e3086b2e90`, using this environment and the
same containing provenance file plus the exact memory test: **3 failed, 1 passed**.
The first two expect automatic home-project creation and remain owned by
[#52](https://github.com/khabaznak/alphonse-agent/issues/52). The last fixture
omits a temporary users root and tries to write the real profile; the sandbox
rejects it. [#102](https://github.com/khabaznak/alphonse-agent/issues/102) owns
that isolation defect. Do not escalate that test to write household files.

[#48](https://github.com/khabaznak/alphonse-agent/issues/48)'s original result
was 36 failures on a much older revision, not the current baseline. Both
[#54](https://github.com/khabaznak/alphonse-agent/issues/54) tool identity/status
regressions pass in this run, including private-reasoning checks. No failures
were hidden with skips, xfails, weaker assertions or live provider calls.

## Native review and its limits

The initial packaged app used a disposable SQLite workspace, its own Unix socket,
140 synthetic conversation events (100 recent messages loaded), a shared fixture
project, a schedule, and text/choice questions. Both answers were submitted in
native WebKit and persisted for the matching continuation task. Continuation used
an explicitly synthetic processor that wrote a receipt; this is transport/store
coverage, **not** a real household processor or live-model qualification.

Native checks exercised Household/Agenda/Settings navigation, schedule details
and history routes, pause/resume, lock-disabled move/remove controls, both skins,
visible keyboard focus and the 760 × 600 minimum. Disconnect displayed an explicit
disconnected state. Rebuilding the disposable daemon from its SQLite stores
preserved answered questions, history, project and schedule; the in-session draft
and locked card survived reconnect. Agenda previews refresh on their existing
30-second cadence, so a second open view can briefly show the prior schedule state.

A validation restart exposed a fixture-isolation mistake: the client was launched
while its disposable socket was down. Auto-start then launched a bundled daemon
on that socket using the normal profile database. That extra daemon was stopped;
no test mutations were submitted after the unexpected real identity appeared.
That interval is excluded from fixture evidence. Subsequent native review uses a
separately identified `Alphonse Review` bundle, verifies `current_user` before
launch, and explicitly sets both `ALPHONSE_V2_SOCKET_PATH` and
`ALPHONSE_V2_DB_PATH` to disposable storage. A socket override alone is not data
isolation. Do not claim household data was untouched by a daemon startup audit
without inspecting its effects; no reset or deletion was used as a recovery step.

The pre-correction native run included the documented 60-second warm-up and
300-second WebKit sample. Raw synthetic-run metrics are in
[evidence/78/pre-layout-webkit-memory.csv](evidence/78/pre-layout-webkit-memory.csv).
Physical footprint ranged 89.8–204.9 MB, finishing at 108.4 MB; the recorded peak
reached 388.0 MB during navigation. This was a mixed-workload sample of the old
layout, not proof of a leak-free final layout or an endurance pass. The final
revision’s bounded native review, reduced-motion check and fresh sample are
recorded below; sustained regular-use evidence remains pending.

## Household observation record

| Date (America/Mexico_City) | Observer | Observation | Status |
| --- | --- | --- | --- |
| 2026-10-30 | Owner | At hand is cramped; widgets do not honor the accepted mockup. Supplied screenshots contrast the intended Agenda/Movement/To-dos layout with oversized controls and a clipped Agenda. | Blocking finding; correction awaiting review |
| 2026-10-29–30 | Agent | Automated and synthetic native checks described above. | Technical evidence only; no daily-use acceptance |

No successful household workflow, trial duration or owner approval is invented.
Append observations as normal use happens. Each entry should state revision,
member role (without private identifiers), date/timezone, workflow, expected and
observed behavior, persisted effect/source receipt, recovery and linked follow-up.
Keep health records, credentials and private household content out of GitHub.

Before completion, collect actual processor → daemon → Desktop evidence for:
ordinary conversation; text/choice answer and resumed work; shared dinner/trip
planning; project and standalone delegation; schedule creation/edit/history;
exercise context; and authorized care records, including honest missing-provider
behavior where applicable. Also cover failure/blocker/cancellation, context changes,
mascot skill states, revoked access, reconnect and app/daemon restart. Repeat
native skin/focus/contrast/reduced-motion and long-history checks on the accepted
revision. Use the usage/recovery guidance in [desktop/README.md](../../desktop/README.md).

## Remaining work and owner acceptance

- **Within #78:** review the corrected layout with the owner, finish real household
  workflows, then obtain dated regular-use acceptance. The native review below
  qualifies the tested layout revision; repeat it if source changes affect the UI.
  The owner has reported a defect; that is not an approval to merge/close acceptance.
- **Baseline maintenance:** #48, #52 and #102 above. #54 is resolved and passing.
- **Capability prerequisites:** [#4](https://github.com/khabaznak/alphonse-agent/issues/4)
  owns Personal Trainer and [#6](https://github.com/khabaznak/alphonse-agent/issues/6)
  owns Doctor. The default runtime has no live care adapter. Show unavailable when
  a required source is absent; synthetic records do not qualify those capabilities.
- **CLI:** [#80](https://github.com/khabaznak/alphonse-agent/issues/80) remains behind
  #78. No interactive CLI or operational recovery command is removed here.

Acceptance is **pending**. When the owner is satisfied after reliable regular use,
record their actual dated statement, accepted revision, observation evidence and
any explicitly accepted nonblocking follow-ups. Only then complete the acceptance
checkbox, finish the implementation PR, merge and close #78. No fixed trial period
is imposed and no approval statement is pre-filled on the owner's behalf.

### Correction review on `3271125`

Rebuilt the native package after the final CSS specificity fix (removing the old
rule that made every drawer button a filled primary action). The uniquely
identified review bundle connected to the verified synthetic member. Native
screenshots were inspected in dark and light modes: Agenda uses segmented tabs,
a complete compact month grid and readable time/event rows; Movement shows the
75% fixture ratio and dot grid without a footer full of controls. Source details
remain expandable. The same three-card fixture was reviewed at 1120 × 760 and
760 × 600. Locking Agenda disabled both movement and removal. The cards remain
in their own scrolling column, with the collapsed navigation and composer usable
at the minimum. These checks support the correction, not owner acceptance.

## Sidebar, composer and diagnostic widgets — follow-up on `1684a37`

The owner also reported that the sidebar and main composer did not follow the
mockup, and asked for Workload and PDCA to become widgets. This follow-up restores
household navigation spacing, an actual workspace/project filter, project status
dots and the bottom Settings/status/profile area. The filter only searches the
already authorized navigation entries; it does not pretend to search all records.
The member remains explicit, and Settings still changes the sidebar's internal
navigation. Long project lists scroll without pushing the footer out of view.

The composer now has one larger rounded surface, a full-width text area, a quiet
Everyday assistant label and an accessible arrow Send control. Its plus opens the
existing command catalog. Choosing a command prepares a draft only; commands are
disabled while another draft exists, so this control cannot silently discard it.
Escape closes the command menu and restores text focus. Existing Enter/Shift+Enter
behavior, task submission, errors, settings and conversation continuity remain.

Workload and PDCA activity are optional catalog entries, using the existing
in-session samples. They are no longer fixed sidebar panels or hidden at narrow
widths. No new poller, backend authority or resource/tool registration is added.
Placement, pins, locks and dismissal use the existing metadata-only persistence.
Member-tagged samples prevent the previous member's charts from rendering;
member/daemon changes reset the sample history. Disconnect hides the payloads.
These charts remain session telemetry, not persistent task-history records.

Validation: **128 Desktop tests** and the TypeScript/Vite and macOS ARM native
builds pass. Browser checks against the disposable IPC-backed fixture verify
navigation filtering, draft preservation, command preparation without sending,
Escape, adding both charts, disabled removal while locked, and stable single
cards/locks after reload. At 1440 × 900 and 760 × 600 the composer and footer
remain inside the viewport, with no horizontal overflow. The narrow check caught
an old 44 px chart rule inherited from the sidebar; it was removed and both SVGs
now measure 259 px wide in the 324 px drawer at the minimum. Both skins were
inspected in the browser. No Python source changed; the recorded Python baseline
and follow-up ownership above remain applicable.

### Native follow-up review — 2026-10-30

Reviewed the macOS ARM package built from `1684a378d08762115113b78bf5c42d2f825cf7c1`
in the uniquely identified `local.alphonse.issue78.review` bundle. Both socket and
database were explicitly set to the disposable fixture, and `current_user` was
verified as `review-owner` before launch and relaunch. This review used native
Tauri/WebKit, not the browser shim. All visible records are synthetic fixtures.

- Light and dark skins: sidebar hierarchy, bottom member/status/settings, single
  composer surface and compact cards were visually inspected against Study 01.
  Source dates, availability and counts remain explicit instead of copying the
  mockup's sample content. Screenshots are linked below.
- Navigation search narrowed to the synthetic trip and cleared normally. A draft
  survived navigation from Conversation to Household and back. While populated,
  command choices were disabled; Escape closed the menu and returned focus to the
  composer. With an empty draft, `/scheduled-tasks` prepared text without sending.
- Native windows were reviewed at 1240 × 760, 1440 × 900 and 760 × 600. At the
  minimum the drawer overlays the conversation; closing it restores the full
  composer and focuses its toggle. Sidebar icons, bottom settings and member
  identity remain accessible. At wide sizes the drawer scrolls independently.
- Added Workload and PDCA through the catalog; both charts use the drawer width.
  Locking Workload disabled movement and removal. After quitting and relaunching,
  exactly one of each chart remained, including Workload's lock, Agenda's lock,
  the pinned source cards, selected light skin and persisted conversation history.
  Removing To-dos removed its presentation only: the source JSON SHA-1 remained
  `0fd88c8e659df572da73a67ce1b0247d5f03efac` before and after.
- Expired unpinned context showed an explicit expiry message. Pinning restored its
  available source preview; no completed source item was manufactured by the UI.
- With explicit permission, macOS Accessibility → Motion → Reduce motion changed
  from **off** to **on**. Native keyboard Tab moved from the composer to its
  command disclosure with a visible focus outline. The app remained usable.
  Reduce motion was restored to **off**, confirmed in the native settings tree.
  CSS disables animation, transitions and smooth scrolling under that preference.
  This was a manual usability check, not frame-by-frame animation instrumentation.
- WCAG luminance calculations of the semantic colors: light/dark primary text
  against canvas **13.00/15.42:1**; muted text against card **5.58/7.24:1**;
  sidebar muted text **4.96/7.41:1**; primary action text **7.65/10.14:1**;
  focus against canvas **7.13/11.20:1**. These are token-pair checks, not a claim
  that every possible data visualization or disabled state was audited.

The final-renderer sample used PID 43967, `normal` mode, 60 seconds warm-up,
300 seconds observation and 30 second intervals, with long conversation history
and native navigation/widget interactions during observation. Physical footprint
was **71.1–86.6 MB**, ending at **86.3 MB**, with a reported lifetime peak of
**232.2 MB**; live allocations ended at 236,211. The short mixed-interaction run
and responsive scrolling support this native review. It does not prove leak-free
endurance or replace regular household use. Raw measurements:
[final WebKit sample](evidence/78/final-webkit-memory.csv).

Native images: [light shell](evidence/78/native-shell-light.png),
[dark shell](evidence/78/native-shell-dark.png),
[diagnostic widgets](evidence/78/native-diagnostic-widgets-dark.png),
[minimum composer](evidence/78/native-minimum-composer-dark.png),
[reduced-motion focus](evidence/78/native-reduced-motion-focus.png).

This completes the bounded native layout/accessibility/performance review for
this code revision. It does not complete real household processor workflows,
all ownership/delegation and cancellation/failure scenarios, or the explicit dated
owner acceptance after regular use. The fixture's receipt processor remains
synthetic; it cannot qualify real hierarchical processing. No source changed
since the passing 128-test/frontend/native build validation. PR #103 stays draft,
#78 stays open, and #80 stays blocked until the remaining criteria are met.
