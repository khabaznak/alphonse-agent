# Settings workspace — #66

Settings now occupies the main workspace and uses the contextual shell sidebar.
General, Appearance, Tools, Integrations, Automations, Model, System One, Agent
configuration, Skills and Members retain the existing supported APIs. Skills
is a Settings screen rather than a separate top-level workspace. Members
reuses the existing user, role, context and communication-address editor;
Household retains its existing administrator management entry point.

Screens mount on first visit and remain hidden afterward, preserving drafts,
validation feedback and independent scrolling. Agent configuration additionally
keeps a draft per document and ignores stale reads after selection changes.
Skill/file switches ask before discarding edits; cancelling retains the draft.
Folder installation and package updates use `install_skill` and `replace_skill`.
Existing create, edit, file import/read/write/delete, folder opening, uninstall,
artifact attachment, reconciliation, relocation and catalog removal remain
available. Package replacement and uninstall retain daemon protections for
registered artifacts. Artifacts remain skill resources, never runtime tools.

The local current-user response gates privileged screens, including Skills and
Members. Members see General's immediate desktop preferences and Appearance;
privileged editors are not mounted for them. Daemon authorization remains the
authority. Credentials are password fields; saved secrets are not read back.

An existing account missing workspace configuration enters General directly.
Other sections and return navigation stay unavailable until a workspace save
succeeds. Initial administrator creation still uses the existing onboarding
flow. A timezone failure after a successful workspace save is reported as a
partial save. Saving delivery settings uses the persisted workspace directory
without applying or discarding a pending directory edit.

Drafts last for the current app session. Appearance, messaging and desktop
alerts apply immediately; other sections use their existing explicit saves.
This is stated in the Settings header.

## Validation — October 29, 2026

- 63 desktop tests pass; TypeScript/Vite production build passes. The existing
  bundle-size advisory remains.
- 157 focused Python tests pass across IPC, inference settings, System One,
  users, web/Code Mode, integrations, media, artifacts and resources, plus the
  new disposable Settings lifecycle tests.
- The new IPC tests install and replace a disposable skill directory, update
  metadata, import/read/write/delete files, uninstall, reject member mutations,
  attach a minimal artifact fixture, enforce package-replacement protection,
  confirm no runtime artifact tool exists, and unregister while preserving the
  program. No household program or data is adopted into product source.
- Chromium UI verification used in-memory Tauri/daemon responses. Timezone,
  integration token, System One key, new-skill and file drafts survived Settings
  navigation. Cancelling skill/file discard retained edits. Agent configuration
  drafts survived switching documents. Integration save cleared its token.
- Model and System One validation failures remained visible; a successful
  workspace save followed by timezone rejection showed the partial-save
  message. Initial configuration showed only General and a disabled return;
  saving unlocked the complete sidebar. Member fixtures showed only General
  and Appearance, with no password fields or workspace administration content.
- Folder install/update controls completed through their supported IPC methods
  in the UI fixture; real filesystem lifecycle behavior is covered by the new
  Python tests. Notification preferences persisted in local storage; existing
  notification tests cover delivery/focus behavior. Light and Dark selection
  remained functional.
- Settings and Skills were checked at 760, 960 and 1440 pixels without document
  horizontal overflow. Existing shell minimum is 760 × 600. The Settings header
  and sidebar remain available while the active screen scrolls.

No live model/Jev credential or household integration was changed. Browser
fixtures verify the UI contract; actual daemon validation and ownership are
covered separately by Python tests. Full household workflows and daily-use
acceptance remain #78. The native shell and packaging configuration are unchanged.
