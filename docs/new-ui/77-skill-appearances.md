# Skill-owned mascot appearances — #77

Alphonse retains the original base artwork and idle, planning, doing, checking,
acting, error and disconnected state mapping. The existing entrance/bob motion
and global reduced-motion rules remain in use. Optional outfits dress the same
ivory/teal robot; they do not add tools, expertise, credentials or permission.

## Package contract

The optional `appearance` field lives in `SKILL.md` YAML frontmatter. Version 1
supports a shipped preset or package-owned static PNG states:

```yaml
appearance:
  version: 1
  preset: doctor # or exercise-coach
```

```yaml
appearance:
  version: 1
  label: Gardening assistant
  states:
    idle: images/idle.png
    planning: images/planning.png
```

Supported keys are `idle`, `planning`, `doing`, `checking`, `acting`, `error` and
`disconnected`. Missing state keys use the corresponding original base image.
Each path must be relative to its skill package, without traversal, and resolve
inside that package. Absolute paths, outside symlink targets, remote URLs,
SVG/HTML, APNG, corrupt PNG structure/CRC, dimensions over 4096, individual files
over 4 MiB and combined resources over 8 MiB are rejected. An invalid or missing
resource makes the optional appearance unavailable without disabling the skill.
Imports and updates preserve symlinks rather than copying outside targets into
the package and inadvertently laundering their containment. The normal package
owner/admin checks and artifact registration protections continue to apply.

The daemon returns a content revision, state/resource manifest and, only on an
authorized `skill_appearance` request, inline PNG resources. Desktop polling
contains no image bytes, package paths, instructions or private task contents
in its mascot context. Data URLs avoid granting a general host-file image
protocol. CSP adds only `data:` to `img-src`; other resource policies stay intact.
The browser also falls back if a valid PNG container fails to decode.

`doctor` and `exercise-coach` use shipped static 4×2 atlases: seven states and
one unused idle cell. They are resource presets, not new installed capabilities.
Settings > Skills explicitly binds either preset to an editable installed
skill, preserving its name, description, instructions and artifact definitions.
There is no inferred binding based on a skill's name or description. This lets
the existing family-doctor package opt in and makes the coach set available
when that essential skill is installed, without pretending to install it.
Custom packages keep their own state paths. Selecting a preset replaces the
optional appearance field; selecting base removes it. Package replacement uses
the new package's metadata, and removal makes its resources unavailable.

## Selection and preferences

The presentation layer consumes the existing persisted `v3_skill_selection`.
One selected primary skill leads; when none is primary, one selected skill may
lead. Multiple primary skills, multiple selected skills with no leader, missing
packages and unavailable appearances fall back to base. No additional model
classification, LLM routing or Jev permission decision is introduced.

Only the current user's active, nonterminal, uncancelled task can supply an
appearance. A task starts with base until its selection checkpoint is ready.
Direct conversational replies persist that existing decision before composing
their response, so they can show an outfit without entering a tool phase.
Desktop latches the first decided outfit for that task across CAPD states. A
later conflicting, ambiguous or unavailable selection returns to base for the
rest of the task, preventing oscillation. A new task can choose a new outfit.
Content revisions allow an updated image for the same leading skill. Stale
resource responses are ignored, including after user/skill/revision changes.
Disconnection clears the task context and preserves the base disconnected
expression. Completion, failure, parking and cancellation clear the costume.

Settings > Skills shows all seven labeled previews and a credential-neutral
description. Global and per-skill display preferences persist per desktop user;
they do not modify the skill package or its permissions. Outfit bindings use
an explicit administrator-authorized save. The static previews do not animate,
and reduced motion disables the sidebar's existing entrance and bob effects.
Screen readers receive the current activity and outfit name as one image label;
the inner image is decorative.

## Artwork provenance

The two PNG atlases in `alphonse/agent_v2/mascot_assets/` were created with the
built-in imagegen tool, using `desktop/public/alphonse-states/idle.png` as the
identity reference. The coach also used the doctor atlas as a layout reference.
Prompts specified the same ivory spherical robot, teal fins, metallic trim and
turquoise screen eyes; an equal-cell 4×2 transparent sheet; and the seven states
in the order above. Doctor costume: ivory lab jacket, teal stethoscope, neutral
badge, no red cross or credential text. Coach costume: teal/coral athletic vest,
coral sweatband, silver whistle and tiny dumbbell. The final cell repeats idle.
These are finished raster outfits, not the mockup's cap/headband overlays.
The base PNG files remain byte-for-byte unchanged.

Setuptools package data and the standalone daemon's `--collect-data` build flag
include the assets for installed and packaged deployments.

## Validation

- All 69 desktop tests and the TypeScript/Vite production build pass.
- 75 focused Python regressions pass across appearances, Settings lifecycle,
  artifact storage, IPC and V3 task admission. Appearance fixtures cover install,
  update, removal, unsupported contracts, partial states, missing resources,
  PNG validation, traversal, absolute paths, symlink escapes before and after
  import/update, administrator restrictions, cancellation, terminal cleanup,
  user-scoped task selection and ambiguous primary skills.
- Production-build Chromium checks with in-memory Tauri/daemon responses verify
  Settings > Skills, both outfits/all seven decoded state previews, light and
  dark skins, per-skill/global preference persistence, explicit outfit-save IPC,
  activity changes without costume changes, conflicting leader/ambiguous
  fallback, new-task selection and disconnected fallback. Reduced-motion
  emulation confirms `animation-name: none` on the sidebar mascot. Layouts at
  760, 960 and 1440 pixels have no document horizontal overflow.
- A disposable PyInstaller standalone daemon build succeeds. Its archive
  contains both PNG atlases, and the extracted bytes pass the static PNG
  validator. Packaging uses the same `--collect-data alphonse.agent_v2` option
  now included in the sidecar build script.
- The original base images are unchanged. No household skill package, generated
  artifact, live account, medical record or credential was modified or copied
  into product source. Screenshots use disposable fixture names and records.

Browser fixtures validate the presentation contract; real filesystem lifecycle,
task checkpoints and permission checks are tested separately through Python/IPC.
This does not claim a live medical/exercise workflow or macOS WebKit UI run.
Full household workflows and daily-use acceptance remain tracked by #78.
