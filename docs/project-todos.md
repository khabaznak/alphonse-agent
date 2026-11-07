# Project to-dos

Project to-dos are explicit household records, separate from conversation
checkpoints, scheduled work and skill-owned lists. The native store lives in the
existing user-owned scheduling database. Each record has an owner, a project,
a stable ID, a title and a pending, completed or cancelled status. No existing
conversation or skill data is imported or reclassified.

Project details lists the selected owner's to-dos alongside schedules. The
owner can add an item or explicitly mark it complete or pending. Desktop IPC
binds writes to the current authenticated member; it verifies project access
and never changes another owner's item. Project access is checked again on
reads and writes. Account deletion removes that owner's to-dos. Deleting a
project does not migrate its records into another project.

Alphonse can use `native.project_todo` to create an explicitly requested to-do
in the associated project or in an active project named in the current request, and `native.project_todo_list` to
inspect its owner's records. Creating is a project mutation, requires an
active authorized actor and accessible active project, checks cancellation,
and returns a verified store readback. Request-scoped idempotency prevents
retries from duplicating an item. The native capability does not automatically
complete to-dos: status changes are explicit in the project view, so producing a
reply or running an unrelated task cannot manufacture completion.

A requester can name an active project directly in ordinary conversation. The
native tool resolves an exact, case-insensitive name or explicitly supplied ID
against currently accessible projects and checks the current requester’s task
turns. Missing, inaccessible or ambiguous targets require clarification; the
model cannot assign an unmentioned project. An explicitly attached project
continues to work, and the conversation itself remains projectless by default.
Historical conversations and skill-owned to-do snapshots are not silently adopted.
