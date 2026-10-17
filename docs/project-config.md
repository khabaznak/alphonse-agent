# Project configuration

Each registered project folder has a machine-readable `project_config.json`.
It records project-specific context and is the only project context file loaded
for V3 curation. Existing `project_context.md` files are preserved as-is for
their owner's manual review and are never used as prompt context.

The initial schema is:

```json
{
  "schema_version": 1,
  "scope": "",
  "goals": [],
  "owner_user_id": "user-id",
  "duration": {
    "ongoing": true,
    "start_date": null,
    "end_date": null
  },
  "directory_structure": []
}
```

`goals` is a list of project-specific desired outcomes. `duration` may describe
an ongoing effort or use ISO 8601 dates. `directory_structure` is a declared
layout; it does not authorize filesystem changes by itself. Project files remain
inside the registered project folder.

Project lifecycle states are `active`, `paused`, `completed`, and `archived`.
Sharing is private or shared with explicitly selected family-member user IDs.
The owner is always retained in the project registry and cannot be changed by
editing this config.

The first daemon start after this change converts legacy family-wide shared
projects into grants for all currently active family members. This preserves
their existing access while recording it explicitly; owners can then narrow
the member list in project management.
New shared projects do not receive automatic member grants.

Project management lists up to 1,000 visible non-hidden files below the project
folder. File additions copy user-selected files into the project root. Removal
is limited to regular files inside the project folder; config and legacy
context files are protected. The declared directory structure does not itself
create or delete folders.

## Existing project migration

When the project store initializes, it writes a default `project_config.json`
only where one does not already exist. It leaves project folders and existing
`project_context.md` files untouched. Owners can review and extract useful
material from legacy files before deleting them themselves.

Jev's System One context curation receives only authorized project-config
candidates. It can select multiple projects; every candidate whose relevance
probability meets the configured threshold is retained, including equal-scoring
candidates. Project files and config are data, not instructions or authorization.
