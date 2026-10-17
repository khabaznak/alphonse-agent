# Skill-owned artifact resources

Projects track goals. A conversation does not need to select a project to use an
installed skill's authorized artifacts. Artifact programs remain user-generated
resources, never native tool definitions or independent Jev curation candidates.

## Storage and fresh installations

The runtime initializes `<workspace>/artifacts/` alongside `<workspace>/skills/`.
Without a configured workspace the default is `~/.alphonse/artifacts/`.
New registrations default to shared storage:

```
<workspace>/artifacts/<skill-name>/<artifact-id-without-prefix>/
```

For example, `artifact.sensor-reader` belonging to `skill:household` has the
storage root `<workspace>/artifacts/household/sensor-reader/`. Alphonse creates
and maintains its program, dependencies and configuration there. The native
registration tool only validates existing resources and records their metadata.
It never writes, copies or repairs the artifact implementation.

Each registration includes its owning skill, owner, optional explicitly shared
household user IDs, enabled state, JSON Schema, entrypoint and execution metadata:

```json
{
  "storage_kind": "shared",
  "entrypoint_path": "reader.py",
  "execution": {
    "working_directory": ".",
    "runtime_path": ".venv/bin/python",
    "configuration_paths": ["config.env"],
    "required_files": ["reading.sqlite"]
  }
}
```

Entrypoints and resource paths are relative to the storage root, never the task's
project or daemon cwd. Traversal, absolute resource paths and symlink escapes are
rejected. An explicit runtime can be an absolute executable or a contained runtime
path; a virtual-environment interpreter may itself link to the system executable.
Python resources require a registered runtime rather than implicitly using
`python3` from PATH. No dependency installation or authentication repair happens
in the resolver. Missing declared files are recorded as availability issues;
configuration contents and credentials are never read into resource manifests.

`skill` storage supports existing files inside the installed skill package.
`project` storage supports legacy locations, with the original owner/root
containment requirements preserved. Neither kind depends on the task's project.

## Runtime use and authorization

Selecting a skill supplies its base instructions and the catalog-backed resource
manifests and operating instructions the requesting user can access. Artifact
sections belonging to other users are not loaded merely because the skill is
selected. Access is granted to the owner, administrators, and explicitly shared
user IDs, never by Jev or by membership in the current task's project.

An authorized shell call can reference the resource:

```json
{
  "artifact_id": "artifact.sensor-reader",
  "command": "\"$ALPHONSE_ARTIFACT_RUNTIME\" \"$ALPHONSE_ARTIFACT_ENTRYPOINT\" status"
}
```

Before spawning Bash, native code checks that the owning skill is selected, the
user is authorized, the registration is enabled, and the resource is ready. It
supplies the registered working directory and `ALPHONSE_ARTIFACT_ROOT`,
`ALPHONSE_ARTIFACT_ENTRYPOINT`, and `ALPHONSE_ARTIFACT_RUNTIME`. An incompatible
explicit cwd is rejected. The shell command and requested effects remain subject
to the ordinary native-tool/phase controls. This is a resource resolver, not an
OS sandbox for arbitrary generic Bash commands. These controls do not guarantee
that the artifact implementation or its dependencies work.

## Existing installations and reconciliation

Database upgrades add metadata fields with legacy-compatible defaults. Existing
IDs, skill associations and JSON Schemas are preserved. The runtime snapshots the
legacy storage project's authorized root so archiving or deleting a goal project
does not hide an artifact. No program, environment, configuration or data files
are moved by this migration. Missing project records without a captured root are
reported as unavailable; their path is not reconstructed by guessing.

The `reconcile_artifacts` IPC request reports availability and exact known-location
candidates under registered shared, skill or legacy roots. It never chooses among
copies or scans unrelated filesystem locations. A stale or ambiguous registration
requires the intended resource to be identified before changing its location.

Desktop's Skills workspace shows resource status, resolved locations and editable
location/runtime/configuration/sharing metadata. The TUI `/artifacts` screen shows
the same diagnostics and accepts location metadata as JSON. `relocate_artifact`
IPC and the native catalog metadata-update tool perform manager-authorized,
validated updates and synchronize the owning skill definition, preserving its
operating instructions and argument schema. Neither changes artifact files.

Unregistration never deletes program or data files. Skills with registered
artifacts cannot be uninstalled or replaced until their artifacts are preserved
and unregistered. Workspace changes copy the shared repository as a whole, retain
the old files and reject a nonempty destination rather than merge conflicting
copies. Legacy roots are refreshed from the explicit project-root migration.

Artifact-specific dependencies, configuration handling, operational instructions
and implementation repairs belong to Alphonse and its user's artifact, not to
this product repository.
