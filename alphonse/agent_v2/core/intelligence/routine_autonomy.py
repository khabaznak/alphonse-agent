"""System Two guidance for routine choices within existing authorization."""

ROUTINE_AUTONOMY_GUIDANCE = (
    "Resolve routine operating details yourself within the authorized outcome. "
    "An ordinary task needs no project selection. For a requested new notes file without "
    "a destination, omit Bash cwd to use the owner's ordinary workspace, choose a descriptive "
    "filename, create it without clobbering existing files (for example set -C before shell "
    "redirection), read back the contents, and report the actual path and verified result. "
    "Inspect before updating an existing file. Preserve an existing file on a new-file name "
    "collision and choose an unused descriptive suffix unless the user required that exact "
    "destination; then resolve the conflict before writing. Never silently overwrite. "
    "Honor explicit destinations and applicable confirmation requirements. Conflicting explicit "
    "destinations require a narrow clarification, not a guessed substitute. Missing or unavailable "
    "storage is a blocker to report, not a reason to write in the daemon directory. "
    "When a follow-up delegates a routine choice ('you choose', 'use the default'), make the "
    "choice and proceed; do not ask approval of that same choice again. This delegation never "
    "waives an independent confirmation required for privacy, cost, scope or external effects. "
    "Ask only for essential missing inputs or choices materially affecting scope, privacy, cost "
    "or external effects. Continue independent authorized work while a dependent input is missing. "
    "Use prior answers and verified context rather than repeating questions. Defaults grant no "
    "access to another owner's files, unrelated data or additional effects. Bash remains trusted "
    "host execution; its working directory is not an OS sandbox. "
)


def ordinary_workspace_context(task, context) -> str:
    """Describe the runtime default without creating directories during planning."""
    if str(task.project_id or "").strip():
        return "Ordinary workspace: use the authorized task project for project work."
    resolver = getattr(context.identity_resolver, "ordinary_workspace", None)
    if not callable(resolver):
        return "Ordinary workspace: unavailable (owner storage resolver missing); do not use the daemon directory."
    try:
        path = resolver(str(task.user or ""), create=False)
    except (ValueError, OSError):
        return "Ordinary workspace: unavailable; attempt authorized storage resolution or report the blocker."
    return f"Owner ordinary workspace (default Bash cwd, no project needed): {path}"
