# Ordinary task workspaces

An ordinary conversation does not need a project. Native Bash with no `cwd` and
no selected skill package uses the active owner's `users/<user-id>/workspace/`
directory beneath the configured Workspace. Legacy installations use the same
subdirectory beneath their existing users root. The directory is created lazily
with private permissions; planning can describe it without creating it.

Explicit working directories and authorized project directories take precedence.
A missing or inaccessible project, unknown or inactive owner, unavailable storage,
or linked owner/workspace directory must fail rather than fall back to the daemon
working directory. Changing the configured Workspace copies ordinary workspace
files without replacing existing destination files and preserves the old source.

For a requested new notes file, Alphonse chooses a descriptive filename, creates
it without clobbering existing files, reads it back, and reports its actual path.
A name collision calls for an unused suffix unless the user required that exact
name. Conflicting explicit destinations need a narrow clarification. A follow-up
that delegates a routine choice should lead to action without asking for the same
choice again. Essential inputs and decisions affecting scope, privacy, cost or
external effects still require the applicable answer or confirmation; independent
authorized preparation can continue.

The default is a destination, not an access grant or filesystem sandbox. Native
Bash remains trusted host execution with its existing authorization and effect
checks. No artifact catalog or generated program is changed by this feature.

`tests/test_agent_v3_routine_autonomy.py` exercises disposable owner stores, actual
shell writes/read-back, collision preservation, explicit destinations, unavailable
storage, linked roots, owner isolation, migration, and authenticated question
resumption. Model decisions are scripted in the resumption regression; it proves
the guidance, answer and default path reach the planner/executor and the chosen
action produces evidence, not a measured live-model success rate.
