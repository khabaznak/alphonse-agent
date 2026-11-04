# Evidence-based feasibility assessment

Issue #155 adds shared System Two guidance in `v3/feasibility.py` to strategic
planning and tactical selection. The reporting portion also reaches terminal
failure explanations, where execution has already stopped. It replaces the
blanket instruction to attempt tools before any inability claim: an explicit
denial or unavailable authorized check can justify stopping without an unsafe
experiment.

When a dedicated tool is absent, System Two considers selected workflows,
composition of authorized native tools, existing CLIs and small temporary
programs. The native Bash descriptor and static Jev relevance question make
runtime/dependency inspection discoverable even when users name no library.
Jev still answers tool-relevance Nouls; System Two chooses approaches and commands.
Positive or ambiguous relevance never grants execution permission.

Assessment normally uses one inspection and up to two materially different
follow-up checks with command timeouts. This is planning guidance, not a new
filesystem sandbox or a new deterministic counter. Existing rejected-action,
failed-command, semantic-duplicate, cancellation and outer no-progress stops
remain in force. Replanning must reuse evidence rather than restart investigation.
An installation, external submission or credential change requires its applicable
authorization; a denial must not be circumvented through another capability.

Reports distinguish:

- An approach not investigated: feasibility remains unverified.
- A missing dependency: a prerequisite was absent in the runtime checked.
- An observed execution failure: the actual attempted operation failed.
- Missing essential input: a specific field is needed for a dependent action.
- Explicit access denial: an actual policy or service refusal was observed.

A failed import does not prove that every runtime or alternative is unavailable.
A successful import does not prove a form was submitted. Missing personal fields
allow independent inspection and a clearly marked placeholder draft; placeholders
must not be submitted as personal data. Capability-only questions warrant an
assessment, without activating the operation or creating a proposed schedule.

## Verification

`tests/test_agent_v3_feasibility.py` runs the real planner, reveal policy, tactical
executor, native Bash process and terminal failure reporting against disposable
localhost HTML forms. Minimal disposable modules represent installed, absent and
broken prerequisites. User prompts name no automation library. Cases cover:

- Inspecting required fields and preparing a placeholder with no POST.
- An absent import and broken initialization preserving distinct error evidence.
- A standard-library path that needs no optional dependency.
- HTTP 403 preserving failure despite an optimistic semantic judgment.
- Rejecting scheduling outside a capability assessment's allowed phase.
- A timed-out check stopping without a repeated command.

Model choices and Jev responses are scripted in these regression tests. They
verify prompt delivery, execution evidence and deterministic boundaries; they do
not measure live model compliance. A live-model evaluation should use the same
synthetic cases, record commands and requests, and score both evidence-based
reporting and absence of unrequested effects. It must not use real household
forms, account sessions or personal data.
