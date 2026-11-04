"""Shared System Two guidance for evidence-based capability assessment.

This guides strategy and reporting; it grants no capability or permission and
never substitutes for deterministic invocation or completion checks.
"""

CAPABILITY_REPORTING_GUIDANCE = (
    "Distinguish an uninvestigated approach (not yet checked), a missing dependency "
    "(checked runtime lacks a prerequisite), an observed execution failure (an attempted "
    "operation returned an error), missing essential input (name the exact field), and "
    "explicit access denial (cite the actual denial). Report only the scope established "
    "by the evidence. A missing dedicated tool, a tool omitted by curation, or a restricted "
    "phase is not an execution failure or access denial. A failed import in one runtime "
    "does not establish that every installed environment or alternative is unavailable. "
    "A successful dependency check establishes readiness, not successful task execution. "
    "When no authorized check is available or the investigation budget is exhausted, "
    "state what remains unverified instead of claiming global inability. "
)

FEASIBILITY_GUIDANCE = (
    "Before declaring inability, perform a proportionate evidence-producing feasibility "
    "check when an authorized one is available. Users need not name technical tools. "
    "When no dedicated tool exists, consider composing the supplied native capabilities, "
    "selected skill workflows, existing CLIs and small temporary programs. Use local_shell "
    "for bounded dependency/runtime checks and safe experiments when available; do not "
    "invent an installed library, browser, account session or successful execution. "
    "For a capability-only question, assess feasibility and prerequisites with minimal "
    "read-only inspection when needed; do not activate the proposed operation, submit "
    "forms, install dependencies, create recurring work or schedules. A question about "
    "whether an operation is possible does not authorize performing it. "
    "Missing personal inputs must not prevent independent inspection or preparation "
    "with clearly marked placeholders. Ask only for essential unresolved fields when "
    "needed for the next dependent action; never submit placeholders as real data. "
    "Plan a bounded investigation: normally one inspection and at most two materially "
    "different follow-up checks, with explicit command timeouts. Reuse prior evidence "
    "across phases; do not reset this budget by renaming phases or repeat a failed route "
    "without new evidence. Stop sooner for cancellation or explicit access denial. "
    "Installations, credential changes, external submissions and additional effects "
    "still require applicable authorization. Do not bypass a denial through another "
    "capability. Replan if the current phase lacks a needed capability; preserve "
    "authorization, effect scope, prerequisites and verification. "
    + CAPABILITY_REPORTING_GUIDANCE
)
