# Candidate recovery in the synthetic comparison

This post-observation experimental change lets the simple-loop candidate retry
an unverified answer at most twice when at least one advertised, authorized
fixture source has not been read. The retry receives generic feedback and no
answer key or list of required sources. Retry state is checkpointed. Read
permission, cancellation, source freshness, deadlines and budgets still apply.
There is no production routing change and no user-facing question delivery.

This is a candidate-only mechanism, so the arms do not have identical recovery
logic. The candidate's synthetic verifier also checks whether required source
references are present; its acceptance/rejection is therefore a source-coverage
signal unavailable to V3 in the same form. Eligibility for a retry uses only
advertised source names, completed read references and host authorization, not
the hidden required-source list. This limits but does not eliminate the
comparison asymmetry. The independent post-run oracle still evaluates both
arms. Keep results from before this change separate from results after it;
scripted-provider tests establish harness behavior, not live effectiveness.
