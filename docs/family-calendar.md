# Family calendar package adoption and validation (#41)

`alphonse/agent_v2/builtin_skills/family-calendar/` is explicitly adopted as a shipped
reusable Agent Skills package. This is product-maintained service support, not a
copy of an individual household artifact. Its SKILL.md, ordinary Python script and
references use existing skill discovery/installation, account bindings and native
Bash. There are no calendar tools, script manifests or catalog entries. Package
exports contain no account state. See its references for connection, request and
authorization procedures. No automatic invitations are supported.

Future project-scoped read context and incremental sync extend this package under
the [optional-account implementation contract](optional-account-integrations.md)
(#119). They reuse its account bindings and preserve exact write authorization,
readback and retry guards. The current script is not a proactive acquisition worker
and does not yet enforce per-page project retrieval windows or secure credential
references. No new Calendar tool or per-account registry is required.

## Synthetic validation

Run `python3 -m pytest -q tests/test_family_calendar.py tests/test_agent_v2_skill_bindings.py
 tests/test_skill_script_execution.py tests/test_agent_v2_skills.py` (on one line).
The calendar tests inject synthetic OAuth and HTTP responses; no real network or
calendar records are used. Binding and native execution tests cover authenticated
actors, sharing/removal, revoked accounts, redaction, cancellation and checkpoints.

## Opt-in real-account validation

Do not run during tests, installation or daemon startup. A human must explicitly
opt in, connect their chosen test Google account through trusted setup, select an
empty dedicated test calendar and review its ACLs. Never use a production family
calendar for smoke validation. Keep all resulting records and tokens outside Git.

1. Confirm Python/timezone readiness. List a finite date window and compare to the
   Google UI. Query free/busy and verify busy intervals; a provider error is unknown.
2. Confirm an unselected actor/calendar is rejected before any network call. Disable
   the binding and confirm managed denial; re-enable only with explicit owner action.
3. Only after separate approval for a real write, provision a short-lived exact hash
   grant. Create a uniquely identified test event with no attendees and
   `send_updates=none`. Read back its ID, title, offset, timezone and dates. Repeat the
   exact request with the same operation ID and confirm one event, including after
   simulating interrupted readback. This test must not send invitations.
4. Read the ETag, approve the precise update and grant its hash. Change the title,
   read back, and repeat to verify no second PATCH. Make a manual conflicting edit
   and confirm the old ETag fails without overwriting it.
5. With explicit series approval, test a COUNT=2 weekly series across a daylight
   saving transition in a relevant IANA timezone. Confirm local wall time in Google,
   list expanded instances, then authorize only one instance update and verify that
   the other occurrence is unchanged. Confirm ambiguous dates/nonexistent local
   times require clarification or fail before mutation.
6. Revoke OAuth consent and confirm reconnect guidance without token/error-body
   disclosure; test missing scopes using a separate read-only consent. Remove test
   events manually in Google (deletion is outside this package), disable grants and
   test bindings, and retain only a redacted pass/fail report outside the package.

Real-account validation is intentionally not claimed by the synthetic suite. Google
provider delivery/concurrency behavior remains an opt-in integration check. Requests
with attendees, invitation delivery, conferencing, deletion and complex recurrence
are outside the initial implementation and must never be approximated silently.
