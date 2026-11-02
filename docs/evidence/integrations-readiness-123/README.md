# Integration readiness evidence — 2026-11-02

Issue: [#123](https://github.com/khabaznak/alphonse-agent/issues/123).
Decision: **pilot BLOCKED; public release BLOCKED**.
See [operator checklist and blocker details](../../integrations-readiness.md).
No real accounts, provider writes, remote inference or host Keychain entries used.
All credentials, messages, finding facts and backup files were disposable fixtures.
The operator confirmed no deployment registration/processor records exist yet.

## Revisions and validation

Inspected main: `41cfd25e05d42e6eaf035a2a590080160d1a57bb`.
Tested implementation: `478a202031ecd9e8431af0e40ba62ee32cc847fa`.
The subsequent evidence commit changes documentation only; the implementation PR
records the final branch SHA. Tests ran in an isolated worktree; unrelated dirty
work in the original checkout was preserved.

Initial prerequisite suite at inspected main: **146 passed, 2 skipped**, no failures.
The two skips required `requests_oauthlib`. Expanded suite after fixes, before
installing optional dependencies: **297 passed, 37 skipped**, no failures.

Final synthetic suite at the tested implementation: **573 passed, 35 skipped**,
18.99 seconds. The remaining skips are solely the optional external `skills_ref`
validator (34 conformance cases and one installed-validator check), not credential,
OAuth, retention or revocation tests. Python 3.12 from the existing project venv;
`requests-oauthlib==2.0.0` and `oauthlib==4.0.1` were installed under a disposable
`/private/tmp/alphonse-issue-123-deps` directory to exercise the real OAuth library
against mocked HTTP. No project dependency or shared environment was modified.

Reproduction (install the declared OAuth/credential extras in a disposable test
environment; `skills_ref` is a separate optional conformance dependency):

```sh
PYTHONPATH=/private/tmp/alphonse-issue-123-deps python -m pytest -q -rs \
  tests/test_secure_credentials.py tests/test_oauth_lifecycle.py \
  tests/test_account_grants.py tests/test_source_acquisition.py \
  tests/test_findings.py tests/test_agent_v2_event_recovery.py \
  tests/test_agent_v2_scoped_events.py tests/test_agent_v3_event_evidence.py \
  tests/test_agent_v2_skills.py tests/test_project_checks.py \
  tests/test_project_check_scheduling.py tests/test_gmail_adapter.py \
  tests/test_graph_mail.py tests/test_graph_mail_acquisition.py \
  tests/test_calendar_acquisition.py tests/test_classroom_adapter.py \
  tests/test_family_calendar.py
```

`git diff --check` passed. No Actions workflows exist in this revision's
`.github/workflows`; local results are not a claim of hosted CI or live-provider
qualification. An initial expanded command named nonexistent provider test files;
pytest collected nothing. The corrected command above is the reported final run.

## What the evidence establishes

- Credential fixtures exercise an encrypted-file vault, injected native macOS API,
  missing/locked backends, no plaintext fallback, generation/refresh races,
  projections/redaction, disconnect, pending deletion, and OAuth provider revocation
  semantics. Missing optional packages remain isolated from unrelated startup.
- Source fixtures exercise bounded acquisition, source tombstones, expiry, current
  grants/windows, independent processing checkpoints, cursor replay and truthful
  incomplete coverage. Scheduler fixtures enforce windows and policy revisions.
- Strengthened finding tests deny current and historical versions, delivery status,
  new notifications and outbox claims after source deletion/revocation/unknown state.
  Cleanup removes all stored finding versions and redacts local outbox text.
- A new historical-version test demonstrates an older version cannot bypass its
  own retention deadline through a newer current version.
- Cleanup distinguishes inactive-policy cache expiry from source revocation after
  rechecking pinned account/resource authority. Existing pause/cache-expiry history
  tests remain green; revoked source history becomes unavailable.
- A synthetic SQLite backup/restore test deliberately restores stale evidence and
  findings after grant revocation. Current independent account authority denies
  current/historical views, evidence/checkpoint expansion, delivery status and
  outbox claims before cleanup; cleanup then redacts restored bytes and versions.
  This proves the requirement to preserve current authority. It does **not** qualify
  a full deployment restore that might also restore stale account/grant databases.
- Existing event recovery tests fence work more than 90 days from trusted acceptance;
  source/finding references cannot manufacture authority. Existing skill lifecycle
  tests cover portable package export/private-state rejection. No private evidence
  export service or general copied-content prompt-cache pipeline is qualified.

## Acceptance criteria status

1. **Incomplete:** exact Google/Microsoft deployment registrations absent (G1/M1).
   Scope/account-type/verification requirements are documented; approval is not.
2. **Achieved:** off-host Jev/System Two flows, public terms, local CLI persistence,
   user-facing disclosure draft and unresolved processor prerequisites recorded
   explicitly (P1/P2/P3). Applicable deployment agreement/settings remain blockers.
3. **Incomplete:** adopted limits and dependent-view regressions documented, but
   30-day metadata expiry, safe replay receipt pruning and full backup/export
   retention/restore qualification remain unresolved (R1/R2).
4. **Achieved for the supported implementation:** synthetic credential/OAuth,
   leakage/redaction and disconnect/revocation evidence passes. Actual host backend
   qualification is an explicit operator checklist action, not a test claim.
5. **Achieved:** dated pilot and separate public-release checklists published with
   evidence and operator/engineering ownership. Both gates remain blocked.

Leave #123 open with criteria 1 and 3 unchecked. Closing it requires their actual
completion and updated dated evidence. No implementation merge changes this status.
