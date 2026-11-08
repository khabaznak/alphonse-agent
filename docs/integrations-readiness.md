# Optional integrations readiness — #123

Review date: **2026-11-02**. Inspected baseline:
`41cfd25e` (full revision and commands in [synthetic evidence](evidence/integrations-readiness-123/README.md)).
Decision: **BLOCKED for real-account pilot and public release**. No real accounts,
provider consoles, household content or host Keychain entries were accessed.
The operator confirmed that deployment registration/processor records do not yet
exist. This review does not create consent or approve a deployment. Recheck the
linked provider policies and this record whenever scopes, processors, account
classes, distribution, retention or transports change.

Related contracts: [OAuth](oauth-accounts.md), [credentials](secure-credentials.md),
[accounts/project checks](optional-account-integrations.md),
[scoped evidence](scoped-event-evidence.md), [Gmail](gmail-readonly.md),
[Classroom](classroom-readonly.md), [Calendar](family-calendar.md).

## Exact deployment registration record

**G1 — BLOCKED, operator:** supply the Google Cloud project ID/number, OAuth
client ID, application type, owner/support contact, enabled APIs, exact redirect
URI, audience (Internal/External), publishing status, selected scopes, test-user
eligibility, consent/privacy URLs and verification/assessment decision with date.
Actual values: **not supplied; not verified**. Client IDs are public identifiers;
keep test-user addresses and provider-console evidence in restricted operator
storage. Record a redacted evidence locator here, never a secret or callback code.

The implemented configuration is `ALPHONSE_GOOGLE_OAUTH_CLIENT_ID` plus
`ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI`. It expects an installed/public desktop
registration and code/S256 PKCE. The service accepts an explicit-port/path
`http://127.0.0.1` callback or HTTPS; syntactic acceptance does not prove that a
provider accepts that client/redirect pair. Qualify the actual pair before use.
Do not embed a confidential web-client secret or adopt service accounts/domain-wide
delegation to work around a rejected flow. See [Google desktop OAuth](https://developers.google.com/identity/protocols/oauth2/native-app).

Consumer Gmail accounts need External eligibility. Internal means the owning
Workspace organization; it cannot cover unrelated household consumer accounts.
Workspace/school administrator restrictions apply independently, including
Classroom enrolled-student access; guardian status alone is insufficient.
Use separate test and production projects. Testing is not public approval;
External/Testing refresh tokens for these resource scopes expire after seven
days. Document whether a personal/development exception actually applies and
its limits. Public sensitive/restricted scope distribution needs the applicable
verification, permitted-use review and assessment determination. Gmail
`gmail.readonly` is restricted; server storage/transmission, including remote
inference, must be included in the assessment boundary. Do not assume a local
callback makes remote processing exempt. References:
[production configuration](https://developers.google.com/identity/protocols/oauth2/production-readiness/policy-compliance),
[refresh-token expiry](https://developers.google.com/identity/protocols/oauth2),
[restricted verification and assessment](https://developers.google.com/identity/protocols/oauth2/production-readiness/restricted-scope-verification).

**M1 — BLOCKED, operator:** supply the Entra tenant ID, application/client ID,
registration owner, `signInAudience`, native/public-client platform settings,
exact redirect URI, delegated permissions, consent policy/admin decisions,
publisher domain/verification status and privacy/support URLs, with dated evidence.
Actual values: **not supplied; not verified**. Configure
`ALPHONSE_MICROSOFT_OAUTH_CLIENT_ID` and `ALPHONSE_MICROSOFT_OAUTH_REDIRECT_URI`.
The implementation uses `/common` authorization/token endpoints. For the advertised
personal plus work/school coverage, register
`AzureADandPersonalMicrosoftAccount` (accounts in any organizational directory
and personal Microsoft accounts). Hotmail/Outlook.com are personal accounts.
A single-tenant or organization-only registration does not qualify this coverage.
School/tenant policies may require admin consent even for delegated permissions
that ordinarily permit user consent. Use a public client with PKCE, no client
secret, application permissions, mailbox impersonation or `revokeSignInSessions`.

Publisher verification is distinct from user/admin consent and from a security
assessment. Public distribution needs a dated publisher-verification decision;
verification needs an eligible organizational registration and partner association.
An unverified app can be blocked by tenant consent policy. Never describe a
successful synthetic token exchange as publisher verification. References:
[account types](https://learn.microsoft.com/en-us/security/zero-trust/develop/identity-supported-account-types),
[publisher verification](https://learn.microsoft.com/en-us/entra/identity-platform/publisher-verification-overview),
[redirect restrictions](https://learn.microsoft.com/en-us/entra/identity-platform/reply-url).

## Minimum implemented scopes

`integrations/oauth.py` is the executable allowlist. Request only the capabilities
actually chosen; a provider token's broader scope does not expand an Alphonse grant.

- Google always uses `openid` for identity; offline access is an authorization
  parameter. `mail_read`: `https://www.googleapis.com/auth/gmail.readonly`.
- Google `calendar_read`: `https://www.googleapis.com/auth/calendar.events.readonly`.
  Optional discovery is separately selected `calendar_discover`:
  `https://www.googleapis.com/auth/calendar.calendarlist.readonly`.
- Google `classroom_read`: `https://www.googleapis.com/auth/classroom.coursework.me.readonly`.
  Separately selected `classroom_announcements_read`:
  `https://www.googleapis.com/auth/classroom.announcements.readonly`.
  Each Classroom capability additionally requires
  `https://www.googleapis.com/auth/classroom.courses.readonly` and
  `https://www.googleapis.com/auth/classroom.rosters.readonly` for course/enrollment
  checks. Roster scope breadth must be disclosed even though the adapter checks
  only the connected student's enrollment. No Drive, submissions, writes or
  all-students coursework scope is requested.
- Microsoft uses `openid`, `offline_access` and selected delegated
  `https://graph.microsoft.com/Mail.Read` or
  `https://graph.microsoft.com/Calendars.Read`. No `User.Read` is requested by this
  implementation; qualify the actual OIDC userinfo identity path in registration
  evidence rather than silently adding permissions. `Mail.ReadBasic` cannot supply
  the message bodies/attachments needed by this adapter.

Provider permission descriptions and current sensitivity classifications must be
captured from the actual registration, not inferred from the word “readonly”. See
[Gmail scopes](https://developers.google.com/workspace/gmail/api/auth/scopes) and
[Graph delegated permission reference](https://learn.microsoft.com/en-us/graph/permissions-reference).

## Remote processing and disclosure

Both System One and System Two are off-host processors. Bounded source text,
excerpts, interpretations and task/conversation context can leave the household
host. Jev receives explicit decision questions and alternatives plus their bounded
content; its Noul/Choice/Score responses are judgments, never authority. System Two
receives deliberate reasoning inputs and composes language/tool calls. Source
credentials, authorization codes and callback URLs must reach neither processor.
Minimization and recipient grants apply to derivations as well as originals.

**P1 — BLOCKED, operator:** record the deployed Jev endpoint/model and customer
agreement/DPA version, account entitlement, subprocessors/region, content and
telemetry retention, deletion procedure and any agreed ZDR evidence. Repository
default: `https://api.typesafe.ai/v1/systemone`, `jev-latest`; those defaults do not
identify the deployed account or agreement. TypeSafe states that requests/responses
are not used to train Jev, but retention depends on agreements/data/legal needs;
service/usage/abuse telemetry is distinct. A no-training statement is not ZDR.
Sources checked on the review date:
[submitted data](https://help.typesafe.ai/articles/5974925716-how-is-submitted-data-used-or-retained),
[DPA, April 24 2026](https://typesafe.ai/legal/data-processing),
[privacy policy, November 19 2025](https://typesafe.ai/legal/privacy-policy).
No deployment-specific retention commitment was supplied.

**P2 — BLOCKED, operator:** record the exact System Two provider/model, CLI version,
sign-in product/account plan, governing terms, training controls, remote retention,
subprocessors/region, local session/log/cache paths and deletion procedure. Shipped
`InferenceSettingsStore` uses the signed-in Codex CLI subscription provider.
`OpenAICodexProviderConfig.ephemeral` defaults to false: a temporary working directory
does not prove session/prompt deletion outside that directory. Qualify and bound
CLI persistence before private data enters it. No inference was made in this review.
[OpenAI API data controls](https://developers.openai.com/api/docs/guides/your-data)
describe API endpoint-specific storage, with default abuse-monitoring retention up
to 30 days subject to exceptions. Those API terms do not establish subscription CLI
retention or this account's training settings. Obtain applicable product evidence;
do not promise API ZDR for a signed-in subscription path.

**P3 — BLOCKED, operator:** publish the final disclosure in setup before consent,
record affirmative agreement to the exact processor/purpose/audience, and verify
that processor terms permit the source data. Google Workspace
[Limited Use policy](https://developers.google.com/workspace/workspace-api-user-data-developer-policy)
restricts transfers, human access and general model training, including derivations.
It also requires secure handling/encryption and compliance disclosure. Public
no-training claims alone do not demonstrate the deployed transfer is permitted.

Disclosure text for operator completion (bracketed fields must be replaced):

> Alphonse reads only the accounts, resources and time windows you authorize.
> Connecting an account starts no monitoring. Selected content and task context
> may be sent to [Jev operator/region/retention] for bounded decisions and
> [System Two operator/product/region/retention/training controls] for reasoning.
> Source content is cached for at most seven days, subject to shorter provider
> limits. Findings have separate retention described below. You choose recipients;
> project membership does not grant account access. Disconnect blocks new local
> reads and deliveries. We redact dependent local views, but cannot recall an
> in-flight request, delivered/downloaded copy or remotely retained processor data.
> [Explain provider-consent removal, deletion contact and backup expiry.]

## Adopted retention, replay and recovery boundary

These are maximums. Earlier source deletion/revocation overrides them. Provider
cache headers/terms can require a shorter interval. No active-run content pin or
longer durable retention is qualified. This section supersedes draft 30-day pin
language in the original #105 contract.

- Normalized source content/excerpts: **seven days from acceptance**, no extension.
  Raw MIME is not persisted; direct attachment bytes are not a durable shared
  cache. Callers may not introduce untracked attachment/prompt caches.
- Findings: finite policy end plus **90 days**, or **365 days from initial finding**
  for an ongoing policy (deadline, not automatic renewal). Each historical version
  must independently pass current authority and its own deadline. Cache expiry or
  policy pause alone preserves authorized finding history; revoked/deleted sources
  invalidate it. Unknown source states fail closed.
- General memory, UI indexes and processing checkpoints: opaque evidence/finding
  references only, re-resolved on every use. No mailbox bodies/facts copied into
  ordinary task checkpoints or general memory. Existing CAPD/classification
  checkpoints store evidence IDs. Publication events contain coverage/run refs.
- Terminal diagnostics/input manifests/event history: **30 days**, with minimal
  active recovery references retained until terminal. Source cursor/page/processing
  metadata must expire or be invalidated; it must not restore old authority.
- Replay acceptance: **90 days from original trusted acceptance**. After that,
  recovery requires a newly authorized occurrence. Keep content-free identity
  tombstones while referenced findings/recovery require them. Never delete replay
  receipts without an independent trusted-age rejection mechanism.
- Routine exports: **no private source/finding content, credentials or opaque
  provider cursors**. Portable skill export is package-only. Private data export is
  unqualified/disabled for the pilot; it needs separate authorization, current
  source checks, audience/scope/expiry metadata and a no-recall explanation.
- Backups: encrypted, separately protected from credential keys, **30-day maximum**;
  never serve directly. Restore offline with workers/inference/delivery disabled,
  apply current source deletion and grant/connection revocation state, run cleanup
  and verify denied views before enabling access. Delete older backup sets and
  account for SQLite WAL/free pages, CLI sessions, host snapshots and remote copies.

**R1 — BLOCKED, engineering:** the inspected runtime enforces seven-day source
reads/cleanup and finding deadlines, but has no complete time-based 30-day cleanup
for source streams/pages, terminal run receipts and all event/input manifests.
A partial cleanup now removes terminal source page receipts and source cursors
whose newest known page is older than 30 days. Cursor eviction is deferred while
any run is active, and cursors without reliable page timestamps remain; expiry
records a content-free scope marker so the next bounded baseline reports a coverage
gap. This is not a complete 30-day retention guarantee. Processing checkpoints,
terminal run receipts and event/input manifests remain unqualified.
Operational retention also deletes event history older than 30 days when no
nonterminal execution depends on it and its independent replay receipt exists;
unknown/missing replay state keeps the event. Receipts remain in place. The event
receipt table rejects duplicates older than 90 days but keeps acceptance tombstones indefinitely; pruning these would
allow unseen old IDs to become new without a trusted original-age fence. Keep
those content-free receipts until such a fence is implemented. This review does
not relabel that behavior as a qualified 90-day physical deletion policy.

**R2 — BLOCKED, operator/engineering:** no encrypted deployment backup/restore or
private export pipeline was supplied for qualification. Live-row redaction does
not erase WAL, freed SQLite pages, filesystem snapshots or independent backups.
A pre-deletion backup can resurrect content if it also restores old authority.
Do not enable a real-account pilot until a synthetic restore with current durable
tombstones has demonstrated denial/redaction and the actual backup expiry is
verified. A copied database alone is not a safe restore procedure.

The offline `account_recovery.fence_restored_accounts` primitive provides a
conservative staging step: it disables all restored managed connections and
revokes all restored grants in one transaction, requiring explicit reconciliation
and preserving their IDs and credential references. Synthetic tests restore a
pre-revocation snapshot and verify that normal account binding cannot use it.
This is **not** an automatic restore detector or a complete restore procedure:
queues, external copies, credentials, backup encryption/expiry and re-consent
still require qualification. Never invoke it on the running household database;
an offline staged copy must remain isolated until all recovery checks pass.

## Credentials and disconnect qualification

Supported implementations are macOS Keychain (the explicit native keyring backend)
and explicitly configured encrypted-file AES-GCM storage on headless Linux with a
separate protected 32-byte key. No default headless backend or plaintext fallback
exists. Unqualified OS/backend combinations stay blocked. Synthetic tests inject a
macOS API and exercise a disposable encrypted backend; they do not prove access to
a deployed, unlocked Keychain or host file permissions. Follow
[secure backend setup and migration](secure-credentials.md).

Disconnect invalidates local account/grants before vault deletion or networking.
Locked vault cleanup stays deletion-pending; retry cleanup before reconnecting.
Google attempts per-token remote revocation, reporting pending on failure.
Microsoft requires provider-account consent removal; no broad session revocation is
requested. Offline local disconnect cannot assert remote revocation. Already sent
credentials/requests cannot be recalled. Reconnect preserves connection identity
and requires new grants; it does not revive old policies.

## Personal-test pilot checklist

- [x] 2026-11-02: inspect prerequisite implementations; synthetic-only validation,
  source-state and historical-retention regressions; see dated evidence.
- [x] 2026-11-02: inventory exact scope allowlist, data flows and supported backends;
  record missing deployment records as blockers rather than approvals.
- [ ] G1/M1: operator supplies exact registrations for each enabled provider and
  records personal/development exception eligibility, account types and redirects.
- [ ] P1/P2/P3: operator supplies applicable processor terms/settings, resolves
  retention/training/local-session questions and publishes completed disclosures.
- [ ] R1/R2: engineering implements the outstanding retention/replay controls;
  operator qualifies encrypted storage/backups, restore and export exclusions.
- [ ] Operator verifies supported backend on the actual host without exposing
  credentials in evidence; verifies disconnect and provider-consent removal.
- [ ] Obtain separate explicit household consent for selected accounts, recipients,
  project/window/budgets and the real-account pilot. This issue grants none.

## Public-release checklist (separate gate)

- [ ] All pilot blockers resolved with exact revision, date and restricted evidence
  locators; complete provider-independent isolation and end-to-end workflow gates.
- [ ] Separate production registrations; Google branding/scope verification and
  restricted-data security-assessment decision/annual renewal ownership recorded.
- [ ] Microsoft account classes, tenant consent restrictions, publisher domain and
  verification decision documented and tested for advertised distribution.
- [ ] Published privacy/Limited Use disclosure, processor terms, deletion/support
  process, retention enforcement and backup/restore rehearsal independently reviewed.
- [ ] Supported Desktop/headless setup, callback transport, credential recovery,
  no-secret diagnostics, upgrade/rollback and release ownership qualified.

Neither checklist becomes green because tests pass or an implementation PR merges.
