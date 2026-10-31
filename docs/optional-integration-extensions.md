# Optional integration extensions: scope decision (#126)

Decision date: 2026-10-31. Status: **accepted; defer all four extensions** for
Optional Integrations & Project Proactivity (milestone #14). This is a scope
and acceptance contract, not implementation of transports or monitoring.
Prerequisite [#105](https://github.com/khabaznak/alphonse-agent/issues/105) is complete.
Inspected main revision: `8d18b63672ce1f1200fa8a8b37b1f91106e86a1d`.

The [account integration contract](optional-account-integrations.md) remains the
authority for accounts, grants, timing, evidence and delivery. This decision
resolves its optional-extension question without changing those contracts or
claiming completion of its indexed runtime work.

## Evaluate local scheduled operation first

The stated current-user cases are a Monday school-trimester review and finite
household/bill reviews. Neither requires sub-schedule latency or a continuously
available machine. There is no documented current-user requirement for immediate
mail arrival alerts, an independently deployed collector, or an MCP-only provider.
This is a decision from the stated requirements, not evidence from a live-account
pilot or a claim that every future household need is met.

Use the existing local scheduled worker, typed project-check dispatch (#113),
bounded acquisition (#114), native managed adapters, event store, inbound queue,
V3 and outbox. These account-check components still require their own delivery
and qualification; deferring extensions does not declare them shipped. A connection
or installed skill starts no monitoring. Ordinary authorized one-off reads remain
project-optional. Projects authorize proactive goals, windows and recipients;
account grants and explicit binding choices independently limit access.

Sleeping/offline hosts provide no coverage while unavailable. Show stale coverage
and missed intervals, then apply the policy's next-allowed-window default; late
catch-up needs explicit bounded authorization. Do not create a cloud webhook,
remote relay, always-on service, or replacement scheduler to hide this limit.
Qualify the local path first through #124/#125 and release rehearsal #127.

## Decisions and accountable delivery roles

Each role below is assigned to the maintainer implementing the named issue;
these are rollout responsibilities, not new services or inferred user consent.
No extension is selected for implementation by #126, so no extension code or
blocking implementation follow-up is required to close this decision issue.
Reopening requires a linked delivery issue approved for the concrete scope;
that issue must deliver code and pass the gates below before activation.

- **Continuous polling — defer.** Scheduled bounded reads satisfy the stated
  cadence with less acquisition and no off-window observation. Revisit only
  after a household member requests a concrete freshness target the local
  schedule cannot satisfy, and budgets, windows and host availability are agreed.
  The #112 policy and #113/#114 acquisition maintainers own any future design;
  #124/#125 own qualification and #127 owns opt-in/rollback evidence.
- **Provider push — defer.** Arrival hints add ingress, subscription credentials,
  renewal and replay failure modes without improving a Monday-only retrieval
  policy. Revisit after measured scheduled operation misses an explicitly
  requested latency/cost target, and the provider's validated notification path
  and local reachability are established. #111/#117–#120 provider maintainers own
  ingress/subscriptions with #106 event-scope review; #123 owns provider/processor
  readiness and #127 owns rollout. No implicit publicly hosted receiver.
- **Independent restricted workers — defer.** Daemon-controlled adapters already
  share the managed publication boundary; no stated collector requires separate
  deployment or credentials. Revisit for a specific deployment/isolation need
  that cannot use the in-process adapter. The #106 event-boundary maintainer owns
  the restricted endpoint and credential lifecycle with #110 account authority;
  #125 owns household isolation qualification and #127 owns rollout. Existing
  trusted local sensor workers keep their shipped contract.
- **MCP transport — defer.** Current native adapters and the family-calendar
  package provide the intended integration route; no requirement establishes a
  transport gap. Revisit for a named MCP-only capability or measured maintenance
  benefit, with a reviewed server, credential custody and conformance results.
  The relevant #117–#120 adapter maintainer owns the transport comparison;
  #110 reviews authority, #123 reviews any off-host processor, #125 qualifies
  equivalence and #127 controls opt-in/rollback. Do not introduce a generic MCP
  runtime merely to package a skill.

Hosted availability and a general third-party plugin marketplace remain **outside
milestone #14**. Adding either requires separate explicit user approval and its own
scope, owners and delivery gates; requesting push or MCP alone is not that approval.
A future extension may remain entirely local; if its design requires hosting it
must stop at this scope boundary before deployment.

## Future push and worker contract (unimplemented reopening gates)

No push or independent-worker interface is adopted now. The requirements here
make a later proposal reviewable; they are not descriptions of existing APIs.
The owner-only IPC socket is trusted same-user host access, not per-worker
identity. `publish_managed_event` is currently in-process, absent from IPC.
Never give an external worker the general daemon socket as its limited credential.

1. Bind each subscription and worker credential server-side to authenticated owner,
   exact connection/resources, permitted event types/versions and current grants.
   Pin project/policy revision where relevant. Register through an authenticated
   management service; payload owner/account/run fields confer no authority.
   Rotation, revocation and expiry must fence pending requests.
2. Validate notifications before queueing: verify provider signatures or documented
   authenticated verification tokens as appropriate, subscription/tenant/resource
   binding, audience, expiry and bounded timestamp freshness. Reject unknown keys,
   versions, failed handshakes and malformed/oversized payloads. Define per-provider
   verification and rotation tests from current official documentation before
   implementation. A caller-provided worker ID or source IP alone is insufficient.
   Never treat a signature as proof that message content or claimed facts are true.
3. Accept only bounded content-free notification hints and provider change IDs,
   not mail bodies, tokens, prompts or instructions. Authenticated receipt grants
   no data access and starts no CAPD analysis. If a provider cannot avoid content
   in a notification, reject that mode or obtain separately reviewed explicit
   intake authorization; do not bypass a retrieval window by calling it a hint.
4. Coalesce hints until the next authorized occurrence. Before each retrieval,
   page, attachment, retry, analysis and delivery, resolve current binding/grants,
   lifecycle, revisions, window and cancellation. Out-of-window hints may record
   bounded operational receipts only; they may not read or analyze source content.
   Two projects need independent authorized runs, not shared inferred permission.
5. Persist subscription ID/resource scope, expiry, credential generation and
   renewal attempts separately from source content. Renew only while enabled
   and currently authorized; maintenance must fetch no content. Revocation or
   project end stops renewal and attempts unsubscribe. Offline/failed renewal
   surfaces degraded coverage and a bounded resync in the next allowed window,
   never an apparently complete no-match. Report residual provider subscriptions
   when unsubscribe fails; local revocation must deny their hints immediately.
6. Persist deduplication before acknowledgement, with a trusted acceptance time,
   subscription/credential generation and provider notification identity. Reused
   identities with different scope are rejected. Bound accepted replay age to the
   event contract's 90-day horizon; validation timestamps cannot reset it. Preserve
   receipt/tombstone protection after payload pruning; do not delete legacy sensor
   identities or promise eternal exactly-once delivery. Keep source-version, run,
   publication, finding and outbox deduplication distinct. Reordered or missing
   hints require bounded cursor reconciliation with visible gaps.
7. A future restricted worker endpoint permits publication of allowed hints only,
   with a daemon-issued revocable, expiring worker credential. It offers no worker/
   schema/automation registration, history/catalog reads, source expansion, grant
   management, queue submission or general daemon methods. Enforce schema, size,
   rate and backlog limits before persistence; bind scope from trusted registration.
   Only the daemon opens authorized check runs and retrieves source content.
   Validate loopback/local transport peer credentials or an explicitly designed
   authenticated remote transport; same-user malicious host processes are outside
   the local isolation guarantee. No endpoint or network listener is added now.

## MCP equivalence contract (unimplemented reopening gates)

MCP would be an adapter transport below the shared account service. It must not
provide a second account selector, scheduler, event bus, grant store or delivery
channel. Skills and plugins package workflows/dependencies; installation, tool
advertising, server prompts and Jev relevance are never authority boundaries.

For any proposed server, compare native and MCP execution against the same
synthetic fixtures and assert the following before opting in:

- **Account and grants:** resolve the exact existing SkillBinding/account choice;
  check actor/owner, operation/resource scope, provider consent, active grant and
  recipient intersection. A server default account or advertised tool cannot
  replace selection. Missing or ambiguous context denies before a server call.
  Constrain credentials/server access to the reviewed scope; an unrestricted
  remote mailbox proxy fails conformance even if its UI looks read-only.
- **Timing and lifecycle:** all calls, pagination, attachments and retries use the
  policy occurrence, half-open retrieval window, revision, deadline and durable
  budgets. One-off queries use separate task authorization. Server background
  prefetch/subscriptions cannot evade local windows or introduce observation.
- **Cancellation:** pause, revoke, end, cancel or stale revisions stop subsequent
  calls, expansion and delivery. Propagate cancellation/deadlines, reject late
  replies and fence checkpoint writes. Record any in-flight read already completed;
  transport cancellation cannot promise to recall bytes or external effects.
- **Provenance:** retain exact connection, provider resource/version, timestamps,
  coverage, evidence IDs, authorized audience and retention/deletion behavior.
  Server text/tool results are untrusted data. A prose summary without auditable
  source identity and coverage fails; it cannot become proof of complete acquisition.
- **Errors and recovery:** map denial, expired credentials/cursors, throttling,
  timeout, unsupported capability/version, server failure and partial pages to
  sanitized typed states. Never convert transport failure to empty success or
  replay an ambiguous mutation. Preserve acquisition/analysis checkpoints and
  stable run/publication IDs through disconnect/restart; no server tokens or
  private response bodies in logs, policy files or inference telemetry.

Discovery must expose only reviewed managed capabilities; it cannot install
arbitrary advertised tools into curation or bypass native mutation guards. Any
remote server adds a credential/data processor requiring #123 review before live
use. Existing explicitly authorized family-calendar writes retain their separate
write authorization/readback contract; project checks remain read-only.

## Concrete acceptance scenarios and rollout gates

These are future extension tests, **not tests claimed to pass in #126**. Use
fake adapters/clocks/servers and disposable stores, with no real accounts/writes.
The named maintainers above must attach exact revisions and results to a future
linked implementation PR, including baseline failures and disabled-startup tests.

- **Local first (#113/#124/#127):** a Monday 08:00–09:00 policy receives a Tuesday
  hint and an offline Monday occurrence. Assert zero off-window reads/analysis,
  a visible gap and one next-allowed-window run. Connection/install alone creates
  no schedule; feature-disabled startup preserves sensors, reminders and channels.
- **Continuous polling:** two owners/two same-provider accounts, concurrent policy
  ticks, pagination at 09:00, pause/revoke during a read and restart after cursor
  persistence. Assert bounded concurrency, stable run identity, no off-window
  continuation, no cross-account evidence and truthful partial/blocked states.
- **Push:** invalid signature/token, wrong audience/tenant/resource, expired key,
  replay after payload pruning, reordered hints and a duplicate across restart.
  Assert rejection before dispatch, at most one logical retrieval per occurrence,
  no content in hints/logs and no hint-based grant. Renewal failure/host sleep
  produces degraded coverage; revocation fences renewal and queued hints.
- **Restricted worker:** credential A publishes scope B, attempts general IPC
  methods, spoofs owner/run, uses expired/rotated credentials, exceeds rate/size
  limits and replays after restart. All unauthorized requests deny before any
  private lookup/publication; valid hints still require daemon-side retrieval
  authorization. Legacy sensors continue without gaining private-event powers.
- **MCP parity:** run the native adapter conformance cases through a fake MCP
  server returning wrong-account data, malicious instructions, incomplete source
  IDs, late responses, partial pages and typed/transient errors. Assert exact
  account/grant/window/cancellation/provenance/error equivalence above, no provider
  writes, no new tools from unreviewed discovery and no false no-match/completion.
- **Rollback:** opt-in by concrete reviewed policy only after conformance and
  provider/processor prerequisites. Disable the extension, fence outstanding
  credentials/subscriptions/jobs, retain necessary receipts/history and show
  unresolved coverage. Scheduled local operation resumes only under its existing
  authorization; no widened fallback. #127 records restart and rollback evidence.

For #126 itself, verify the published decision, links and current local event,
recovery, scope, evidence and scheduler regression fixtures. Passing these fixtures
supports preservation of the local boundary; it does not qualify any deferred
transport or declare the broader account milestone complete.
