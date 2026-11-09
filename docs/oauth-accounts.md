# Managed optional account OAuth (#111)

Implemented over the #109 credential vault and #110 member-owned connections.
Provider configuration and dependencies are opt-in. No real-account connection,
retrieval or provider writes were performed to validate this implementation.
Connecting an account registers no sensor, event subscription or schedule.

## Client provisioning and readiness (#201)

The distribution owner controls the provider registrations and release artifact.
The daemon reads the versioned public manifest
`alphonse/agent_v2/integrations/oauth_clients.json` from its installed package.
It is included in the Python wheel; the Desktop sidecar build collects package data.
A packaged sidecar must still be built and qualified for the target release.
The checked-in manifest is empty: **no Alphonse-managed registration is asserted**.
A release owner can populate its `providers` object with only `google` and/or
`microsoft` entries, each with exactly `client_id` and `redirect_uri` strings.
For example, the following identifiers are illustrative, not registered clients:

```json
{"version": 1, "providers": {
  "google": {"client_id": "example-google-client", "redirect_uri": "http://127.0.0.1:8765/google"},
  "microsoft": {"client_id": "example-microsoft-client", "redirect_uri": "http://127.0.0.1:8766/microsoft"}
}}
```

Never put client secrets, tokens or account data in this file. Validate and qualify
both actual client/callback pairs with their providers before release. The same
manifest is loaded on each daemon start; replacing a package replaces its defaults.
Retain the same registrations and callbacks across upgrades. Changing a client ID
can invalidate existing tokens or change the provider-scoped identity; migration and
reconsent for such a change are not yet qualified. Do not silently replace a client
in a release with existing connections. Back up and test an explicit transition.

For development or a personal pilot, a household administrator can use
Integrations → Optional accounts → **Configure a personal test client** and enter
only the public client ID. The saved ID and fixed loopback callback are in the
household database and survive daemon restarts and package upgrades. The action is
available only through authenticated local Desktop IPC, not conversation tools.
Use `/stop` to stop the daemon, then reopen Desktop; an already-running daemon
continues using its previous configuration. A connected account can retain the
same client only when its recorded issuance provenance matches; older accounts
without provenance and different clients require explicit reconciliation. The
Google callback is `http://127.0.0.1:8765/google`; Microsoft uses
`http://127.0.0.1:8766/microsoft`. These values are not claims that a provider has
qualified the corresponding registration.

Some Google Desktop registrations require a client secret at the token endpoint
even though Google's general Desktop documentation lists it as optional. An
administrator can enter that registration's secret in **Add a Google test client
secret**. It is stored in the configured credential vault, tied to the exact active
client ID, and used only for that client's token exchange and refresh. It is never
included in the distribution manifest, authorization URL, catalog, logs or model
tools. Never paste it into a conversation. This operator-provided value does not
establish a secret-free distribution registration. A staged restore detaches and
preserves the old vault reference separately for reconciliation.

An advanced operator may instead install the `accounts-oauth` extra and the
deployment's secure credential backend, then set
`ALPHONSE_GOOGLE_OAUTH_CLIENT_ID` and `ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI`, or
the corresponding `MICROSOFT` variables, in the daemon's environment. Either
variable selects the override for that provider, and both are required: an incomplete
override fails closed rather than falling back to a distribution client. Overrides
are distinct per provider. Environment values take precedence over saved operator
clients, which take precedence over the distribution manifest. Configuration is
fixed for the daemon lifetime; restart after changing it. A packaged household user
should receive qualified distribution configuration, not edit environment variables.
The current empty manifest means that path is not yet provisioned.

Desktop presents a general data-processing notice and asks the person launching
consent to acknowledge it. This UI acknowledgement is not a persistent consent
record or a substitute for the deployment-specific processor disclosure and pilot
readiness requirements in [#123 readiness](integrations-readiness.md).

The catalog reports missing, invalid and configured clients per provider. Configured
means local syntax and Desktop transport are acceptable, **not** that the client is
registered, approved, dependencies work or provider consent will succeed. HTTPS
callbacks remain usable by a separately arranged headless callback transport but
are unavailable to Desktop's loopback receiver. The existing account state and
health controls report connection and reconnect status independently.

Google requires a Desktop public client, the Gmail and Google Calendar APIs enabled,
and a consent audience eligible for the account: consumer accounts need an External
app; Testing requires the chosen account to be an allowed test user. Workspace admin
policy may block access. Microsoft must use a public client supporting personal
Microsoft accounts (Outlook.com/Hotmail); `/common` also supports eligible work or
school accounts when the registration and tenant policies permit them. Register the
exact supported callback using the provider's rules. The executable scope allowlist
and consent behavior are described below; the detailed API/scope, publishing, tenant,
verification and retention gates live in [#123 readiness](integrations-readiness.md).
References: [Google desktop OAuth](https://developers.google.com/identity/protocols/oauth2/native-app),
[Microsoft account types](https://learn.microsoft.com/en-us/security/zero-trust/develop/identity-supported-account-types),
and [Microsoft redirect rules](https://learn.microsoft.com/en-us/entra/identity-platform/reply-url).

### Separately authorized pilot record

No pilot has been run or authorized by issue #201. After the #123 pilot gates are
met, the operator must select an eligible test account, obtain its owner's explicit
consent to the specific read-only test, and record in restricted operator storage:
provider and registration owner; public client/configuration revision and exact
callback; account class and test-user/tenant eligibility (without publishing account
identifiers); enabled APIs and exact selected delegated scopes; consent/verification
status; deployment revision and credential backend; approved read-only operation,
retention and deletion plan; date, sanitized outcome, provider-side error code and
operator action. Record only a redacted evidence locator in the repository or issue.
A registration, synthetic exchange, or this checklist is not account consent.
Do not start a live test while #123 remains blocked.

The callback is an exact configured URL: an explicit port and path on
`http://127.0.0.1`, or registered HTTPS. A native setup surface must bind its local
callback receiver before opening the URL in the system browser, use the same exact
callback for that attempt, suppress request/query logging, and pass the URL directly
to `complete`. Do not route code/state through conversation, events or model tools.
The #121 Desktop surface opens the system browser; its authenticated IPC starts the
loopback listener before consent. No listener or connection is started implicitly.
Headless deployments need an explicitly arranged supported callback transport; do
not substitute an out-of-band code flow or disable transport checks globally.

## Trusted service contract

`V2Daemon.optional_accounts(AccountSession(...))` returns the process-local setup
service, lazily and without importing provider libraries. The session is constructed
by authenticated host code. The legacy IPC `user` payload is not authentication;
these methods are deliberately not dispatched from that field. #121 must use its
authenticated context for the setup/health projections, not deserialize an actor
supplied by a model. `health` is a redacted Python API ready for that projection.

- `begin(session, connection_id=..., provider=..., capabilities=...,
  expected_revision=..., display_name=...)` reserves an exact connection ID for a
  new account (`expected_revision=0`), or starts explicit consent/reconnect on that
  same ID with the current revision. It returns an authorization URL and five-minute
  lifetime. Old consent attempts and managed work are fenced immediately. State and
  the S256 verifier live only in bounded process memory; restart requires beginning
  again. Each callback is single-use and bound to the actor and connection revision.
- `complete(session, callback_url)` validates exact callback, state, expiry, unique
  code and provider errors before any exchange. It verifies identity through the
  fixed TLS userinfo endpoint; it never trusts an unverified ID token or guessed
  email address. Reconnect must match the stored provider subject. Only the first
  successful consent replaces the reserved pending subject. Same-label accounts
  remain separate IDs; no provider-wide fallback exists.
- `health(session, connection_id)` returns status, revision, capability/scopes,
  credential availability and deletion/revocation-pending flags. It returns no
  credential reference, token, configuration or provider subject. It performs no
  provider I/O; healthy means consent/local credential state, not fresh remote
  coverage. Use provider acquisition receipts for the latter.
- `pause` and `disconnect` require the current revision and owner. Pause fences
  new uses and in-flight results; resume is explicit consent/reconnect. Disconnect
  first disables the connection, revokes local grants and deletes credentials via
  the vault. It clears pending consent verifiers and holds no token cache. There
  are no source-content caches created by this module. Existing account authorization
  denies subsequent pages/checks; future acquisition/cache owners must honor the
  same connection/grant revision and retention contract before publishing results.
  Locked vault deletion retains a deletion-only reference and reports pending.
  `retry_deletion` retries locally; reconnect waits for that cleanup.
- `token(session, authorization=...)` is solely a trusted bounded adapter API.
  Its request is the full explicit `AccountService.authorize` request. Never expose
  its token result to IPC, skills, events, inference or a generic tool. It checks
  grants before credential access and again after refresh. Consumers must repeat
  authorization for every page and before releasing acquired results, as with
  `SkillBindings.use_account`. Tokens are stored as separate secure values so the
  existing managed binding redactor recognizes each secret independently.

A durable SQLite per-connection lease serializes refresh across service instances
and processes. Rotation is persisted by connection-revision and credential-generation
CAS before tokens can be used. Routine token rotation preserves the account
authority revision pinned by project policies; only changed consent/scopes/lifecycle
advances it. Pause/disconnect/reconnect cannot be overwritten by
an old response. Failed/indeterminate refresh or persistence requires reconnect;
known `server_error`/`temporarily_unavailable` responses remain retryable. A crashed
worker leaves a fail-closed lease; explicit reconnect fences its revision and clears
that lease instead of automatically replaying a possibly consumed refresh token.
No token/ID token is retained in process caches. Provider library DEBUG diagnostics
are disabled because they otherwise contain request/response bodies. Application
errors are static codes; callback URLs and provider descriptions are not returned.

## Provider behavior checked 2026-10-31

Google uses authorization code + PKCE, `openid` and only the selected capability
scopes, with `access_type=offline` and explicit consent/account choice. Microsoft
uses authorization code + PKCE, `openid`, `offline_access` and only selected Graph
capabilities. The initial capabilities are `mail_read` and `calendar_read`, matching the project
policy operation contract: read-only mail and calendar events.
Graph's short resource scope names are canonicalized only against our explicit
allowlist. OIDC/refresh issuance scopes are not treated as resource permissions;
missing capability scopes or refresh consent fail closed. Unused ID tokens are not
stored. Refresh responses may omit a replacement refresh token; then the existing
one is preserved, while an issued replacement always supersedes it.

Google local disconnect attempts its per-token revocation endpoint after local
invalidation. Network failure reports `revocation_pending`; token bytes are deleted,
not retained indefinitely to retry remote revocation. The user can remove access
from the provider's account settings. Microsoft provides no equivalent per-token
endpoint here: disconnect reports provider revocation pending and requires removal
of the app's consent in account settings. We do not request the broad permission or
call Graph `revokeSignInSessions` merely to disconnect one Alphonse connection.
Reconnect never restores locally revoked grants; explicit new grants are required.
No provider registration/verification or production eligibility is asserted here;
#123 owns deployment readiness before a separately authorized real-account pilot.

Primary references:

- [Google desktop OAuth](https://developers.google.com/identity/protocols/oauth2/native-app)
- [Google OAuth best practices](https://developers.google.com/identity/protocols/oauth2/resources/best-practices)
- [Microsoft authorization code flow](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow)
- [Requests-OAuthlib API](https://requests-oauthlib.readthedocs.io/en/latest/api.html)

## Disposable verification

`tests/test_oauth_lifecycle.py` covers both providers with synthetic transport,
including the real optional Requests-OAuthlib client against mocked HTTP. It covers
S256, no client secret, exact callbacks, state failures/expiry/replay/actor mismatch,
missing scopes/refresh consent, identity mismatch and multiple accounts, refresh
races across services, persisted rotation, reconnect after disconnect/crash,
pause/revoke during network work, locked backend cleanup, optional dependencies,
redacted health/errors/DEBUG logging and managed binding token redaction. Existing
secure-credential, grants, integrations, daemon, IPC and binding tests accompany it.
