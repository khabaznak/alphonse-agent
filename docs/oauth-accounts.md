# Managed optional account OAuth (#111)

Implemented over the #109 credential vault and #110 member-owned connections.
Provider configuration and dependencies are opt-in. No real-account connection,
retrieval or provider writes were performed to validate this implementation.
Connecting an account registers no sensor, event subscription or schedule.

Install the `accounts-oauth` extra plus the deployment's secure credential backend.
Configure `ALPHONSE_GOOGLE_OAUTH_CLIENT_ID` and
`ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI`, or the corresponding `MICROSOFT` variables.
Both values are required for each configured provider. Use a provider registration
qualified for public desktop clients (Microsoft personal/work/school eligibility
must match the registration). There is no client-secret setting or bundled secret.
Configuration is fixed for the daemon lifetime; changes require a restart.

The callback is an operator-configured exact URL: an explicit port and path on
`http://127.0.0.1`, or registered HTTPS. A native setup surface must bind its local
callback receiver before opening the URL in the system browser, use the same exact
callback for that attempt, suppress request/query logging, and pass the URL directly
to `complete`. Do not route code/state through conversation, events or model tools.
The #121 Desktop surface owns browser launch/callback transport and trusted session
establishment. This service does not add a new listener or implicit connection flow.
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
CAS before tokens can be used. Pause/disconnect/reconnect cannot be overwritten by
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
capabilities. The initial capabilities are read-only mail and calendar events.
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
