# Skill accounts, configuration and household data

Implemented for [issue #36](https://github.com/khabaznak/alphonse-agent/issues/36).
See the [standard skills contract](standard-skills-contract.md) for the package
boundary. A plain Agent Skills package needs no Alphonse metadata and no script
registration. Its instructions can name logical roles such as `calendar`,
`records` and `settings`, explain required credentials/dependencies, and describe
how scripts consume the environment below. Never include credential values,
private databases or host configuration in the package.

## Storage and authority

`SkillBindings` uses two tables in the existing `SQLiteIntegrationStore` database:
`v2_skill_bindings` holds skill-level account/data/configuration references, owner,
explicit audience, enabled state and requirements; `v2_skill_binding_choices`
holds each member's explicit mapping from `(actor, skill, role)` to a binding.
Account references point to the existing integration ID. Unmigrated legacy
credentials remain in integration `secrets`; managed OAuth credentials use #109's
secure vault references. No per-script catalog is created. Legacy secret JSON is
local SQLite, not encrypted secret storage. Protect it with the host's existing file/account protections.

The [optional-account implementation contract](optional-account-integrations.md)
extends these same bindings and explicit member choices with owned connections,
revisioned capability grants and secure credential references. Secure storage shipped in #109; member-owned grant enforcement is described below. New optional OAuth connections must use
the qualified secure backend when implemented; no parallel binding system or
implicit project-derived access is introduced.

Legacy integration records have no per-member ownership. Therefore only an
active administrator from `V2UserStore` may provision bindings, including attaching
an existing household account to a member. An owner or active administrator can
share or revoke that binding. Provisioning a different location/account requires
a new binding and explicit member selection; IDs cannot be overwritten to
silently retarget existing choices. Configuration is non-secret host state;
put every credential in integration `secrets`, never `config` or binding
`configuration`. Administrators must review this separation during setup.

Every supported use checks the authenticated actor and active owner, enabled
binding, owner/explicit sharing/admin access, installed skill, explicit member
choice, exact integration ID and enabled account, required secret keys, required
executables, and data availability/canonical location. An administrator must
also explicitly select a binding before using it. A removed grant, inactive
member/owner, disabled account, missing credential or missing path fails closed.
There is no fallback to a different user's account or another provider record.
Selection by Jev, skill instructions, project ownership or a guessed email
address cannot grant this access. Native execution still requires its existing
capability, authorization and mutation/effect checks.

## Trusted setup and native use

Management UI/IPC is a separate workstream (#38). The supported setup API is
trusted Python host code, with canonical authenticated user IDs supplied by the
host, never copied from model-provided arguments. For a running host:

```python
bindings = runtime.core.skill_bindings
bindings.configure(
    actor_id=authenticated_admin_id,
    binding_id="member-calendar", skill_id="skill:calendar",
    owner_id=member_id, kind="account", target=existing_integration_id,
    required_secrets=("access_token",), executables=("python3",),
)
bindings.select(
    actor_id=authenticated_member_id, skill_id="skill:calendar",
    name="calendar", binding_id="member-calendar",
)
```

Use `kind="data"` with an explicit absolute existing file/directory as `target`
for SQLite or other household records. Use `kind="configuration"` and a
non-secret `configuration` dictionary for settings such as timezone or calendar
ID. Data roots must be outside both installed and bundled package trees and
cannot contain those trees. Setup canonicalizes the supplied path; subsequent
use refuses a missing path or a symlink retarget. It never searches projects,
creates/moves a database, installs dependencies or repairs data automatically.

`executables` checks availability using the host PATH. Required Python modules,
versions and isolated environment setup should be documented in `compatibility`
and instructions; an executable check alone does not verify them. Runtime
profiles/readiness belong to #35. Installation or repair requires the normal
authorized mutation flow. Missing dependencies produce failure, never success.

A managed native adapter uses `bindings.use(actor_id=authenticated_actor_id,
skill_id=..., names=[...], operation=callback)`. The callback receives only the
requested roles, with `kind`, `configuration`, and either `provider`/`credentials`
or a data `path`. Only credential keys explicitly named in `required_secrets`
are delivered; an empty list delivers no credentials. Resolve per operation; never cache the payload as an authority
or save it in task state. The callback returns JSON evidence; known credentials
are redacted recursively and provider exceptions become static errors without
printed exception chains. The adapter must verify effects through its normal
read-back/completion checks.

Sharing and revocation, again from authenticated management code:

```python
bindings.set_access("member-calendar", actor_id=owner_id,
                    shared_with=(other_member_id,))
# The other member must explicitly select this accessible binding to use it.
bindings.set_access("member-calendar", actor_id=owner_id, enabled=False)
# Or disable the underlying account through the existing integration store.
```

Revocation applies to subsequent managed resolution/dispatch. It cannot recall
credentials already supplied to a running trusted process, stop provider-side
actions in flight or invalidate copies made by that process. Provider-side
credential revocation is a separate account operation.

## Script execution

Native Bash accepts `skill_id` alone for [package script execution](skill-script-execution.md),
and optional `binding_names` for managed account/data access. The skill must be
in the task's selected skill IDs; binding ACLs are checked independently. No
individual script is registered. Example native arguments:

```json
{"command":"python3 scripts/check.py","cwd":"/verified/package/root",
 "skill_id":"skill:calendar","binding_names":["calendar","records"]}
```

The child process receives JSON in `ALPHONSE_SKILL_BINDINGS`. A script reads it
without putting credentials in the command, SKILL.md, a reference, or a prompt:

```python
import json, os
bindings = json.loads(os.environ["ALPHONSE_SKILL_BINDINGS"])
account = bindings["calendar"]
token = account["credentials"]["access_token"]
records_path = bindings["records"]["path"]
# Pass token directly to the authorized client; never print/log it.
```

Only requested roles are delivered. The parent environment is not modified;
ordinary Bash dispatch strips any inherited `ALPHONSE_SKILL_BINDINGS`. Credentials
are absent from tool arguments/receipts and the ephemeral payload's repr is private.
Bash redacts known literal, JSON-escaped and Python-repr credential values from
stdout/stderr **before** truncation, including timeout and cancellation output. It returns static
errors on binding execution exceptions. The integration record's repr also omits
secrets; its normal `to_dict()` remains masked.

This is defense against accidental logging, not an information-flow guarantee.
Trusted code must not log transformed credentials, write them to files or send
them elsewhere. Arbitrary encodings/derived strings and logs produced directly
by subprocesses/providers are outside this literal-output redactor. Never expose
`prepare()` or the raw execution payload through an LLM tool, prompt, telemetry,
export or routine log. `prepare()` is internal trusted-executor access only.

## Data, lifecycle and actual limits

Bindings and choices live outside the package and are preserved on update,
export and uninstall. Household files and integration credentials are untouched.
After uninstall, resolution refuses the missing skill. Reinstalling the same
skill can reconnect retained choices only after all access/readiness checks run
again. Existing package exports contain package content only; they do not add
binding tables, live data or credentials. This does not scrub private information
that a person manually embedded in a package: such legacy content must be
separated/reviewed before export, as required by the standard contract.

A data binding grants access to a whole file/directory. Direct SQLite access
provides no row-level or per-table isolation. A connection may read all records
and, according to ordinary OS permissions, write them or attach other databases.
Use a separate managed service/OS boundary when narrower access is required.

Generic Bash and trusted native Python execute with the Alphonse OS identity.
They can bypass this API and open host files, read the integration database or
inspect available process environments. The binding resolver protects managed
access, not unrestricted filesystem/database access. A cwd, selected skill or
allowlist does not sandbox a process or isolate household members sharing the
same OS identity. Run only authorized, trusted code. This feature makes no
host-level or row-level isolation claim.

Synthetic regression coverage: `tests/test_agent_v2_skill_bindings.py` verifies
cross-user denial, explicit sharing/removal, inactive actors/owners, account and
credential revocation/missing records, missing/retargeted data, package lifecycle
preservation, missing dependencies, selected-skill checks, native/Bash redaction
and redaction before timeout/output clipping.

## Secure optional-account credentials

Account bindings preserve their explicit integration targets and ACLs when an
account adopts a secure credential reference. The trusted resolver loads the
current generation only after authorization and passes only required credential
keys to managed execution. Metadata/readiness never includes the vault payload.
A locked/missing backend fails closed with a static status; no plaintext fallback
or provider-based account substitution occurs. See [secure credential storage](secure-credentials.md)
for deployment, explicit legacy migration, rotation and offline disconnect.

## Member-owned connections and bounded grants (#110)

`AccountService` now extends the existing integration ID with versioned connection
metadata: immutable member owner/provider subject, normalized capabilities, actual
provider scopes, status, health, last success, revision and timestamps. Credential
references/generations and revocation/deletion state remain in the secure columns
introduced by #109; they are not duplicated in binding configuration. The additive
`connection_json` column defaults to `{}` for legacy integrations. Migration never
infers ownership from an address, selected skill, project or existing binding.

The supported management boundary is trusted host Python, as with #36. Construct
`AccountSession` only from authenticated channel/task identity, never from model
arguments. There is deliberately no new IPC method accepting an asserted member ID;
the existing owner-only local socket is a trusted same-OS-user boundary, not
independent household-member authentication. Desktop management is #121. Provider
adapters use this same service; connection setup creates no schedules or monitoring.

Explicit setup proceeds in this order:

1. Use `create_connection(session, connection_id=..., provider_key=...,
   provider_subject=..., display_name=...)` to reserve a new ID for the authenticated
   member before consent (revision 1, `reauth_required`). Install credentials via
   the #109 secure credential service. No new plaintext OAuth values are accepted.
2. Call `AccountService.save_connection(session, connection_id=..., expected_revision=1,
   provider_subject=..., capabilities=[...], granted_scopes=[...])` to reconcile
   ownership explicitly. For existing unowned integrations only an authenticated
   administrator may reconcile ownership (`expected_revision=0`) after secure
   migration; `reconcile_owner_user_id` explicitly names its verified member owner.
   This field cannot transfer an already owned connection. Existing subjects/owners/
   provider keys cannot be retargeted.
3. The connection owner calls `save_grant` with stable grant ID, exact connection,
   beneficiary, operations, resources, recipients, optional project, expiry and
   expected revision (`0` creates). Empty recipients permit no disclosure. Lists
   use exact identifiers; wildcards are rejected. Only this owner can issue/expand
   or revoke grants, including when a different member is household administrator.
4. Configure an account binding with `grant_id`, `capabilities` and
   `required_scopes`. An account owner can provision their own managed connection;
   legacy provisioning remains administrator-only. For existing bindings use
   `attach_grant(session, binding_id=..., grant_id=..., expected_revision=...,
   capabilities=(...), required_scopes=(...))`. This preserves all IDs and choices.
5. Each member explicitly selects their role with the existing `select` API.
   Binding sharing and the beneficiary grant both apply; neither replaces the other.

`resolve_account(session, skill_id=..., name=..., operation=..., resources=[...],
recipient_user_ids=[...], project_id=None)` rechecks the installed skill's usability,
active actor/binding owner/connection owner, ACL, exact choice, enabled connection,
status, backend health, grant beneficiary/project/expiry/revocation, operations,
resources, recipients and provider scopes. It returns an `AccountHandle` with exact
IDs, revisions and permitted request constraints, without configuration or tokens.
No choice means setup is required, even when only one account exists; adding another
account with the same provider/name never changes a selection. Projects are optional
for ad hoc reads; a project-specific grant matches only that exact project. Project
membership, Jev relevance and editable package/configuration content confer no grant.

Trusted provider adapters call `use_account(session, request={...}, operation=callback)`
per operation/page. They must enforce the handle's bounded operation/resources and
recipient constraints. Only the in-process trusted callback receives secure credentials;
results are JSON-normalized/redacted, provider errors are static, and authority is
checked before and after the callback. A handle is a snapshot, not a reusable bearer
capability. There is no transaction spanning provider I/O: revocation cannot undo a
request already issued, but denies subsequent pages and returning an in-flight result.
Policies must pass pinned `expected_binding_id`, `expected_connection_id`,
`expected_grant_id` and `expected_grant_revision`; a role change or revision mismatch
requires review rather than retargeting. Provider acquisition remains its own workstream.

Owned accounts cannot use legacy unrestricted `prepare/use` credential delivery.
Existing unmigrated and secure-but-unowned #36/#41 bindings retain their protected-host
compatibility path until explicit reconciliation. Generic Bash is still a trusted
host capability, not an OS sandbox; these checks protect managed access, not against
malicious code running as the same OS user.

Grant revisions fence updates and revocation. Managed binding ACL changes require
`expected_revision`; legacy ACL callers remain compatible. Scope narrowing is checked
on the next use. Both connection-service disconnect and direct secure-credential
disconnect fence use and revoke dependent grants. Offline deletion/revocation state
uses #109's existing behavior; no provider revocation success is claimed. Reconnecting
credentials/consent belongs to #111 and never resurrects a revoked grant. Package
update/export/uninstall cannot change grants or choices; uninstall blocks use until
an available skill is installed again. No private grant/connection state is exported.
