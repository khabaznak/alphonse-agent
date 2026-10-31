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
Account references point to the existing integration ID. Credentials remain in
that integration's `secrets`; no new credential vault or per-script catalog is
created. Existing integration storage is local SQLite, not encrypted secret
storage. Protect it with the host's existing file/account protections.

The [optional-account implementation contract](optional-account-integrations.md)
extends these same bindings and explicit member choices with owned connections,
revisioned capability grants and secure credential references. That is planned
work, not current secure storage support. New optional OAuth connections must use
the qualified secure backend when implemented; no parallel binding system or
implicit project-derived access is introduced.

The current integration records have no per-member ownership. Therefore only an
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
