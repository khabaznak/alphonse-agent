# Secure optional-account credentials

Issue #109 extends the existing integration IDs and SkillBindings resolver. It
adds no provider consent, account grants, monitoring, provider writes or IPC method
accepting tokens. Trusted setup services own authentication before calling these
internal APIs. #110/#111 own member account/grant services and OAuth adapters.

## Backends and setup

Desktop macOS uses `MacOSKeychainBackend`, selecting `keyring.backends.macOS.Keyring`
directly rather than a user-configured fallback. Install the optional extra
`pip install '.[credentials-desktop]'`. The daemon must run as the same OS user
with access to the unlocked keychain. Python processes under that OS user are
within the trusted host boundary; this is not a sandbox for hostile local code.
See [keyring security considerations](https://keyring.readthedocs.io/en/stable/).
No real Keychain entry is created by the automated tests.

Headless Linux uses `pip install '.[credentials-headless]'` and explicit operator
configuration:

```
ALPHONSE_CREDENTIAL_BACKEND=encrypted-file
ALPHONSE_CREDENTIAL_DIRECTORY=/var/lib/alphonse/credentials
ALPHONSE_CREDENTIAL_KEY_FILE=/run/credentials/alphonse.service/oauth-key
```

Provision the directory as the daemon user with mode 0700. Provision a raw random
32-byte key as a separate owner-only regular file (0600 or 0400), for example with
systemd credentials. Keep it outside the ciphertext directory and its backups;
never put the key in an environment variable, package, project, prompt, repository
or shared backup. The operator must protect parent directories from replacement.
The backend rejects noncanonical paths, symlinks, group/other access, incorrect
ownership, a contained key and invalid key length. It never generates a key or
fixes permissions implicitly.

Records use [cryptography AESGCM](https://cryptography.io/en/stable/hazmat/primitives/aead/#cryptography.hazmat.primitives.ciphers.aead.AESGCM)
with a 256-bit key, random 96-bit nonce per write and opaque reference as associated
data. Connection ID and generation are inside the authenticated encrypted payload
and checked on read. A copied ciphertext cannot be used under another reference,
connection or generation. Files are created exclusively as 0600 and flushed before
metadata publication. Backups require separately protected key recovery; losing
the key requires fresh consent. Cryptographic key rotation requires operator
re-encryption and verified readback; do not overwrite a live key in place.

No backend is selected by default on headless hosts. Backend imports and access
are lazy. Missing packages, missing/locked keychains, unavailable keys and corrupt
ciphertext block only credential operations with static `CredentialError` codes;
there is no fallback to `secret_json`. Unrelated daemon/messaging setup still works.
Other deployment/backend combinations require qualification before live use.

## Storage and managed delivery

Create nonsecret integration metadata, then call `install_credentials(id, values,
expected_generation=0)` in authenticated setup. New OAuth token/client-secret
writes through ordinary `upsert(secrets=...)` are rejected. The vault returns an
opaque immutable reference after verified readback; one SQLite update installs
that reference/generation and clears `secret_json`. Metadata updates preserve the
reference and cannot retarget its provider or restore plaintext credentials.

Records, IPC projections and readiness contain only references, generation and
health/pending flags. Even `to_dict(mask_secrets=False)` has no managed token.
SkillBindings retains explicit actor/skill/role choice and access checks, then
resolves the exact integration through the vault. Only requested credential keys
reach the trusted child process; all resolved secret values feed existing literal,
JSON-escaped and nested output redaction. Nothing adds token payloads to task state,
events, checkpoints, packages or inference inputs. Native adapters must use the
managed boundary; generic Bash remains trusted host execution.

## Rotation, failure and disconnect

`install_credentials(..., expected_generation=n)` implements generation-checked
replacement. Each candidate has a new immutable reference. Backend write/readback
finishes before the database CAS. Only one concurrent writer for generation n can
win; a loser deletes its candidate and must discard its result and resolve again.
A stale refresh cannot resurrect an account after disconnect advances generation.
Never use a rotated token before its successful persistence. A database/storage
failure is an error, never evidence of successful refresh.

Provider adapters (#111) must single-flight the **provider refresh request** per
connection before consuming a rotating refresh token. Storage CAS prevents stale
persistence; it cannot undo two provider requests or recover a provider rotation
lost before persistence. Such ambiguous provider effects require reconnection,
not repeated use of an old refresh token. A failed storage write preserves the
old reference but does not prove that the provider still accepts it.

After commit, old ciphertext is deleted best-effort. Crashes between vault write
and metadata commit, or backend deletion failures after commit, may leave secure
unreferenced records. They are never selected automatically. Operators may reconcile
unreferenced ciphertext against a stopped daemon/database backup before deletion;
do not delete anything merely because a running process has not published it yet.

`disconnect_credentials(id, expected_generation=n)` first atomically disables the
account, increments generation, removes plaintext and records `revocation_pending`.
It requires neither network nor unlocked storage to stop local use. Secure deletion
is attempted next. If unavailable, `credential_deletion_pending` and a deletion-only
reference remain for `retry_credential_deletion`; no managed consumer can read it.
A completed local deletion leaves provider revocation pending. No provider request
was made, and revocation must be completed through provider account settings or a
future authorized adapter. New consent/reconnect service belongs to #111; ordinary
enable/upsert cannot revive a disconnected secure account. Already delivered/in-flight
credentials cannot be recalled; adapters must recheck authority before later calls.

## Explicit migration and rollback

Existing Telegram/Discord secrets stay unchanged. Existing unmigrated Calendar
records retain protected-host legacy resolution until explicitly opted in through
`migrate_credentials(id)`. Migration writes/verifies the secure copy, atomically
switches the existing ID to its reference, then clears plaintext in the live row.
Existing bindings, choices, audiences and skill files are preserved. Retry after a
committed switch is idempotent; failure before the switch leaves the legacy record
for explicit recovery and never silently selects another account/copy.

SQLite historical pages/WAL and old backups can still contain **pre-migration**
plaintext. Stop the daemon and apply the established database backup/checkpoint/
compaction and backup-retention procedure when retiring those copies; deleting a
live JSON field cannot erase historical media. New secure credentials never enter
SQLite plaintext. A rollback must use a binary supporting reference columns or
keep secure accounts disabled; never copy tokens back to plaintext for downgrade.

## Validation

`tests/test_secure_credentials.py` uses only disposable encrypted directories/keys
and an injected fake macOS API. It covers restart, metadata projections, narrow
managed delivery/redaction, native backend selection, missing dependencies/backend,
locked/corrupt/missing files, permissions/symlinks, connection/generation/reference
substitution, concurrent writers across store instances, disconnect during refresh,
pending deletion and explicit migration. Calendar package tests use the same secure
fixture. No automated test authorizes real accounts, provider effects or a claim
of live Keychain/provider qualification.
