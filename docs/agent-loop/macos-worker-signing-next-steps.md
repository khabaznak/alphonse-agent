# macOS protected worker: signing prerequisites

This is a build and validation plan, not an installation procedure. Host Bash
remains disabled. On this Mac, `security find-identity -v -p codesigning`
reported zero valid identities both inside and outside the tool sandbox on
2026-11-21. This does not establish whether the owner has a developer account
or an identity on another machine.

## Identity needed for the current candidate path

The current Team-ID peer policy requires an Apple-anchored code signature, an
exact configured code identifier and a matching ten-character Team ID. A
valid Apple Development signing identity may be suitable for local testing;
the actual signed worker must still pass the peer verifier. A certificate
without its private key is not an identity. A generic self-signed certificate
does not satisfy this Apple-anchored policy. A separate pinned-CDHash mode
exists for ad-hoc hardened native programs, but the tested PyInstaller worker
cannot load its Python library under hardened library validation using that
approach. No supported no-identity path has been proven for this worker.

Apple documents [managing signing identities in Xcode](https://developer.apple.com/documentation/Xcode/sharing-your-teams-signing-certificates)
and [certificate types and Team IDs](https://developer.apple.com/help/account/certificates/certificates-overview).
Provisioning an account or certificate may require owner action and is outside
this repository. Never paste, export, or send a private key or keychain
password to an assistant. The `security` command below displays certificate
fingerprints and names, not private keys; its output may still identify the
account and should be treated accordingly.

```sh
security find-identity -v -p codesigning
```

Once an appropriate valid identity is available, use its 40-character SHA-1
certificate fingerprint and its matching ten-character Team ID with the
non-installing candidate builder, from the repository root:

```sh
.venv/bin/python desktop/scripts/build_protected_worker.py \
  --identity '<CERTIFICATE_FINGERPRINT>' \
  --team-id '<TEAM_ID>' \
  --output /private/tmp/alphonse-protected-worker-candidate
```

The output path must not already exist. The builder checks all discovered
Mach-O signatures and Team IDs, the main executable's hardened-runtime flag
and prohibited entitlements, and symlink containment. It has not yet passed a
positive build on this Mac. These checks are not proof that the candidate
starts, connects as the required worker, or is suitable for installation.
The current candidate build process and its output run in user-writable
storage; neither is a protected production artifact.

Protect the signing key from use by the same account that can run Alphonse's
Bash. In particular, a keychain arrangement that allows an arbitrary Bash
process to sign a replacement worker would defeat Team-ID-based role identity.
Use an independently controlled signing/provisioning process or equivalent
OS-enforced controls, and test unauthorized signing attempts. A certificate
alone does not provide that separation.

## Service and worker are separate deployment problems

The worker connects to the authority service and must pass its configured
peer-identity and hardened-runtime checks. Its executable, loaded libraries,
Python code, environment, launch arguments and update process all need
protection from the Bash user. A signed launcher importing writable Python
code is insufficient.

The authority service instead relies on launchd starting it under a dedicated
OS identity the Bash user cannot assume. Its interpreter, imports, native peer
verifier, configuration, database, launch arguments and parent directories
must be provisioned and protected independently. The current foreground
entrypoint cannot validate the code already loaded before it starts. It has no
installed, protected Python environment or positive system-domain launchd
test; signing the worker does not solve that service deployment.

## Required validation before enabling

1. Build the candidate with the selected identity and inspect its identifier,
   signatures and hardened-runtime policy. Launch it with disposable state.
2. Install neither the service nor the candidate until a protected runtime,
   launch configuration, dedicated service identity and update/rollback plan
   are concrete and reviewed. Provisioning changes the host and must use the
   appropriate OS authorization.
3. Test real Desktop and worker peers against the installed service, including
   wrong-role and impersonation attempts, grant/revoke, restart and recovery.
   Then test an ordinary owner conversation with disposable data. A passing
   builder or a certificate alone is not an end-to-end Bash result.

Apple's [library-validation documentation](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.disable-library-validation)
describes the same-Team-ID or Apple-signed library requirement. Disabling
library validation is not a substitute for this project's peer policy.
