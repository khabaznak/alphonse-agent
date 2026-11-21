# macOS authority deployment readiness

For the concrete signing prerequisite and the distinction between worker and
service provisioning, see [macOS protected worker: signing prerequisites](macos-worker-signing-next-steps.md).

This document records what must be made concrete before the experimental
authority service can support host Bash in ordinary owner conversations. It
is not an installer, a claim that the service is protected, or a deployment
instruction to run against a household machine.

## Existing build and identity gaps

The Tauri host's bundle identifier is `local.alphonse.desktop`. Debug builds
start `.venv/bin/python -m alphonse.agent_v2.daemon` or another `python3` from
the project directory. Release builds package the daemon using PyInstaller
`--onefile`; `desktop/src-tauri/tauri.conf.json` does not itself configure a
signing identity or hardened-runtime policy. Neither an ad-hoc signed generic
interpreter executing a writable project nor an unverified packaged sidecar
is a trustworthy role identity. An independently signed launcher is not enough
if it imports code or reads configuration that the Bash user can replace.

The Unix service also supports distinct root-provisioned exact CDHashes for
ad-hoc signed hardened native clients. This is a local code-pinning option,
not proof of a developer identity or human intent. Native checks reject an
unhardened/debugged process and several unsafe entitlements. An identical
binary copy still has the same identity, so the executable's behavior, all
resources and libraries it loads, generic forwarding interfaces, UI origin,
and installation/update chain must be secured separately. The development
Python worker and unverified release sidecar are not thereby trustworthy.

On this machine the current keychain search list reported zero valid code-signing
identities. That is an observation of that search list, not
a claim about the user's developer-account status or all possible deployment
methods. The Team-ID authority policy requires an Apple signing chain, a
configured developer Team ID and a code identifier for each distinct role.
Accepted Desktop and worker builds, their signing policy, the runtime code
they execute, and protection of signing credentials all remain to be defined
and verified. Installing a developer certificate without protecting its
signing key from the Bash user would not prevent impersonation.

The local hardened PyInstaller probe also found that ad-hoc signing an onedir
worker executable is insufficient: macOS rejected the bundled Python dynamic
library under library validation. This is separate from protecting the
installed directory and does not justify disabling library validation. A search
on this host found apparent Homebrew `libpython3.11.a` and `libpython3.14.a`
files, but both are symbolic links to dynamically linked framework binaries,
not static archives (`file -L` reports Mach-O shared libraries). No genuine
static `libpython*.a` was found in the searched Homebrew, `/usr/local`,
`/Library/Frameworks`, or pyenv locations. A static-linking alternative has
therefore not been tested here; even with a genuine archive, Python extension
modules and dependencies would still need to pass hardened library validation,
and imported code and resources would need a protected installation.

`desktop/scripts/build_protected_worker.py` is a non-installing candidate
builder for testing that approach when a suitable local signing identity is
available. It requires the certificate's SHA-1 fingerprint and expected Team
ID, creates an isolated PyInstaller onedir build, checks bundled Mach-O
signatures and Team IDs, and checks the main executable's hardened-runtime flag
and prohibited entitlements. Its output remains user-writable build material:
it is not a protected installation, a successful daemon launch, or an approved
peer. This machine currently has no valid signing identity, so the positive
build and library-loading behavior have not been verified.

The Desktop currently exposes a generic Tauri `daemon_request(method, params)`
command and talks to a same-user Unix socket. Signing that application alone
does not prove that a person initiated a particular request: the protected
ingress must not be a generic signing or forwarding proxy, and automation,
debugging, injected content and WebView calls require separate assessment.
No current authority endpoint proves that the verified client carried an
authorized human request.

## Provisioning contract to implement and test

1. Provision a dedicated authority service OS identity that the Bash user's
   identity cannot assume. Protect the service executable, all code and
   libraries it loads, its role/actor configuration, its database and the
   trusted client-side service-UID configuration against that user. Check
   parent directories, symlinks, ACLs, group access and update/rollback paths;
   file mode checks on the leaf alone are insufficient.
2. Install a service in an OS-managed system domain, configured to run as
   that identity. The local `launchd.plist(5)` manual states `UserName` applies
   to system-domain services, not user agents. For socket activation,
   `launch_activate_socket(3)` can supply launchd-owned listeners; the current
   Python transport accepts already-bound Unix listeners. An experimental
   launchd check-in adapter is implemented, but no service is installed and
   no system-domain positive test has been performed. Any plist,
   executable paths and environment must be generated from verified protected
   provisioning, not from a writable household preference or model output.
3. Bind the Desktop and worker to distinct service-owned role requirements.
   The running worker must use immutable, protected executable code; verify
   which actual process establishes the connection rather than assuming the
   Tauri launcher and packaged Python sidecar have the same identity. Ensure
   accepted clients cannot invoke a general forwarding deputy or supply the
   actor, role, peer identity or authority grant as request data.
4. Connect Desktop ingress, the protected ledger and the owner queue through
   a durable reconciliation protocol. Preserve source actor and exact payload,
   handle a crash at every inter-store transition, and do not replay a started
   Bash action. A service-authorized request cannot be inferred merely from a
   matching string in the existing same-user queue.
5. Verify end to end with a separately protected service UID and real signed
   positive and wrong-role clients. Attempt same-user shell impersonation and
   record that a rejected client cannot submit, rewrite a claim or receive
   protected request data. Exercise interruption, cancellation, changed actor,
   restart and effect readback with disposable data. A protected service UID
   proves the peer identity only while no untrusted program can execute under
   that UID or replace the service; it does not itself prove human intent.

The local probes in `macos/README.md` and
`macos-authority-identity-probe.md` show useful positive and negative OS
mechanics, not the foregoing complete system. Do not enable host Bash based
on those probes or on a prototype deployment alone.

## Experimental foreground entrypoint

`authority_service/macos/service_entrypoint.py` is a foreground building block,
not an installer or a runnable deployment package. It requires an isolated
Python process with the same real and effective UID, root-owned configuration
and verifier library through protected, non-symlink, ACL-free paths, and an
existing private ledger owned by the configured service UID. It refuses to
run as root, as the configured household UID, or under a different UID than
the protected configuration specifies. It accepts two already-bound socket
descriptors and checks their protected filesystem names and native Darwin
listener state; it does not create, chmod or unlink their paths. Its native
verifier checks incoming role identity. Missing configuration, unavailable
kernel checks, invalid UID, paths or listener descriptors fail closed.

The entrypoint cannot validate the integrity of its own Python interpreter,
already-imported code or launcher before that code runs. Provisioning must
protect those dependencies and launch configuration externally; `-I` alone
does not prove that. It also does not install a service, provision users,
authenticate a human request, enable Bash, or reconcile the service ledger and
owner database. There is no production signed-client positive test yet.

macOS returned `ENOPROTOOPT` for `getsockopt(SO_ACCEPTCONN)` on real
AF_UNIX listeners on the test host. The protected native verifier instead uses
`proc_pidfdinfo` for the current process's fd to check the kernel listener
option without accepting a pending connection or changing its backlog. Tests
cover a bound non-listener, a listener, an already queued client and a closed
fd. A missing or failing listener verifier prevents the server loop starting.

## Launchd preparation (not installed)

The service entrypoint accepts `--launchd`, which checks in the fixed
`Desktop` and `Worker` launchd sockets with `launch_activate_socket`. Each
role must yield exactly one descriptor; partial failures close acquired
descriptors, and the entrypoint still checks protected socket paths and the
native listener state. The explicit `--desktop-fd` and `--worker-fd` form
remains available for controlled tests and cannot be combined with
`--launchd`. The activation tests exercise failure outside a launchd job and
mocked ABI/cleanup cases; they do not establish successful system-domain
launchd activation.

`launchd_plan.build_launchd_plist` produces an in-memory plist with the fixed
role names, service UID, group-connectable sockets and an isolated Python
entrypoint. It does not inspect or install files, create accounts, select an
identity, or establish that any path is actually protected. Group access only
permits a connection attempt; the native peer verifier still checks role
identity. The plan must be checked against the provisioned account, protected
paths, Desktop and worker artifacts, and launchd behavior before installation.
