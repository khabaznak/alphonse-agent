# macOS authority XPC peer-policy component

`PeerPolicy.swift` builds two separate role-bound XPC listener policies from
provisioned Apple developer Team IDs and code-signing identifiers. It uses
Foundation's connection code-signing requirement, rather than a caller's
claimed role or a PID as proof of identity. The Desktop and worker identities
must differ. It does not expose an XPC protocol, start a service, authenticate
a household member, or enable host Bash.

Before use, the service binary, its role configuration, executable code and
stored state must be protected against the Bash user's modifications. The
signing private key must also be protected against that user: access to a key
that can sign the same identity could allow an impersonating client. A signed
generic Python interpreter is not a trustworthy worker identity if it can
execute attacker-controlled scripts. A signed client alone does not prove a
human authorized a request. Accessibility, debugging, signed-client proxying,
request scope, cancellation, actor mapping and cross-store recovery still need
independent design and tests.

Clients must also authenticate the service before sending requests or trusting
its replies; authenticating incoming clients alone does not prevent a false
service from impersonating the authority. Incoming connection callbacks must
not perform privileged effects before XPC has verified the applicable peer
requirement. A useful next fixture is a signed, two-process ping with separate
role identities, a verified service identity and wrong-role rejection, before
adding authority operations or installing a system service.
Client-side requirements must not be assumed to protect an initial outgoing
message: test whether it can reach an impostor before a rejected reply, and
do not send household data or credentials in an initial probe. A harmless
probe alone is not proof against connection-lifetime races.

## Local verification

On macOS 26.6.2 arm64, using Apple Swift 6.4, the following standalone checks
compiled and passed, with compiler cache and binaries outside the repository:

```sh
xcrun swiftc -module-cache-path /private/tmp/alphonse-swift-module-cache \
  alphonse/agent_v2/authority_service/macos/PeerPolicy.swift \
  tests/macos_authority_peer_policy.swift -o /private/tmp/alphonse-peer-policy-check
/private/tmp/alphonse-peer-policy-check

xcrun swiftc -module-cache-path /private/tmp/alphonse-swift-module-cache \
  alphonse/agent_v2/authority_service/macos/PeerPolicy.swift \
  tests/macos_authority_xpc_reject.swift -o /private/tmp/alphonse-xpc-reject-check
/private/tmp/alphonse-xpc-reject-check
```

The first check exercises policy parsing and rejects malformed or identical
role identities. The second first confirms an anonymous in-process XPC ping
works without a policy, then checks that the local ad-hoc-signed test binary
cannot send the ping to the role-guarded listener. This does not test a real
signed positive peer, cross-process identities, a protected service account,
service installation, or owner-loop integration. It is not proof of a complete
OS authorization boundary.

### Disposable cross-process fixture

On the same host, `python tests/macos_authority_xpc_multiprocess.py` built and
ad-hoc-signed separate server, Desktop and worker fixture executables under a
temporary directory, registered a uniquely named temporary per-user launchd
job, and removed and checked removal of the registration in a `finally` block.
It required unsandboxed access to the login launchd domain on this host. Both
clients reached an unrestricted ping listener, while both were rejected by the
real team-and-identifier role requirements. Two additional fixture-only
listeners used weaker identifier-only requirements: each ad-hoc client reached
its matching listener and was rejected by the other. Identifier-only signatures
are readily impersonated and must never be treated as the product boundary.

The Mac had zero valid developer code-signing identities, so there is still no
accepted developer-signed client test, separately protected user, authenticated
service reply, actual owner ingress, crash recovery or host Bash test. The
fixture sends only a harmless fixed ping. Launchd cleanup is attempted after
ordinary errors, but a force-killed test process could leave a registration;
the unique label permits manual inspection/removal if this happens.

The fixture also tested client-side service requirements across processes.
A client with an identifier-only fixture requirement received a matching reply.
When it required a different Apple-signed service identity, its reply was
rejected, **but the mismatched service recorded the first harmless ping**.
This directly demonstrates that the requirement did not protect outgoing
message contents in this setup. Do not put owner requests, claims, tokens or
credentials in the first message, and do not treat a harmless ping followed
by a later request as secure without resolving connection-lifetime races.

A self-signed fixture certificate was also generated and imported into an
ephemeral keychain, then deleted. macOS reported zero valid signing identities
in that keychain and `codesign` refused it. No system or user trust store was
changed. This is not proof that all enterprise signing configurations fail:
a provisioned and trusted signing identity may be an alternative, but needs a
separate, explicitly designed trust and key-protection workflow. The current
`anchor apple generic` production-style policy specifically requires an Apple
signing chain; substituting another anchor would need separate security review.

## Pinned local native clients

The Unix-socket verifier supports an explicit root-provisioned pair of
distinct 40-character lowercase CDHashes as an alternative to Team IDs.
Both roles must use the same identity mode; mixed policies and identical role
hashes are rejected. CDHash pins exact signed code, not its signer or path:
an identical copied executable has the same identity. Both identity modes
require a running hardened runtime, reject a process marked as debugged, and
reject known debugging, JIT, unsigned executable memory, DYLD-environment and
library-validation relaxation entitlements. Unknown/non-dictionary entitlement
representations fail closed. These checks do not prove protection from every
injection, a trusted human action, or the integrity of resources, scripts and
dependencies read by the code. Generic interpreters with writable programs and
generic forwarding deputies remain unsuitable. The legacy XPC test listener
does not enforce runtime checks and explicitly refuses pinned-CDHash policies;
it is not the production authority ingress.

`python tests/macos_authority_cdhash_probe.py` builds disposable ad-hoc signed
native clients and a verifier library without changing keychain trust. A real
kernel audit-token check accepted the matching hardened role and identical
binary copy, rejected a different role/hash, and refused a correctly pinned but
unhardened binary or one with risky entitlements. A synthetic dylib injection
was blocked or ignored for the hardened binary and successfully loaded by the
unhardened control. These are limited observations on the test host, not a
complete security proof or an authorization for installation.

## Unix-domain transport building blocks

`local_transport.py` provides a macOS Unix-stream connector that reads the
kernel-provided peer effective UID using `getpeereid` immediately after
connecting and before sending any bytes. The connector refuses an invalid or
same-user configured service UID, closes the connection on any failure, and
returns a stream only when its peer has the provisioned UID. With a dedicated
service user that the Bash user cannot assume, and a trusted UID configuration
that the Bash user cannot modify, this avoids sending the first request to a
same-user socket impostor. It authenticates a UID, not a particular program:
every process that can run as the service UID could impersonate the service.
It is not safe to point this at a normal same-user daemon or a service account
whose executable, private state, or launch configuration the Bash user can
modify. The returned socket is not a grant, and no host Bash route uses it.

`LocalPeerVerifier.swift` is a server-side building block: it obtains the
kernel's `LOCAL_PEERTOKEN` from the accepted Unix socket and checks the
corresponding running code against the preconfigured role-specific Apple
signing requirement. The token is never accepted from the caller or inferred
from a PID. A valid signed client does not prove a human request; a signed
generic interpreter that can execute attacker-supplied programs does not
provide a trustworthy worker identity. The service must also authenticate
the household actor, bind exact requests, protect its signing/provisioning,
and coordinate queue, cancellation, effects and crash recovery.

The local verification commands below exercised a real kernel peer credential
and a separate same-user process that received **zero bytes** before rejection.
One positive socket ordering test used a mocked peer UID; it did not provision
another account. The Swift verifier rejected an invalid descriptor and an
untrusted local process; there was no accepted Apple-signed client. These
tests do **not** establish a production boundary or connect the pieces to
Desktop, the authority ledger, or the owner loop. A real deployment requires
cross-process, cross-UID positive and adversarial tests on provisioned signed
clients plus protected service/user configuration.

```sh
python -m pytest tests/test_macos_authority_local_transport.py -q
xcrun swiftc -module-cache-path /private/tmp/alphonse-swift-module-cache \
  alphonse/agent_v2/authority_service/macos/PeerPolicy.swift \
  alphonse/agent_v2/authority_service/macos/LocalPeerVerifier.swift \
  tests/macos_authority_unix_peer.swift -o /private/tmp/alphonse-unix-peer-check
/private/tmp/alphonse-unix-peer-check
```

Apple documentation: [NSXPCListener code-signing requirements](https://developer.apple.com/documentation/foundation/nsxpclistener/setconnectioncodesigningrequirement(_:))
and [SMAppService](https://developer.apple.com/documentation/servicemanagement/smappservice).
