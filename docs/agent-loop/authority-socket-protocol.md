# Authority socket protocol slice

`authority_service/transport.py` connects the existing authority ledger to a
bounded length-prefixed JSON protocol. The service accepts prebound Unix-stream
listeners for the Desktop and worker roles. Listener selection—not any JSON
field—determines role. A service-provisioned actor and client ID are passed
to the ledger only after the supplied peer verifier approves the socket. The
claim response carries that service-side actor separately from the exact claim
object, so a worker must compare it with the active task owner.

The transport accepts one request per connection and rejects duplicate JSON
keys, oversized frames, unknown operations, extra fields and cross-role calls.
It limits concurrent connections, uses an absolute frame I/O deadline, closes
unverified peers before reading their requests and does not create or remove
socket paths. A verifier exception yields no reply. Started actions and
worker-reported returns remain unverified; an uncertain reply must not cause
blind replay. The caller owns listener lifecycle and socket permissions.

Bash grant and revoke require the caller's last observed capability generation.
The service checks it in the mutation transaction. Revocation advances the
generation even when already disabled, so a setup grant awaiting authorization
cannot silently override that revocation. A stale request or ambiguous reply
requires a fresh state read; it must not be automatically replayed.

`macos/PeerVerifierBridge.swift` exposes the existing kernel socket audit-token
and code-signature verifier to `macos/peer_verifier.py`; the bridge compiled
and rejected an untrusted local process for both roles. The companion
`macos_authority_client` uses `getpeereid` to check the service's configured
UID before sending a request. Neither mechanism provisions a protected user,
library, executable, configuration or socket, or verifies a human's intent.
The configured worker must be a trustworthy signed executable, not a generic
interpreter running user-writable code. Client verification by signature alone
does not establish that Desktop did not act as a confused deputy.

The tests use fake peer verification to exercise actual socket-to-ledger
operations, plus a separate real native verifier negative check. There is no
accepted trusted signed client, protected service UID, application installation
or production Desktop ingress in this slice. The authority ledger and the
owner queue/checkpoint/effects still live in separate stores; their recovery
protocol and cross-store atomicity are not implemented. This protocol is not a
reason to enable regular owner host Bash until those boundaries are proven.
