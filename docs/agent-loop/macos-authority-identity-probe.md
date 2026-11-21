# macOS authority identity probes

These checks exercise local OS mechanisms without installing an Alphonse
service, changing account configuration or accessing household data. They are
not a production authorization boundary or a test of owner intent.

## Different-user service peer

On this Mac, the root-owned filesystem entry `/var/run/mDNSResponder` had a
live kernel-reported peer effective UID of `65` (`_mdnsresponder`), distinct
from the interactive user. The existing `connect_verified_service` connector
accepted that real connection when the trusted expected UID was 65, and
rejected the same connection when it was 0. Both probes closed the socket
without sending any bytes. This confirms cross-UID kernel-credential checking
against an existing macOS service; it is not a test of a protected Alphonse
service, protected configuration or service provisioning. File ownership of
the socket was not treated as the peer process identity.

This check uses an already running system service, which may vary across macOS
versions and installations. No normal automated test should depend on it or
contact the service automatically. It is recorded as an observational probe,
not a product dependency.

## Apple-signed client peer

`python tests/macos_authority_apple_peer_probe.py` compiles a temporary verifier
and starts disposable Unix sockets. It runs the existing `/usr/bin/curl` as a
separate client process and checks the socket's kernel audit token with the
same Security framework primitive as the service verifier. The test accepted
`anchor apple and identifier "com.apple.curl"` and rejected a different Apple
identifier; only a fixed empty HTTP response was sent and the temporary
verifier and sockets were removed. This requires macOS, Swift, and permission
to create local sockets.

The positive requirement is **test-only**. Any user can run the general-purpose
curl binary, so its signature does not establish an authorized human, Desktop
or worker. Apple's system signing chain differs from the product's configured
developer Team ID requirement. The test does not exercise a trusted signed
Alphonse client, prevent forwarding or process control, prove protected
authority, or enable normal host Bash. The public verifier still requires its
service-owned role policy; the generic requirement helper used by the probe is
internal to the compiled module.
