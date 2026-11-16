# Owner Gmail credential refresh investigation

The currently enabled owner Gmail unread-count path uses an existing,
unexpired OAuth access token and never refreshes it. An expired or near-expiry
token blocks the read. This is a practical limitation: the owner will need an
explicit reconnect or another authorized credential lifecycle until refresh is
integrated.

A disposable implementation was explored using the existing durable,
per-connection OAuth lease. It guarded refresh with the live owner claim,
rebound the credential generation after verified persistence, and retained the
lease after an uncertain provider dispatch so reconnect would be required.
Fake-provider tests passed for successful refresh, grant revocation, claim loss,
uncertain provider failure and explicit reconnect. No real provider or household
data was used.

The experiment was removed: the owner conversation host currently accepts the
Gmail tool under an explicit `read_only=True` descriptor. OAuth refresh is a
provider request and a credential-vault/database mutation, so implementing it
inside that read-only tool misclassifies a real effect. A safe restoration needs
a separately classified, claim-bound managed credential lifecycle step with
explicit uncertain-effect recovery and durable settlement. The claim check and
external provider dispatch cannot be made atomic; their remaining race must be
handled in that contract rather than presented as a pure read.
