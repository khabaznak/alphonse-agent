# Protected owner authority for trusted host Bash

## Evidence

The host Bash child runs as the daemon's macOS user. In disposable fixtures,
`tests/test_owner_bash_trust_boundary.py` exercises two independent failures of
the current boundary:

* A Bash child calls the real `experimental_owner_ingress` socket and the
  server records its request as a trusted local-owner ingress. The child is
  not the Desktop and no owner interaction took place.
* A Bash child changes the lease owner and generation in the real queue
  database. A synthetic receipt matching those values then passes the real
  owner claim fence, while the original receipt is rejected.

The tests intentionally demonstrate the current limitation. A peer PID check,
a 0600 file owned by the shell user, a renamed socket, environment filtering,
or a token stored under that same user is insufficient to fix both failures.

## Smallest credible boundary

Keep general Bash as host execution under the household user's normal OS
identity: no project-only directory, network restriction, command whitelist,
or per-script registry is required by this design. Move **authority**, not
the shell, behind a separately OS-protected local authority service:

1. The service runs under an OS identity that the shell user cannot impersonate
   or modify. Its authority and claim records are writable only by that
   identity. Client processes cannot bypass it by editing a same-user SQLite
   row. The OS must protect the service's executable and configuration too.
2. The service authenticates ingress using kernel-supplied connection identity
   and a configured, verifiable Desktop client code-signing identity. On macOS
   this can be implemented with an XPC service and audited peer identity;
   accepting an arbitrary process merely because its UID matches is inadequate.
   The Desktop interface must not let a same-user command send arbitrary owner
   prompts through a generic signing/proxy interface. Accessibility automation,
   debugging and other ways to control the Desktop are separate threats that
   must be assessed; a signed process alone is not proof of a human action.
3. A verified owner request is bound to its actor, exact payload and action
   scope in protected storage. The service mediates claim generation,
   cancellation and terminal transitions transactionally. Task-scoped service
   grants do not change the shell's ordinary host access, and the Bash
   subprocess does not receive the service's authority credentials.
4. Retain process output as evidence that a process returned, not proof that
   external effects occurred. Interrupted or ambiguous effects are surfaced
   for recovery without automatic replay. Dojo checks effects independently.

This is an implementation direction, not an existing boundary. Platform code,
signed-client provisioning, service installation, data migration and recovery
remain unimplemented. Creating a system service or account would change the
host and requires a concrete installation plan and the appropriate OS approval.
Do not move, overwrite, or infer authority over existing household data during
that work. Other platforms need an equivalent separately protected mechanism;
macOS peer identity should not be treated as a portable solution.

## Small service prototype

`alphonse/agent_v2/authority_service/prototype.py` is a state-machine
prototype. It is not exposed over IPC. It
requires an injected transport authenticator and derives the actor, Desktop or
worker role, and client identity from that authenticator rather than request
fields. Its service-owned SQLite ledger binds a request to an exact input
digest, a claim to its owner and worker, and each action to an exact argument
digest and idempotency key. Multiple distinct actions can be recorded for one
request. A dispatch is recorded before the worker attempts it; restarting the
service does not make that same started action dispatchable again. Cancellation
is serialized with dispatch and reports when execution may already have started.

The prototype records a **worker-reported** process return as unverified; it
does not itself prove that a process ran, that an external effect happened, or
that a model-selected action is within the user's intended scope. A new action
key can represent another action, including one with similar contents: no
semantic duplicate detection is implied. The prototype has no terminal-success
API or automatic retry/reconciliation, and its unit tests use a fake transport
authenticator. They establish transactional state-machine behavior, not an OS
security boundary.

The owner worker now has an explicitly injected **Dojo-only** connection to
this prototype. After the existing owner queue claim is validated, the adapter
submits the exact synthetic task and obtains a prototype claim. Before each
disposable Bash dispatch it reserves and starts the exact action in that
ledger; cancellation before dispatch or a failed authority transition prevents dispatch, and a
returned process result remains unverified. The existing owner checkpoint and
effect transaction is retained. The service is called in process with fake
peer identities: the synthetic Desktop submission occurs after worker pickup,
not at the real Desktop ingress. Thus this tests a worker integration seam,
not ingress authentication, cross-process isolation, atomic cross-store
recovery, or production authority. The adapter and its storage are reachable by
the worker, so this must not be used to enable normal host Bash.
Each adapter dispatch derives its action key from the owner checkpoint's
already-persisted tool reservation. A retry with that reservation maps to the
same ledger action and cannot restart it or change its arguments. A distinct
reservation is a distinct action. This does not provide end-to-end exactly-once
execution: recovering a claim across process restarts, reconciling the two
stores after crashes, and deciding when a new reservation is appropriate are
not implemented. The live host Bash path is not connected to this adapter.

Before integration, implement and test the actual OS-protected service account,
store permissions and authenticated transports for **both** Desktop and the
worker. A generic signed Python interpreter is not by itself a worker identity,
and a signed Desktop must not act as a shell-callable forwarding deputy. The
existing owner runtime commits the queue, claim, checkpoint, local effects and
conversation records together in one SQLite transaction. Moving only a claim
row to another database would break that atomicity. Either the protected
service must own the necessary coupled transitions, or a durable cross-store
intent and reconciliation protocol must be designed and tested, including
crashes on both sides. No migration of household data is performed by this
prototype.

## Acceptance checks

Run in an isolated multi-identity fixture before any installation: an approved
Desktop-origin request succeeds; the Bash child cannot submit a second owner
ingress, rewrite a claim, alter a grant, or read protected signing material;
revocation and cancellation take effect before mutation; restarting mid-action
does not replay it; and the genuine owner can still use Bash with ordinary host
files and network. Test unsigned, wrong-signed and reused-PID clients as well
as the positive path. An authenticated client is necessary but does not by
itself prove that a particular request came from a person or was authorized.

## Owner host path and running daemon status

The existing host Bash adapter has a **manual, internal test opt-in**, not a
household setting. `tests/test_owner_host_bash_conversation.py` exercises a
synthetic owner through the actual local IPC, owner worker, native host Bash,
file read-back and client acknowledgement, with a scripted model and
disposable files. This proves the plumbing works in that fixture; it does not
prove model behavior, authenticated Desktop origin, same-user isolation, or
that host Bash is enabled in the installed daemon.

The read-only `runtime_identity` IPC result includes `owner_host_bash_mode` in
builds with this change: `disabled`, `experimental_unprotected`, or
`unverified`. It reflects the selected owner's loaded processor flag, not a
guarantee that a request can be dispatched, the worker is running, or the OS
authority boundary is sound. A stopped or differently configured daemon must
be assessed using `loop_status` as well. The disposable Dojo's container Bash
does not change this field. Earlier daemon builds may not report the field.

## Experimental owner-worker service bridge

The actual owner worker can now attach an explicitly injected
`OwnerAuthorityBridge` to its already-opt-in native host-Bash path. It claims
a **pre-existing** service request with the exact task goal, verifies that the
service-reported actor matches the task actor, and records an exact Bash action
before dispatch using the owner's durable tool-reservation identity. The
bridge checks the service claim during execution and records a worker-reported
process return as unverified. The worker cannot submit a new Desktop request
through this bridge. An unavailable service, missing request, mismatched actor,
lost start response, or revoked claim blocks or interrupts work; it is not
silently retried.

`tests/test_owner_authority_bridge.py` exercises the real owner worker and
native host Bash against disposable files and a socket-backed ledger with
synthetic service identities and verifier. Its binding callback is an injected
fixture, **not** authenticated Desktop ingress. Tests cover success, refusal
before dispatch, cancellation during a process, actor mismatch, and a lost
start reply. There is no configured production binding or service, and this
hook does not enable host Bash. An experimental opt-in without this bridge
remains unprotected, as it was before.

The owner checkpoint and service ledger are separate transactions. The service
may record an action as started before the owner checkpoint marks dispatch, and
a lost reply leaves the action uncertain even if no process was launched.
The current code refuses automatic replay but does not reconcile the stores or
recover service claims after restart. A claim check and process start cannot be
made atomic across the service and operating system; a cancellation may race
with the process and any effects already performed cannot be undone. Polling
for cancellation is not proof that no later side effect occurred. Do not treat
this test seam as authorization for real household host Bash.
