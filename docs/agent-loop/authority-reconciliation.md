# Authority and owner queue recovery observation

The experimental service stores owner requests separately from the existing
owner queue, checkpoints and effects. A crash can occur after the protected
service records a request but before the queue commits, after the queue commits
but before the client receives its receipt, or after an action starts but
before its return is recorded. No transaction spans these stores.

`authority_service.reconciliation.inspect_owner_request` is a read-only
recovery diagnostic, not a recovery executor. It first asks the authenticated
Desktop service endpoint to inspect a request, which requires the protected
owner, exact message ID and goal. It then compares that result with one local
SQLite queue snapshot and its ingress/history binding. Local data can be
modified by the shell user and is never treated as authority. A caller must
authenticate the service and client; the inspector does not do that itself.
If the caller retained the message ID but lost the service's submit response,
the inspector can perform a protected lookup by owner, message ID and exact
goal to recover the request ID without creating a new request. A `not_found`
result is only a snapshot: an earlier in-flight request could still commit.

The result distinguishes unavailable or mismatched protected state, missing
or mismatched local records, a pending local row, cancelled authority, and
an uncertain claimed/started action. An absent queue row does not prove it was
never processed: it may already have been finalized or removed. No case
automatically reenqueues, cancels, claims or replays an action, and no process
return establishes an external effect. A claim, started action, lost response,
or expired lease still requires an authorized recovery path and independent
effect evidence before any future execution.

`tests/test_authority_reconciliation.py` uses disposable databases and a
synthetic peer authenticator to cover missing queue state after service submit,
matching and cancelled requests, a started action with a missing queue row,
local tampering, protected ID/goal mismatch and failed local status. Those
tests do not establish a protected OS boundary or a complete automatic
cross-store reconciliation protocol.
