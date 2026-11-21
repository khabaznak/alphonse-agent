# Service-first socket authority Dojo

This synthetic Dojo path submits the owner goal and a preallocated message ID
to `AuthoritySocketService` through `AuthorityClient` **before** it queues the
owner request. The actual owner worker resolves the queued request binding,
then `OwnerAuthorityBridge` claims the same actor/message/goal and records each
container Bash action against the worker's reservation. The authority protocol
uses real Unix socket pairs and framing in the test process; its peer verifier
is deliberately fake. The Desktop client is synthetic. This is not a protected
service account, signed Desktop, trusted human ingress, native host Bash, or
proof of cross-store crash recovery.

The [scripted allow run](evidence/dojo-socket-ingress-scripted-20261120-01.jsonl)
created the expected disposable shopping file. The
[scripted denial run](evidence/dojo-socket-ingress-denied-20261120-01.jsonl)
rejected action start and recorded zero Bash starts and unchanged input files.
Its original journal called the missing effect a task failure; because the
test intentionally denied execution, that result does not assess the agent's
ability. Later evaluations report `unassessed_authority_denied_before_bash`
while retaining the factual missing effect.

The [real-model run](evidence/dojo-socket-ingress-real-20261120-01.jsonl)
produced the requested `/workspace/shopping-list.json`, with the expected
quantities. Independent container readback found the correct file and
unchanged input hashes; the synthetic client acknowledged the response. Five
container Bash actions were preceded by authority starts and followed by
worker-reported, unverified returns. The model also read the result back.
This is one observation, not a reliability estimate, an adversarially
protected oracle, or proof that the user actually saw the response. The
source-tree digest is an initial snapshot and other work occurred concurrently.

In a [later real-model run](evidence/dojo-socket-ingress-real-20261121-01.jsonl),
Alphonse computed the correct quantities but created and verified
`/workspace/household/shopping-list.json`, while the owner had requested
`/workspace/shopping-list.json`. The independent readback found the requested
target missing and the correct content at the other path. His reply named the
actual path but said the task was finished. This attempt failed the requested
file effect, despite the tool processes reporting success and despite earlier
command history being present in the decision context. The source files were
unchanged and the container was cleaned up. This is another single synthetic
observation, not a real protected Desktop or native host-Bash test.

Run with `--socket-authority` on `owner_dojo_shopping`. It is a separate mode
from the older `--synthetic-authority` adapter, which creates its request only
after worker pickup. Both deliberately use fake peer verification and remain
unsuitable for enabling regular host Bash.
