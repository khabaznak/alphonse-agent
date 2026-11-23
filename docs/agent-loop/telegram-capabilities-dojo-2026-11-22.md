# Telegram capability and longer-task Dojo

The candidate implementation raises the per-request owner budget from five to
ten tool actions and from eight to twelve model calls. The 165-second deadline
stays below the existing 180-second claim lease; a blocking external call can
still outlast either limit. Each action is charged regardless of outcome.
Once the cap is reached, the current loop still returns an error rather than a
final summary. Existing in-flight checkpoints retain their original budgets.

The [scripted owner-worker regression](../../tests/test_owner_long_task_budget.py)
completes six tool actions and rejects an eleventh before dispatch. It checks
the limit rather than proving that a real model can plan a longer task.

The [real-model Telegram Dojo](evidence/dojo-telegram-resourceful-20261122-01.jsonl)
used a fake Telegram client and a disposable leads CSV with the actual owner
worker and trusted-host Bash. Alphonse made four Bash calls, created and
verified the expected follow-up report, left the CSV unchanged, and returned
one synthetic Telegram reply. This did not exceed the former five-action cap.

The longer six-source exercise did not establish model-backed budget coverage.
In the [first run](evidence/dojo-six-facilities-20261122-01.jsonl) Alphonse
asked for the missing organization and clarification review was unavailable.
After the fictional organization and published reports were supplied, a
[second run](evidence/dojo-six-facilities-20261122-02.jsonl) retrieved two
of six sources and stopped. A [third run](evidence/dojo-six-facilities-20261122-03.jsonl)
claimed to have searched although the owner-tool and fixture logs recorded no
search. A general prompt adjustment used in that third run was removed because
the observation did not support it. These outcomes expose unresolved
resourcefulness and truthful-reporting issues; they are not budget successes.

All provider observations are single development attempts. The Telegram
provider is fake, and the model and host Bash use the local user's permissions.
No real Telegram message or household fixture was sent by these tests.
