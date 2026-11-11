# One candidate clock attempt: reviewable scope

This is a development smoke attempt, not the matched five-family pilot or a
production rollout. **No live call has been made or authorized by this file.**
Preview the proposed scope without reading a key or using the network:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.clock_pilot_setup
```

The proposed attempt sends one fixed synthetic request to OpenAI Responses with
`gpt-6-luna`, reads only a local synthetic clock, and compares the result with
a separate fixed-time oracle. It allows at most ten model calls, each with an
input reservation of 65,536 and an output limit of 512 tokens. A 100-second
cooperative deadline rejects late results but cannot interrupt an in-flight
HTTP request. No household records, Gmail, credentials other than the supplied
provider key, TypeSafe, or external agent are part of this attempt.

The preview defaults to a proposed $0.25 reservation budget and reports a
$0.09325568 reservation estimate using the stated rates and all ten calls. Those
figures are **not a guaranteed maximum bill**. Its input counter uses UTF-8
request bytes plus an allowance; it is not verified to bound provider tokens,
and pricing, account settings, framing and other charges can differ. The
[published model prices](https://developers.openai.com/api/docs/models/gpt-6-luna)
must be checked against the account and service at execution time. A rejected,
failed or uncertain call is not automatically refunded by the local ledger.

The local journal contains hashes, sanitized events and result categories;
the local SQLite ledger contains reservations. They persist until the operator
deletes them. OpenAI receives the synthetic prompts and observations; its actual
API account and endpoint data-retention terms must be checked separately.
Neither the local journal nor `store=false` establishes provider deletion.

Once the user separately approves this exact provider/model, synthetic data,
retention, number of attempts and budget **including its estimation uncertainty**,
and the process isolation is independently reviewed, an operator may run:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.clock_pilot_setup \
  --execute --scope-approved --accept-estimate-uncertainty --isolation-reviewed \
  --budget-usd 0.25 --journal /private/tmp/alphonse-clock-run.jsonl \
  --ledger /private/tmp/alphonse-clock-ledger.db
```

Both output paths must be new. The command prompts for the OpenAI API key
without echoing it; it does not take the key on the command line or write it
to the journal. The flags are operator assertions, not independently verified
approval or an OS sandbox. No live action should be invoked until the actual
conditions are checked. Do not automatically rerun an interrupted attempt.
The attempt's independent oracle is not proof of agent attribution, real user
delivery or production safety, and a single candidate result cannot establish
an advantage over V3 or an external assistant.

## V3 and targeted V3 clock attempts

Each V3 variant has an equivalent one-attempt preview. It uses the same
synthetic clock, OpenAI model, ten-call and per-call limits, and estimated OpenAI reservation
as above, and additionally calls TypeSafe Jev with the requested alias
`jev-latest`:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.v3_clock_pilot_setup --arm v3
.venv/bin/python -m alphonse.agent_v2.evaluation.v3_clock_pilot_setup --arm targeted_v3
```

The owner approved a total Jev budget of $1.86; the pilot retains a proposed $0.10 per attempt, with at most 30 HTTP attempts,
including retries, and at most 65,536 payload bytes per attempt. The input
reservation estimate assumes the [published TypeSafe rate](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost)
and at most one token per payload byte. Neither assumption is verified for the
account or a hard cost bound; the ledger enforces attempt count and byte size,
not dollars or output-token limits. Jev can report an actual model different
from the requested alias. Its provider retention must be checked separately.

Only after the user separately approves both providers, the synthetic state,
retention, attempt count and the two estimated budgets—including possible
overshoot—and process isolation has been independently reviewed, an operator
may invoke one arm at a time in a new process:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.v3_clock_pilot_setup \
  --arm v3 --execute --scope-approved --accept-estimate-uncertainty --isolation-reviewed \
  --openai-budget-usd 0.25 --jev-budget-usd 0.10 \
  --journal /private/tmp/alphonse-v3-clock.jsonl \
  --openai-ledger /private/tmp/alphonse-v3-openai.db \
  --jev-ledger /private/tmp/alphonse-v3-jev.db
```

Use `--arm targeted_v3` and **three different new paths** for the other arm.
The command prompts separately, without echo, for the OpenAI and TypeSafe API
keys. It writes neither key to its local journal or ledgers. Each attempt owns
its own provider, Jev object and ledgers; the checks are not a cross-process
lease. No flag establishes approval or isolation, and the cooperative deadline
cannot interrupt an already-running OpenAI or Jev request, nor Jev retry sleep.
The V3 schema-in-prompt approach differs from provider-constrained decoding;
the V3 and candidate prompts and response mechanics also differ. They receive
the same synthetic clock source and OpenAI limits, not identical prompt bytes.
Record these differences and all failed/unsupported attempts when comparing
results. An observed result remains a development pilot: source attribution,
actual delivery, total billing and production safety are not established.
