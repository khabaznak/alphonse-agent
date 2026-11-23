# Known-address device discovery Dojo

This development challenge asks the owner loop to investigate a synthetic
device at a supplied loopback address and dynamically allocated port, identify
it, and read its measurement without changing it. It does not specify a tool,
protocol, endpoint, command, or response. The disposable HTTP fixture reports
an invented air monitor and a randomly selected humidity value. It is not a
commercial device or a real LAN, and supplying the port makes discovery easier
than finding an unknown service on a LAN.

The fixture records requests independently of the model's answer. The runner
uses a temporary owner, workspace, database, and daemon, with the real simple
owner loop and trusted local host Bash. The answer is recorded for manual
review; the automated checks only report endpoint access and whether the answer
contains the selected value. Neither a numeric match nor a request log alone
proves semantic correctness or causation. The model and Bash run as the local
user, so the fixture and its observer are not protected from them. The Codex
provider has its own environment; its activity is not captured by the native
Bash process counter.

## First real-model observation

The [journal](evidence/dojo-host-device-discovery-20261121-01.jsonl) records
one run using the Codex subscription model `gpt-6-luna`, with the existing
owner-loop limits. The daemon reported `simple_loop` and trusted local host
Bash. The model made two Bash decisions; two host Bash processes exited
successfully. The fixture observed a request to its status endpoint and no
mutation request. Alphonse identified the air monitor and reported the
fixture's humidity value, 45%. The Desktop client transport acknowledged the
answer. The journal does not associate each network request with a particular
process, so it does not independently establish that Bash caused the request.

This is a useful first observation, not a reliability estimate, real LAN scan,
device control test, or proof of production Desktop behavior. The host Bash
path is unrestricted; only the test target is disposable, and the prompt asks
the model to investigate only that target.

To run a fresh attempt with a new journal path:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.owner_dojo_trusted_host \
  --case host-device-discovery --model gpt-6-luna --accept-trusted-host \
  --journal /private/tmp/dojo-host-device-result.jsonl
```
