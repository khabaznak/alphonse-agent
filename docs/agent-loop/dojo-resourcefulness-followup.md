# Resourcefulness in owner follow-ups

The owner's boiler conversation contained a request to inspect a local device,
followed by model questions and explanations instead of the desired practical
investigation. The last two follow-up checkpoints contained no recorded Bash or
memory-search observations; earlier turns did record web searches. The running
daemon reported the simple loop and trusted local Bash. The exact historical
model prompts were not retained, so the model's reason is unknown.

The final follow-up's device address fell just outside the eight prior events
projected by the then-current code. This is a plausible contributor, not proof
of why the model asked the user about Wi-Fi. The revised loop projects up to 16
prior events under the existing 6,000-character bound. Conversation history
remains context, not independent authorization. The general decision prompt
also asks the model to make practical progress with available authorized
capabilities and search earlier conversation where appropriate before asking
the user to repeat information or perform routine work. It still asks for
genuinely missing information and requires truthful observations.

## Exploratory real-model runs

All runs used `gpt-6-luna`, the owner daemon and trusted host Bash, temporary
synthetic accounts and data, and the existing task limits. The device is a
read-only loopback service. Bash and the inference provider have the local
user's host access; the fixtures are not isolation or protected-oracle tests.
They do not exercise the installed Desktop. No comparison here establishes a
reliability rate or causal improvement.

| Scenario | Before | After |
| --- | --- | --- |
| Spanish device follow-up, address more than eight turns earlier | [Succeeded](evidence/dojo-resource-device-baseline-20261121.jsonl); address absent from first model input but visible later; two Bash processes, status read, correct reading in Bash output and answer. | [Succeeded](evidence/dojo-resource-device-after-20261121.jsonl); address visible from first model input; two Bash processes, status read, correct reading in Bash output and answer. |
| Spanish invoice follow-up, directory in earlier turn | [Succeeded](evidence/dojo-resource-document-baseline-20261121.jsonl); two Bash processes and correct total. The initial literal-match check missed the thousands separator in the answer; manual review confirms the amount. | [Succeeded](evidence/dojo-resource-document-after-20261121.jsonl); two Bash processes, correct total, unchanged final fixture. The later checker accepts a thousands separator. |
| Spanish greeting | Not run | [Answered](evidence/dojo-resource-greeting-after-20261121.jsonl) in one model call with no Bash. |
| Unavailable remote sensor | Not run | [Reported unavailable](evidence/dojo-resource-unavailable-after-20261121.jsonl) in one model call with no Bash and without inventing a reading. |

Each is one attempt. A value match and request log are not by themselves a
semantic grade; the answers were reviewed alongside observed process output.
The separate Codex provider can have activity outside the owner-tool log.
These examples cover neither unknown-port LAN discovery nor device control,
and do not prove that unrelated topic changes or conversations longer than the
bounded context will always be handled correctly.

Run a fresh case with a new journal path:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_resourcefulness_followup \
  --case device-followup --model gpt-6-luna --accept-trusted-host \
  --journal /private/tmp/resourcefulness-device.jsonl
```
