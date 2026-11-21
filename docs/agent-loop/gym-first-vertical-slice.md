# Gym first vertical slice

This is a small plumbing baseline, not a model benchmark or release qualification.
`owner_gym_smoke` sends one synthetic arithmetic request through the local IPC
protocol, owner daemon, fenced queue, owner processor and simple-loop candidate;
it observes the Desktop-facing draft, acknowledges it and checks recorded history.
The oracle expects exactly `42`. It rejects a wrong answer and leaves a missing
draft unknown. A scripted provider returns the answer, so a pass only demonstrates
that this particular path and oracle worked with a predetermined decision.

Run with permissions for local Unix sockets and a fresh journal path:

```sh
python -m alphonse.agent_v2.evaluation.owner_gym_smoke /tmp/gym-attempt.jsonl
```

The journal is created exclusively and fsyncs its directory, a start and a final record. If the
process dies between them, the lone start is an incomplete attempt, not a pass.
Runner errors get a final unknown result if the process can write one. A timeout
is unknown, not a successful answer. The fixture uses a disposable home and
database, and no real household records, integrations or Desktop UI.
The acknowledgement is a synthetic IPC client acknowledgement, not evidence a
person received a message. Temporary storage and a scripted provider are deliberate
differences from a release installation; the full runtime configuration and
installed Desktop are not validated here. The journal fingerprints the Python
source tree, but lacks a complete source/configuration/dependency manifest.

An explicit `--model MODEL` opts into the Codex provider implementation with a
40-second provider timeout. Run in a separate process: environment and `Path.home`
overrides are process-wide. Codex uses the real home environment for authentication;
its temporary working directory and read-only option are not proof of filesystem
isolation. This does not replicate all release configuration.

On 18 November one `gpt-6-luna` attempt reached the provider but returned no
model decision. The daemon reported model unavailable; the sanitized provider
category `provider_error` does not reveal the root cause or establish whether
the remote model received the request. The [unchanged receipt](evidence/gym-first-real-2026-11-18.jsonl)
records an infrastructure error and unknown task outcome. A second, explicitly
reported attempt with expanded sanitized categories also failed; its
[receipt](evidence/gym-first-real-diagnostic-2026-11-18.jsonl) records
`openai_codex_exec_failed`, which means the CLI exited unsuccessfully but does
not identify why. Both task outcomes remain unknown; neither is a model score.

Subsequent synthetic diagnostics identified a rejected structured-output schema:
the unrestricted `arguments` object was incompatible with the provider's strict
schema requirements. The Codex transport now encodes that field as a JSON string
on the wire and decodes it into an object before normal host validation. A later
[attempt](evidence/gym-first-real-schema-fixed-2026-11-18.jsonl) returned `42`
and passed the fixture checks in 4.836 seconds with one completed model call.
That is one development result on synthetic data, not release qualification or a
tool-use assessment. Earlier failed receipts remain unchanged.

The clock obstacle asks for the exact UTC observation and checks the persisted
clock-tool observation, its time within the request, the answer, the host's
displayed evidence, IPC acknowledgement and conversation history. Invoke it with
`--case clock --model MODEL` and a new journal path. One [real-model attempt](evidence/gym-clock-real-2026-11-18.jsonl)
passed in 10.439 seconds with two model calls. The prompt explicitly requests use
of the clock; this does not establish that the agent chooses it without that
instruction, nor does it test Desktop or independently verify the host clock's
accuracy. The first receipt retained the time-window check result and observation
but not the exact boundary timestamps; subsequent runs record those timestamps.
The fixture's oracle is independent of the model output but reads a trusted
in-process checkpoint, not an adversarially isolated external monitor.

The daemon's read-only `runtime_identity` IPC method reports its owner-worker
binding, worker status, process ID, package version label and fingerprints of three
loaded functions. To query the default socket:

```sh
python -c 'from alphonse.agent_v2.ipc import V2DaemonClient; print(V2DaemonClient().runtime_identity())'
```

These fingerprints are partial and interpreter-dependent; they do not establish
a complete build or prove a task ran. The source revision is unknown. A daemon
started before this change must be updated and restarted before it can answer.
