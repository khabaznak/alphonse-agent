# Unknown-port device Dojo

This challenge gives Alphonse the synthetic device's loopback address but not
its port or protocol. A read-only HTTP device is started on a dynamically
assigned port inside the disposable Bash container, which uses Docker
`--network none`. The prompt asks the agent to identify the device and report
its measurement, investigating only the loopback address without changing it.
No command or tool is prescribed.

In the [first real-model run](evidence/dojo-unknown-port-20261121-01.jsonl),
Alphonse selected Bash three times. His first concurrent port scan failed;
he retried with lower concurrency, found the open port, and then queried it
with an HTTP client. The Bash output contained the simulated air monitor's
47.3% humidity reading. The fixture observed two GET requests and no mutation
requests, and the answer reported the reading. The synthetic conversation
client acknowledged the response and the container cleanup was verified.
The run took about 65 seconds and used four model decisions.

This is one exploratory run in a container, not a reliability estimate or
evidence of real household LAN discovery, device control, or Desktop UI
behavior. The fixture program and request log are accessible to the container
user and are not a protected oracle. The separate model provider has its own
environment and is not accounted for by native Bash observations. The
independent grader does not infer semantic correctness; the answer and Bash
trace were reviewed manually.

For a fresh run, use a new journal path, an installed pinned image and Docker
socket:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_unknown_port_device \
  /private/tmp/dojo-unknown-port.jsonl --model gpt-6-luna \
  --image-id sha256:FULL_LOCAL_IMAGE_ID --docker-socket /absolute/path/to/docker.sock
```
