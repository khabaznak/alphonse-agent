# Web failure and alternative source Dojo

An owner asks for a synthetic facility's latest pump-pressure reading. They ask
Alphonse to search the web and, if it is unavailable, find another source. A
disposable loopback search service always returns HTTP 503. A separate
technician note is seeded in a container household folder; the prompt identifies
the notes folder but specifies no tool, command or filename.

The grader records whether the service actually returned 503, whether owner
Bash was called *after* that failure, whether its output included both the
synthetic reading and date, and whether the fixture's final hash was unchanged.
A model that never reaches the failing service has not exercised recovery. The
output markers are observations, not a semantic grade or proof of grounding;
the answer and trace require manual review. A correct final hash does not prove
the file was never temporarily modified. The model provider has its own
environment, whose activity is not captured by owner-tool observations.

In the [real-model attempt](evidence/dojo-web-failure-recovery-20261121-01.jsonl),
the configured search received a 503, and Alphonse chose Bash afterward. The
container did not have `rg`; he adapted with `find`, located the note, and read
the synthetic pressure and date. The final answer stated that web search failed
and reported the note's 54.2 psi reading and 2026-11-14 date. The independent
check observed both note markers in the post-failure Bash output and the note's
unchanged final hash; the client acknowledged the reply and cleanup succeeded.
The trace was manually reviewed, while the automated semantic grade remains
unassessed. This single attempt used five model calls and three Bash calls.

Run one attempt with a fresh journal, a configured model, an existing pinned
container image and a local Docker socket:

```sh
python -m alphonse.agent_v2.evaluation.dojo_web_recovery /tmp/dojo-web-recovery.jsonl \
  --model MODEL --image-id sha256:FULL_LOCAL_IMAGE_ID --docker-socket /absolute/path/to/docker.sock
```

This is a synthetic service and container, not the installed Desktop or a
live web provider. Preserve failed and inconclusive attempts as well as any
successful observations; a single run does not establish general resilience.
