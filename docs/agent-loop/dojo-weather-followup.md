# Multi-turn weather follow-up Dojo

This challenge sends three ordinary owner conversations through the real owner
worker: read a synthetic weather report, ask about a second city without
repeating the topic, and then correct the location in a third turn. The
English and Spanish prompts are test variants, not language-specific
production behavior. The fixture supplies the report path but never prescribes
a tool, command, or expected answer. Its numbers are invented; this is not a
test of live weather or a commercial weather API.

The runner uses disposable synthetic owner data and injected container Bash.
It records model decisions, each acknowledged response, a final report hash,
and separate checks for literal mention of the expected city and temperature.
Those checks are diagnostic, not a semantic pass: a response could negate or
misrepresent a mentioned number. Read the complete trace and answers before
grading. Final hash equality does not prove no transient changes, and the
readback shares the container with the candidate. The provider has its own
execution environment, so the container does not confine the provider.

With an existing pinned local image and a local Docker socket, run a new
journal for each variant:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_weather_followup \
  es /tmp/dojo-weather-es.jsonl --image-id sha256:FULL_IMAGE_ID \
  --docker-socket /absolute/path/to/docker.sock --model MODEL
```

Use `en` for the English variant. The runner does not overwrite journals. A
failure or missing acknowledgement stops dependent turns, and its final
summary retains incomplete and unassessed outcomes rather than treating them
as success.

On 21 November, one [Spanish](evidence/dojo-weather-followup-es-20261121-01.jsonl)
and one [English](evidence/dojo-weather-followup-en-20261121-01.jsonl) real-model
attempt completed all three turns. The traces show the requested temperatures
for the initial place, the follow-up place, and the corrected place; source
hashes were unchanged and both disposable containers were cleaned up. Each
attempt used six model decisions and three Bash reads. The scorer's overall
result remains unassessed because it does not automatically grade response
meaning. These two development observations do not measure reliability,
real weather access, or behavior in the installed Desktop.
