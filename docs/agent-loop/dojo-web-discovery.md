# Configured web-search discovery Dojo

This exercise asks a synthetic owner for a location's current temperature,
without naming a tool or supplying an answer. A disposable loopback service
provides a SearXNG-shaped response. The evaluator reports separately whether
general web tools were registered, enabled and advertised to the owner model,
which tools the owner selected, whether the synthetic service saw a matching
search, and the reply. Bash is separately available in a disposable container.
The fixture's weather, location and search results are invented; this does not
test a live weather provider, Internet access, the installed Desktop or host
Bash. The Codex provider has its own environment and its internal activity is
not fully observed.

The [first](evidence/dojo-web-discovery-20261121-01.jsonl) and
[second](evidence/dojo-web-discovery-20261121-02.jsonl) exploratory runs used
the real place name Alderville. In both, web tools were advertised but the
synthetic search service received no request. The replies mentioned a real
place/source rather than the synthetic service result. The owner processor was
being edited during the first run, so that run does not establish a baseline.
The second captured selected runtime fingerprints and unchanged owner source.
Because the place is real, the fixture's invented temperature could conflict
with information in the provider's separate environment; these runs are
preserved but are not a fair correctness grade against the fake temperature.

The [third run](evidence/dojo-web-discovery-20261121-03.jsonl) corrected the
fixture to use the invented place Virdale Beacon. The owner catalog advertised
the existing `native.web_search` and `native.web_fetch` tools. The model chose
the general web-search tool; independent loopback logs recorded one matching
query, and the response reported the fixture's 28 °C. No owner Bash call was
observed. The client acknowledged the response, selected loaded-code hashes
were recorded, and the owner source hash was unchanged during the run.
Response meaning was reviewed manually, and provider-side activity remains
unobserved; one run does not establish broad reliability or live web access.

Run with a fresh journal and configured model, a pinned disposable image and a
local Docker socket:

```sh
python -m alphonse.agent_v2.evaluation.dojo_web_discovery /tmp/dojo-web.jsonl \
  --model MODEL --image-id sha256:FULL_IMAGE_ID --docker-socket /absolute/path/to/docker.sock
```
