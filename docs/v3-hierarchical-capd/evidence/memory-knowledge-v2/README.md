# Issue #13 context characterization, adapter v2

Captured 2026-10-17 using the unchanged synthetic corpus v1. This retains the
[frozen v1 baseline](../memory-baseline-v1/results.json) and does not revise its
expectations. Reproduce with:

```bash
python -m alphonse.agent_v2.evaluation.memory_replay \
  --corpus tests/fixtures/memory_replay/v1.json --output /tmp/memory-candidate-v2.json
```

`results.json` uses `current-v3-context-and-native-search-v2`. Across all 22 cases,
source inclusion, source retrieval, privacy findings, search states, and call
counts match v1. Estimated prompt tokens increase because of the household-memory
heading and explicit historical-message evidence labels. All answer/latency
assessments remain null offline.

The adapter still supplies an empty legacy ledger and no knowledge extractor; it
characterizes context/search labels, not automatic knowledge quality. Dedicated
`test_memory_knowledge.py` fixtures exercise the real store and pipeline separately
with stubbed extraction/review. This evidence does not pass the #11 rollout gates,
claim token reductions, or establish live answer quality or task completion.
