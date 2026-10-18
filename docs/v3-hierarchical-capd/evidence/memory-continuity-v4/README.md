# Continuity candidate v4

Synthetic, offline evidence for #15 using the unchanged v1 corpus. Reproduce with:

```sh
python -m alphonse.agent_v2.evaluation.memory_replay --corpus tests/fixtures/memory_replay/v1.json --output /tmp/continuity-v4.json
```

Compare with `results.json`. Earlier snapshots remain frozen. See the
[implementation and limits](../../11-conversation-continuity.md) for comparisons,
test evidence and remaining rollout work. No live answer-quality claim is made.
