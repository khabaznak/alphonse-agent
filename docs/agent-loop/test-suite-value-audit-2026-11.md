# Test-suite value audit — 13 November 2026

## What the count means

At the start of this audit, `.venv/bin/python3.12 -m pytest --collect-only -q` collected **3,919 cases** in 244 `test_*.py` files, representing 2,463 distinct collected test functions. Other parallel work may change those counts. Collection is not execution: optional dependencies can cause skips, and parametrized functions create several cases. The earlier result of 3,840 passed and 39 skipped was measured on a different checkpoint; it must not be presented as a result for this working tree.

This is a sample audit, not a complete coverage or duplication analysis. Counting tests does not establish which user behaviors or failure modes are covered, whether a test can detect a regression, or whether the runtime cost is justified.

## Evidence from a sample

| Sample | What it checks | Assessment |
| --- | --- | --- |
| `tests/test_heldout_memory.py`: 100 parametrized fixture cases | Each held-out case's source readback and whether a success sentence is incorrectly credited as an effect. | Corpus and oracle integrity, not 100 independent product capabilities or a live benchmark. Each case can contain unique data, so retain full corpus validation. |
| `tests/test_paired_memory.py`: 41 parametrized fixture cases | Disposable-store isolation and permitted tools for each boundary case. | Protects the test environment and data boundaries. Full validation should remain available even if a representative fast subset is introduced. |
| `tests/test_agent_v3_delegated_outcome.py`: 48 invalid-judgment combinations | Eight fields crossed with six invalid values must fail closed. | Potential matrix-consolidation candidate only after verifying common validation and preserving each field and invalid-value class. Do not remove combinations based on count alone. |
| `tests/test_synthetic_delivery_settlement.py`: 17 settlement mutations | Unauthorized or mismatched fixture effects and grants must leave settlement pending. | These represent different safety failures; keep the distinctions. Synthetic evidence does not establish real-provider settlement. |

A local targeted run of those four files passed **277 cases in 8.34 seconds** (`--durations=12`). The slowest individual call was the held-out subprocess crash/restart drill at 2.47 seconds; the next was a paired-fixture abort test at 0.32 seconds. In this sample the most numerous parameterized cases were not necessarily the most expensive ones.

After the parallel gym and legacy changes, a full run on the uncommitted working
tree with local socket access used `.venv/bin/python3.12 -m pytest -q
--durations=25`: **3,890 passed, 37 skipped in 181.60 seconds**. The slowest
call was the repeated delegation-corpus processor check at 17.26 seconds; the
next four were 2.70, 2.65, 2.57 and 2.53 seconds. These are single-run figures
on one machine, not a stable performance benchmark or proof of test value.

## Recommended next steps

1. Repeat the full-suite baseline on a committed revision and record the environment and any permission or dependency requirements. Compare several runs before making runtime-based decisions; do not infer runtime from collected counts.
2. Inventory tests by the behavior and boundary they protect: authorization and privacy; claim fencing, cancellation and uncertain effects; persisted recovery; user-facing integration; synthetic gym/oracle integrity; and implementation-specific units. A test may belong to more than one category. Identify exact duplicates and assertions that merely repeat mocked implementation details before proposing deletion.
3. Define a fast feedback selection from measured costs and representative behavior, plus a comprehensive job that retains every corpus case and safety boundary. No job or marker split exists solely because it is recommended here; establish selection rules and verify the complete suite remains routinely run before changing CI.
4. For any matrix reduction, demonstrate that removed combinations exercise the same shared validation path and retain every meaningful field, invalid-value class and safety boundary. Compare failure detection with deliberately broken behavior before and after; do not trade away fail-closed checks for a smaller count.
5. Add end-to-end acceptance evidence separately: exercise claim propagation, fenced delivery, and uncertain-effect recovery across real integration boundaries when available. Fixture-only successes and large case counts do not qualify the candidate loop for household use.

No tests were removed or moved as part of this audit.
