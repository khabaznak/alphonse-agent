# Offline comparison evidence: issue #218

This is a **fixture and instrumentation check**, not the decision-grade
comparison required by #218. It does not establish that the candidate is
faster, cheaper, more effective, or dependable in actual use. In particular,
none of these runs tests an autonomous model or an external agent. They do not
measure provider billing, end-to-end user latency, or delivered notifications.

## Reproduction and provenance

Run on 2026-11-10 from repository revision
`71088f59cb60d49098114358c8e2e1d72c610d6b`, with no tracked working-tree
changes before these runs. The untracked browser/worktree/output directories
were present but are not inputs to the commands. All fixture data and file
effects were synthetic and disposable. Results below are recorded here; the
commands did not save full result artifacts.

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
from alphonse.agent_v2.evaluation.simplified_loop import compare
from alphonse.agent_v2.evaluation.candidate_comparison import V3ReadFixtureAdapter, SingleLoopFixtureAdapter
from alphonse.agent_v2.evaluation.seeded_v3_resume import V3SeededResumeAdapter

fixtures = Path('tests/fixtures/single_loop_comparison')
for name in ('read_only_v2.json', 'heldout_read_only_v1.json',
             'lifecycle_v2.json', 'v3_seeded_resume_v1.json'):
    seeded = name.startswith('v3_')
    report = compare(fixtures / name,
        baseline=V3SeededResumeAdapter() if seeded else V3ReadFixtureAdapter(),
        candidate=None if seeded else SingleLoopFixtureAdapter())
    print('\n', name, report['corpus_sha256'])
    for case in report['cases']:
        print(case['case_id'])
        for arm in ('baseline', 'candidate'):
            result = case[arm]
            metrics = result.get('metrics') or {}
            print(arm, result['availability'], result['passed'],
                  result.get('status'), result.get('reason'),
                  {key: metrics.get(key) for key in
                   ('system_two_calls', 'jev_calls', 'tool_calls',
                    'questions', 'provider_input_tokens')})
PY
.venv/bin/python -m pytest tests/test_single_loop_paired_adapter.py -q
```

The test command passed: **10 passed**. The corpus digests, validated by the
harness against their `.sha256` files, were:

| Corpus | SHA-256 |
| --- | --- |
| `read_only_v2.json` | `4f5cc2a86cdf6d28a533073a92f46fcd808af7303ffcbea83daad6dba4428fa9` |
| `heldout_read_only_v1.json` | `36700adb9469d5d3a5cf3b0b160b81d402960d1e46b71993d482ff080e6ae388` |
| `lifecycle_v2.json` | `8095496004bc4fc5578df331098d5544b56d93e741df109d50fce4903683e06c` |
| `v3_seeded_resume_v1.json` | `36327a72b21f219d4a071123253830dd860f3d9fa83355c065a5908affb26e57` |

## Observations

- The normal read-only fixture passed on both sides. V3 recorded 7 System Two,
  8 Jev, and 2 tool events; the candidate recorded 2, 0, and 1 respectively.
  These are events from different scripted providers, not model usage or a
  cost comparison. Provider token usage was unknown. The held-out fixture also
  passed on both sides with the same event counts, but varies content and
  wording while sharing the development script structure.
- In the denied-access fixture both protected reads were avoided. V3 failed
  the reason oracle after the scripted provider ran out of fallback actions;
  the candidate reached an internally blocked state. This is not evidence of
  how either live model would recover or communicate with a user. Both sides
  passed the pre-execution cancellation fixture.
- In the lifecycle fixture the V3 arm was **unavailable**, not a pass. The
  candidate blocked after an injected restart with one observed source read.
  In the missing-information fixture it attempted one question, but reported
  `question_delivery_not_integrated`; the test checks that missing delivery is
  disclosed internally. Passing that check does not mean a person received a
  question or that this meets the user's dependability requirement.
- The V3 seeded-resume fixture independently observed the expected file edit
  and read-back without repeating the lookup. The candidate arm was
  **unavailable**. The checkpoint was seeded, not produced by a real crash;
  this is not a paired recovery or production safety result.

The fixtures bypass ordinary admission, use scripted judgments, permissions,
providers and synthetic source data, and do not exercise real delivery. The
harness is not a network or OS sandbox. No current result compares the targeted
V3 simplification or any external agent, and no synthetic result establishes
that an agent pursued all sensible options or raised its hand to a real user.
Those remain necessary before the architectural decision in #218.
