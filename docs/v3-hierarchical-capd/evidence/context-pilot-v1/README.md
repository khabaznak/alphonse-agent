# Five everyday context examples — 2026-10-18

The question: can Alphonse receive more focused context while still finding the
facts and instructions needed to do the right thing?

This small live probe uses the configured `jev-latest` and `openai_codex` /
`gpt-6-luna` (`codex-cli 0.156.1`). It exercises the product context assembler,
registered native memory search/source expansion, and model-selected fixture actions.
It uses invented history, sensors, and a minimal disposable artifact. Household
history and real integrations are not loaded. Only provider configuration/authentication
comes from the local installation, read without changing settings.

This is a decision probe, not the complete daemon/CAPD execution path. Its short
common action prompt, directly supplied tool palette, bounded action loop and simple
outcome checks do not establish phase authorization, full answer quality, latency
or rollout readiness. Review here is by the coding agent, not an independent human.

## What happened

- **“What is the study temperature now?”** Both paths read the fixture sensor and
  answered **24 °C**. Both initially included yesterday's **31 °C** unnecessarily.
  After the fix, the new path excluded that old observation and still answered
  from the fresh reading.
- **“Why did the thermostat reading fail yesterday?”** The diagnostic was placed
  outside the recent conversation window. The legacy path composed a memory
  search; the new path retrieved it during context assembly. Both explained that
  the device was offline and no temperature was returned. The corrected candidate
  also stated that the history did not explain why the device was offline.
- **“Read study humidity,” followed by “Use the guest room instead.”** The legacy
  run read the study sensor and answered **61%**, despite receiving the correction.
  The corrected candidate read only the guest-room sensor and answered **47%**.
  The original candidate attempt had a provider-call failure, so it supplies no
  semantic result for this example.
- **“Recall last month's request to unlock the door.”** Both paths recalled the
  delivery-specific request without attempting an unlock. An unlock operation was
  present in the fixture palette, but the runner would reject it without current
  authorization. The candidate's successful follow-up made three memory searches;
  that overhead remains visible and is not evidence of efficient execution.
- **“Use the fixture sensor skill.”** Long instructions put the necessary command
  beyond the legacy guidance cap. The legacy model recovered by reading the fixture
  skill file, then ran the artifact. The new projection included the complete
  operating instructions immediately. A separate successful candidate run invoked
  the artifact through `native.bash` and answered **23 °C**. No artifact ID became
  an independently exposed tool.

## The product defect we fixed

The new path filtered retrieved hits for relevance but still inserted recent
conversation exchanges unconditionally. That is why the old temperature remained.

Optional continuity now receives bounded Jev relevance judgments too. Selection
keeps an entire exchange, preserving its antecedent; ambiguous/unavailable judgments
keep the exchange. Required task steering remains reserved separately and cannot be
removed by relevance filtering. Regression checks cover stale-history exclusion,
required steering and ambiguous whole-exchange preservation.

## Evidence, including failures

- [Discovery run](discovery.json): five examples in both arms, before the history
  filter. Eight arm-runs completed; the candidate steering and skill arm-runs had
  provider-call failures before an action/answer. Their original error capture
  retained only `RuntimeError`, so the underlying cause is unknown.
- [After the history filter](after-history-filter.json): four candidate examples
  completed with all bounded behavior checks passing. The skill example again had
  a provider-call failure. Sanitized Jev decision traces show actual answered and
  ambiguous judgments, with no captured Jev execution failures in this run.
- [Isolated skill follow-up](skill-provider-diagnostic.json): both provider decisions
  completed; the fixture artifact ran successfully and the answer was correct.
  This successful retry does not erase the earlier failures or establish their cause.

In total, 13 of 16 arm-runs completed. All five intended candidate behaviors were
observed, across the follow-ups; there was **no clean five-case candidate run**.
The initial provider failures were not captured with enough detail for diagnosis.
The adapter now retains only allowlisted error categories for future failures,
without persisting stderr or credentials. Do not interpret this small, adapted,
non-repeated experiment as a reliability comparison or a passed DC gate.

Raw reports retain synthetic context, answers, proposed/executed actions, results,
provider usage and elapsed times. The baseline arms were not rerun after the fix;
only the candidate history path changed. Different attempts and different prompts
must not be pooled into a paired latency or cost claim. The repeated padding in the
long skill is a synthetic boundary fixture, not a real household skill.

Verification after the history change: **130 focused tests pass**. The full suite
reports **767 passed, 36 failed**, with the same 36 failure identities as the saved
unchanged-HEAD comparison. These existing failures remain open.

## Run a probe

Context inspection without providers (missing Jev conservatively retains history):

```sh
python -m alphonse.agent_v2.evaluation.context_pilot \
  --context-only --output /tmp/context-pilot.json
```

Live decisions, using configured provider settings and disposable fixture tools:

```sh
python -m alphonse.agent_v2.evaluation.context_pilot \
  --live --responder alphonse.agent_v2.evaluation.context_pilot_live:respond \
  --output /tmp/context-pilot-live.json
```

Use `--arm candidate` or `--case skill-instructions` for an explicitly separate
follow-up report. Preserve previous reports. The provider adapter cannot execute
its proposed tools; the runner validates the registered tool/schema and restricts
Bash to one fixed fixture command or an exact read of the fixture skill file.

Next useful step: diagnose provider-call failures if they recur, then run one fresh
five-case comparison with all attempts retained. Broader validation remains defined
in [the evaluation protocol](../../13-decision-context-evaluation.md).
