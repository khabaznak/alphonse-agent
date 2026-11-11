# Independent held-out corpus preparation

This is an operational preparation for #217 and #218, not a frozen corpus or
permission to run paid models. The present development material is public and
must never become the confirmatory held-out set. The current arms, executable
oracles, independent evaluator and independent anchor are not yet established.

1. First finish the development arms, their capabilities, tool and authorization
   policies, prompts, model configurations and analysis plan. Record and review
   their actual immutable artifacts and revisions. Once an implementer sees any
   held-out case, changing an arm or its prompt means that exposed case cannot be
   used as confirmatory evidence for that changed arm.
2. Assign a separate evaluator who is not tuning those implementations. They
   independently author at least 24 synthetic cases and answer/effect oracles,
   distinct from development cases, in evaluator-controlled storage. At least
   four feasible cases each must cover prescription and traffic; the protocol
   requires additional negative, recovery, cancellation, mutation and delivery
   cases. Use independently supplied expected outcomes and observed fixture
   events, including an independent receipt where a notification is claimed.
   Model-facing material contains only the active task and authorized fixture
   evidence; it must not include its oracle, other cases or hidden solutions.
3. For each private case, the evaluator records metadata for
   `alphonse.agent_v2.evaluation.heldout_coverage.assess_coverage`: opaque ID,
   author, family, feasible flag, protocol coverage tags, task and oracle
   SHA-256 hashes, oracle implementation/review assertions and supported arms.
   An example of the input schema is in `tests/test_heldout_coverage.py` and
   contains no actual held-out cases. Call the function inside the evaluator's
   private environment; it emits only aggregate counts, missing categories and
   structural blockers, never case IDs, text, authors, hashes or answers. Avoid
   publishing aggregates if the case inventory itself is sensitive.
4. Resolve reported gaps, then independently review each executable oracle,
   source/answer separation, safety checks, available recovery routes, response
   audience, deadlines, matched tools, and outcome/receipt observability. The
   checker evaluates **self-reported labels only**: even zero structural
   blockers cannot establish case quality, diverse authorship, correct
   implementation, actual arm support or meaningful statistical precision.
   Its `qualification` is always `unassessed`. Unsupported cases stay in the
   run denominator and prevent a candidate capability pass.
5. After independent review and *before* exposure or execution, commit to the
   exact evaluator-controlled corpus, answer keys, protocol, code, configurations
   and randomized schedule using the existing `decision_manifest` mechanism,
   and record the manifest digest in an independently controlled, timestamped
   channel. Do not publish plaintext or unsalted hashes of predictable case
   details: a low-entropy hash can leak its value by guessing. An independently
   timestamped digest of an encrypted, high-entropy randomized package can
   preserve a verifiable commitment if an appropriate authenticated-encryption
   and key-custody process is available. The evaluator must retain the exact
   bytes and keys; this repository does not implement encryption, establish an
   independent timestamp, or enforce access separation. Verify all inputs and
   actual executing environment before each run.
6. Explicitly obtain the separately scoped live-run authorization in #217/#218,
   including named providers and models, synthetic payload, retention/logging,
   run count and maximum spend. Then execute the frozen schedule with fresh
   disposable state, preserving all failures, unknowns, unsupported cases and
   interruptions. Never resume an interrupted attempt by silently retrying it.
   Record actual provider usage where available; absence remains unknown.
7. After the confirmatory run, disclose appropriately sanitized case details,
   outcomes and limitations, and apply the predeclared decision criteria. If
   secrecy, matching, configuration, authorization or anchoring fails, label
   exposed outcomes exploratory; author a fresh held-out set for any later
   confirmatory run. Human review and the owner's decision remain necessary.

A digest proves only that supplied bytes match an earlier commitment; it does
not establish independent authorship, secrecy, oracle correctness, runtime
provenance, safety, or permission to transmit inputs. These checks deliberately
remain unverified in the manifest validator.
