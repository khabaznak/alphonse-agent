# Existing grocery list Dojo

A synthetic household folder contains a pantry, meal plan, and an existing
`shopping-list.json`. The owner asks for missing amounts in a new JSON file,
explicitly asks for a different name if the usual one is taken, and asks that
existing files remain unchanged. The test does not direct the agent to use a
particular tool or command.

In the [real-model attempt](evidence/dojo-file-conflict-20261119-01.jsonl),
Alphonse used injected disposable-container Bash, inspected the three files,
and created `shopping-list-missing.json`. The observed new file contained the
correct shortages: rice 2 and tomatoes 3; it also included beans 0, which was
accurate because the pantry already covered that item. The three existing
files' content hashes matched their seeded values. His reply named the new file
and said the old list was left unchanged. Content hashes verify final contents,
not that no transient write ever happened.

The initial evaluator incorrectly marked the output as a failure because it
required omitted zero amounts, even though the owner had not prohibited an
accurate zero. The original journal is preserved unchanged. A separately
recorded [regrade](evidence/dojo-file-conflict-20261119-01.regrade.json) applies
the corrected criteria and records this assumption change. The corrected
scorer accepts only accurate values for the meal-plan items and still checks
that the original files' contents are unchanged. One run is not a reliability
estimate, and the model's general ability to resolve conflicts remains to be
tested with new variants.

The authority-service prototype was being edited while this attempt started,
so the full-tree digest is an initial snapshot rather than an attestation of
all loaded files. [Before](evidence/dojo-file-conflict-20261119-01.before.json)
and [after](evidence/dojo-file-conflict-20261119-01.after.json) selected-file
digests match for the owner worker, runner, provider, and shell adapter. The
scenario module changed only after the run, to correct its grading assumption.
These hashes do not prove which code objects ran. The model provider has a
separate host environment, and this test uses injected container Bash rather
than the normal owner host-Bash path.
