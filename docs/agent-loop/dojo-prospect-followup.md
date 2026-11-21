# Prospect follow-up Dojo

The challenge asks Alphonse to create a follow-up CSV from disposable lead and
contact-history files. Qualified leads contacted on or after the stated cutoff
must be excluded, the output must contain names and companies, and the source
files must remain unchanged. The fixture does not specify a tool or command.

The [first real-model attempt](evidence/dojo-lead-followup-20261119-01.jsonl)
selected Bash to inspect the source files. The provider reported 71,785 input
tokens against a 65,536-token decision reservation, so the owner loop stopped
before executing Bash. No output file was created; the inputs were unchanged.
This is an observed failure to complete the task, not evidence that the chosen
command failed. The reported usage was accounted; it was not missing. The
provider's visible-envelope limit is in bytes and does not cap total provider
tokens. Later code reports a confirmed overrun separately from missing usage.

In the [second attempt](evidence/dojo-lead-followup-20261119-02.jsonl),
Alphonse inspected the inputs, created the CSV, and returned its contents.
Independent readback found the two correct rows and unchanged source files.
The trace also shows he wrote the destination twice: first with the selected
rows and again using a program that recomputed them. Although the final state
was correct, repeating a write is a useful recovery and efficiency observation,
especially for future challenges involving non-idempotent effects.

In a [later real-model attempt](evidence/dojo-lead-followup-20261119-03.jsonl),
the owner worker again used injected container Bash. Alphonse inspected both
inputs, computed and wrote the output once, and separately read it back. The
independent observer found the correct two rows and unchanged inputs. This
contrasts with the earlier repeated write; neither individual run establishes a
reliability rate or exercises the native host-Bash authorization boundary.

These are single development attempts through the real owner worker with an
injected, disposable container Bash. They do not establish that host Bash is
enabled in ordinary owner conversations. The output state is independently
checked, while reply semantics and human receipt are not automatically graded.
