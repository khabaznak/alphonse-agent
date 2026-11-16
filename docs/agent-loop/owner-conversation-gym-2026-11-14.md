# Owner conversation gym: 14 November 2026

These are disposable, subscription-backed development attempts through the
experimental owner worker and local socket, not the real Desktop application
or a household deployment. Data and database were synthetic. The socket's
client acknowledgment does not establish human receipt, and provider usage was
not captured. The worker ran with the local working tree based on `4e3e6a39`;
the tree was modified and the complete source snapshot was not recorded.

- The [short-note attempts](evidence/owner-only-synthetic-2026-11-14.jsonl)
  each returned the expected Thursday answer. The first did not capture tool
  coverage; the second recorded a search but no full-source read.
- The [truncated-update plan](evidence/owner-full-source-2026-11-14-plan.json)
  requires opening a note whose early Thursday plan is corrected to Saturday
  later in the same record. Its [receipts](evidence/owner-full-source-2026-11-14.jsonl)
  preserve two failed attempts with incomplete diagnosis, followed by three
  successful attempts on different post-fix revisions that recorded both
  search and source read and returned Saturday. The failures alone do not
  prove their cause. The later attempts reused a development case and are not
  held-out evidence.

Offline reproduction showed that the owner host rejected truncated search
previews even as discovery, preventing full-source selection. The experimental
host now withholds shortened preview text while retaining complete snippets
and discovery metadata. An independent two-source fixture showed why: an
earlier revision exposed a stale preview, read an unrelated source, and
accepted the stale answer. The initial full-read requirement also blocked cases
where a long unrelated result accompanies a complete useful one; the current
version permits a complete revalidated result without forcing an unrelated
read, and retains a diagnostic demonstrating that wrong guesses remain possible.
No structural check establishes
semantic verification or guarantees all relevant records were read, and a
clarification question may contain unsupported claims. Sources larger than
the native read limit still lack chunked reassembly. Continue multi-source and
conflicting-record exercises with independent outcome and coverage oracles.

For the third through fifth truncated-update attempts, `source_subset_sha256` hashes the
UTF-8 path, a zero byte, and file bytes in this order: `memory.py`,
`owner_read_processor.py`, `owner_read_recovery.py`,
`single_loop_candidate.py`, `candidate_decision_policy.py`, and
`owner_clarification.py`, using their full repository-relative paths. This is
not a complete manifest and does not identify all code or configuration.
