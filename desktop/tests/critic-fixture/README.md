# Disposable conversation activity fixture

From `desktop`, run `npm run dev -- --host 127.0.0.1` and open
`http://127.0.0.1:1420/tests/critic-fixture/`. This test-only page uses
production React components and CSS with fabricated data. Switch states with the
buttons and inspect the log disclosure and recovery controls. Navigation to the
recovery panel and all recovery actions are simulated; the page does not test
App routing, IPC, real model behavior, or household data. It is not an entry
point in the production build.

Switch from live to failed and then disconnected to inspect transitions. The
non-admin scenario omits the work-log and recovery controls. Reloading resets
fixture controls and does not demonstrate persisted production state. The test
fixture exercises the shared focus helper, not the actual App workspace route.

Additional live variants show a command with unknown effects, a reservation
whose execution is unknown, and no observations. The second recovery request
intentionally has an unavailable log. No fixture state persists across reloads.

The pending variant includes a bounded stated intent. Switch to expired reservation
to simulate a reservation without an authorized live intent, to settled to remove
the reservation, or to failed/disconnected. These are fixture snapshots, not proof
of production claim or lease handling.
