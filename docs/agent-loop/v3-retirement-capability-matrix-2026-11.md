# V3 retirement: owner-loop capability matrix

This is a source and disposable-test audit for #229 and #230, not a release
qualification or evidence about any particular installed process. The owner
directed immediate removal of executable V3 without waiting for parity. The
old loop and its executable replay were removed in `290225c3`; source routing,
settings, prompts and dependency findings are documented in
[the retirement surface audit](retirement-surface-audit-2026-11-15.md).
Retired task stamps and household data are not silently converted into new work.

The scope below is the selected owner-only conversation worker, not every
library or administration interface that remains in the repository. A tool's
presence in the general registry or a client library does not make it available
to that worker. Access, account readiness, claim, cancellation and other guards
continue to apply even for an advertised capability.

| Former/general capability | Owner conversation path | Remaining boundary |
| --- | --- | --- |
| Conversation and historical context | Normal owner conversation and bounded authorized source search/read are implemented. | It is not a full-household or general project worker; long, missing or inaccessible evidence may block an answer. |
| Durable knowledge | Source-bound capture and private owner read are under development. | The owner runtime does not register durable-knowledge reads or automatically extract knowledge; provider disclosure and recovery need qualification. |
| Skills | Owner can discover permitted skills and load explicitly named skill instructions. | General language-neutral activation, resource/script execution and automatic selection are not established. |
| Reminders and schedules | Owner schedule reads and a narrowly scoped one-time Desktop reminder are implemented. | Existing scheduled jobs and occurrences are preserved but not automatically resumed. General recurring scheduling, external delivery and broad language-neutral interpretation remain open. |
| Gmail | An authorized single-account INBOX unread message or thread count is implemented. | It is not general message search/send; token refresh requires a separate managed credential lifecycle. Reconnect and account readiness still matter. |
| Calendar | Calendar integration and family-calendar package code remain. | No general calendar read/write capability is exposed through this owner conversation host. |
| Browser, shell, external effects and integrations | Preserved native code and compatibility metadata do not activate these tools in the owner host. | Generic shell, browser effects, arbitrary delivery, event listeners and the old integration/scheduler workers are unavailable here. |
| Old queue, checkpoints and effects | Old state is preserved; owner-scoped, bounded read-only inventories expose selected metadata. | Inventory does not establish execution or delivery, and cannot replace actor-scoped effect reconciliation, explicit recovery, or a tested cross-version rollback. |

The default native-registry regression in `tests/test_owner_read_processor.py`
uses a disposable skill store and checks that guessed unsupported read and
mutation tools are not advertised or dispatched by the owner host. Optional
integrations are not included in the fixture. It uses synthetic data and a fake
provider; it does not establish semantic correctness, external effects, or
completeness of the entire application security boundary.

The source removal/audit is distinct from capability migration, retained-state
recovery and the single-active-processor release criterion. The currently
worded V3 audit checkbox in #229 also includes finishing shared capability and
legacy-state migration, so the audit evidence alone does not satisfy that whole
checkbox. The separate data/recovery and operational qualification criteria
likewise remain open.
