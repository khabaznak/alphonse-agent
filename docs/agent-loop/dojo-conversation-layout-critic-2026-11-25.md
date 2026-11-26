# Critic review: conversation layout

## Native conversation layout review

The user's 1113×747 screenshot showed an open At hand panel overlaying the conversation: message text, failure-card controls and the composer were clipped behind it; persistent footer instructions consumed additional vertical space. In the revised isolated native Desktop at its approximately 1120×760 window, the open At hand panel occupied a separate column, the failure controls wrapped within the conversation, and the composer and send button remained visible. Escape closed the panel and returned focus to its toggle. At approximately 941×681 the sidebar collapsed to icons, the open panel remained separate, and the visible conversation controls and composer did not overlap. An attempt to resize further did not change the window, so the stated 760×600 minimum was **not independently verified**. Message content above and below the visible timeline still required scrolling, as expected; complete keyboard traversal, persistence, screen readers and smaller windows remain unknown.

The synthetic Agenda widget displayed a raw SQLite thread `ProgrammingError` and thread IDs. This was observed with the isolated test worker; its cause and whether it exists with the ordinary household runtime were not established. It is separately a poor diagnostic presentation for this synthetic state and was reported to the UI owner. It should not be mistaken for proof that the responsive layout caused the database failure.

**Layout score: 8/10 for the two observed native window sizes, conditional pass for the demonstrated non-overlap and panel dismissal.** The original obstruction is resolved at those sizes, but minimum size and broader accessibility remain unverified, and the visible synthetic Agenda error is unresolved. This score is for layout, not a product-wide reliability or accessibility verdict.

### Minimum-size review attempt

The UI owner launched a test-only native app configured at 760×600 with a disposable worker and reported that the synthetic Agenda query succeeded after switching to file-backed stores. My independent attempt to select that app via computer use hung and was interrupted before any window state was returned. Consequently the 760×600 layout, empty and long context, overflow, send visibility and keyboard behavior are **unknown**, not passed. The cause and ordinary-runtime impact of the earlier Agenda exception remain unverified by the Critic. The prior conditional 8/10 layout score applies only to the approximately 1120×760 and 941×681 windows actually inspected.

### Reported 760×600 browser check

The UI owner separately reports running the actual App in a bounded browser harness with synthetic IPC at 760×600. In that report, the open drawer occupied x=494–760, the conversation x=72–494, and the composer x=92–474 and y=465–585; editor and send were in the viewport with no horizontal overflow. Closing the drawer reportedly expanded the composer and returned focus to the toggle. This is **second-hand browser evidence**, not an independent Critic observation or a native Desktop result. It improves confidence in the tested browser layout but does not change the independent 8/10 conditional score for the two native sizes observed. Native minimum-size behavior, long/empty-context coverage and assistive-technology behavior remain unverified by the Critic.

### Independent 760×600 browser review

I subsequently inspected the actual App rendered in the synthetic browser harness at an explicit 760×600 viewport. The open panel occupied its own column; the composer and send control were visible. Scrolling the conversation exposed all three failure controls, and Escape closed the panel, widened the conversation and returned focus to the panel toggle. A minor visual defect remained: the “Pending questions” widget heading broke into “Pending / question / s” beside its controls. The browser viewport override was reset afterward. This independently corroborates the narrow browser scenario, but does not establish native 760×600 behavior, long composer/context behavior, full keyboard or screen-reader accessibility. The prior scoped native layout score remains **8/10, conditional**.

### Compact widget heading retest

At 760×600 in the synthetic actual-App browser harness, the revised “Pending questions” heading appeared whole on one line, with its five controls on a separate visible row; the Agenda heading also fit. The specific awkward word break observed earlier is resolved in this browser state. The viewport override was reset. Native minimum-size behavior and broader accessibility remain unknown; the conditional **8/10 native layout** assessment remains scoped to the larger native windows actually inspected.
