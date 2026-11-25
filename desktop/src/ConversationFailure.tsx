/** Dismissal collapses the notice without changing the request or recovery state. */
export function ConversationFailure({ message, dismissed, onDismissedChange, onReviewRecovery, onOpenModelSettings, onDraftQuestion, onViewWorkLog, recoveryTriggerId, hasWorkLog = false }: {
  message: string; dismissed: boolean; onDismissedChange: (dismissed: boolean) => void;
  onReviewRecovery?: () => void;
  onOpenModelSettings?: () => void;
  onDraftQuestion?: () => void;
  onViewWorkLog?: () => void;
  hasWorkLog?: boolean;
  recoveryTriggerId?: string;
}) {
  return <section className="conversation-failure" aria-label="Request failed">
    {!dismissed ? <>
      <strong>Alphonse could not complete this request</strong>
      <p>{message}</p>
      <p>{hasWorkLog && "The work log above may show recorded activity. "}If this request may have changed something, check the relevant service or device before taking further action.</p>
      {!onReviewRecovery && !onOpenModelSettings && !onDraftQuestion && !onViewWorkLog && <p>No recovery action is available here.</p>}
      <div className="conversation-failure-actions">
        {onReviewRecovery && <button id={recoveryTriggerId} type="button" onClick={onReviewRecovery}>Review request status</button>}
        {onViewWorkLog && <button type="button" onClick={onViewWorkLog}>Review recorded activity</button>}
        {onDraftQuestion && <button type="button" onClick={onDraftQuestion}>Reference this failed request in draft</button>}
        {onOpenModelSettings && <button type="button" onClick={onOpenModelSettings}>Open model settings</button>}
        <button type="button" onClick={() => onDismissedChange(true)}>Dismiss notice</button>
      </div>
    </> : <button type="button" onClick={() => onDismissedChange(false)}>Request failed · View details</button>}
  </section>;
}
