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
      <p>You can continue the conversation below.</p>
      <div className="conversation-failure-actions">
        {onOpenModelSettings && <button type="button" onClick={onOpenModelSettings}>Open model settings</button>}
        <button type="button" onClick={() => onDismissedChange(true)}>Dismiss notice</button>
      </div>
      <details className="conversation-failure-details">
        <summary>More about this request</summary>
        <p>The available record may not show everything that happened. If the request involved a change, its effects may be unconfirmed; check the relevant service or device before trying that change again.</p>
        {hasWorkLog && <p>Recorded activity may provide more context; it is not proof of an action’s effects.</p>}
        <div className="conversation-failure-actions">
          {onReviewRecovery && <button id={recoveryTriggerId} type="button" onClick={onReviewRecovery}>Review request status</button>}
          {onViewWorkLog && <button type="button" onClick={onViewWorkLog}>Review recorded activity</button>}
          {onDraftQuestion && <button type="button" onClick={onDraftQuestion}>Reference this failed request in draft</button>}
        </div>
      </details>
    </> : <button type="button" onClick={() => onDismissedChange(false)}>Request failed · View details</button>}
  </section>;
}
