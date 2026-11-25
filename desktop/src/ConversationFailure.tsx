/** Dismissal collapses the notice without changing the request or recovery state. */
export function ConversationFailure({ message, dismissed, onDismissedChange, onReviewRecovery, onOpenModelSettings }: {
  message: string; dismissed: boolean; onDismissedChange: (dismissed: boolean) => void;
  onReviewRecovery?: () => void;
  onOpenModelSettings?: () => void;
}) {
  return <section className="conversation-failure" aria-label="Request failed">
    {!dismissed ? <>
      <strong>Alphonse could not complete this request</strong>
      <p>{message}</p>
      <div className="conversation-failure-actions">
        {onReviewRecovery && <button type="button" onClick={onReviewRecovery}>Review unfinished work</button>}
        {onOpenModelSettings && <button type="button" onClick={onOpenModelSettings}>Open model settings</button>}
        <button type="button" onClick={() => onDismissedChange(true)}>Dismiss notice</button>
      </div>
    </> : <button type="button" onClick={() => onDismissedChange(false)}>Request failed · View details</button>}
  </section>;
}
