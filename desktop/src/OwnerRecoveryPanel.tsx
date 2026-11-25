import { useEffect, useRef, useState } from "react";
import type { OwnerRecoveryItem } from "./ownerRecovery";
import type { OwnerRequest } from "./ownerConversation";
import { ConversationWorkLog } from "./ConversationWorkLog";
import "./ownerRecoveryPanel.css";

const labels: Record<string, string> = {
  pending: "Pending or in progress",
  ready: "An answer is ready",
  failed: "Request needs attention",
  started_unconfirmed: "Answer delivery is unconfirmed",
  action_unconfirmed: "An action may have happened; its result is unknown",
  telegram_delivery_unconfirmed: "Telegram delivery is unconfirmed",
  telegram_reconciliation_required: "Telegram accepted the response; saving its history is unconfirmed",
  effect_committed: "The scheduled task was cancelled; the response is unavailable",
  unavailable: "Result unavailable",
  continuation_declined: "You chose not to continue",
  continuation_accepted: "Continuation accepted; waiting for the worker",
  continuation_expired: "Continuation expired before the worker resumed",
};

const noStopStatuses = new Set(["action_unconfirmed", "telegram_delivery_unconfirmed", "telegram_reconciliation_required", "effect_committed"]);

function RequestCard({ item, ordinal, stopping, actionsEnabled, onStop, onShow, workLogRequest }: {
  item: OwnerRecoveryItem; ordinal: number; stopping: string; actionsEnabled: boolean; onStop: (id: string) => void; onShow: (id: string) => void; workLogRequest?: OwnerRequest;
}) {
  const [confirming, setConfirming] = useState(false);
  const cardRef = useRef<HTMLLIElement>(null);
  const stopButtonRef = useRef<HTMLButtonElement>(null);
  const confirmationHeadingRef = useRef<HTMLHeadingElement>(null);
  const wasConfirming = useRef(false);
  const focusInConfirmation = useRef(false);
  const restoreStopFocus = useRef(false);
  const title = item.prompt || "Untitled request";
  const canShow = actionsEnabled && item.status === "ready" && !item.can_abandon;
  // The server remains authoritative and rechecks eligibility during the action.
  const canStop = actionsEnabled && item.can_abandon && !noStopStatuses.has(item.status);
  const showConfirmation = canStop && confirming;
  useEffect(() => {
    if (showConfirmation && !wasConfirming.current) {
      confirmationHeadingRef.current?.focus();
    } else if (!showConfirmation && wasConfirming.current) {
      if (restoreStopFocus.current && canStop) stopButtonRef.current?.focus();
      else if (!canStop && focusInConfirmation.current) cardRef.current?.focus();
      restoreStopFocus.current = false;
      focusInConfirmation.current = false;
    }
    wasConfirming.current = showConfirmation;
    if (confirming && !canStop) setConfirming(false);
  }, [showConfirmation, canStop, confirming]);
  const status = item.status === "ready" && item.can_abandon
    ? "Answer unavailable; request expired" : (labels[item.status] || "Request needs attention");
  const detailsId = `owner-recovery-details-${item.message_id}`;
  const description = item.status === "action_unconfirmed"
    ? "Check the relevant device or service before trying again. No stop action is available here."
    : item.status === "telegram_delivery_unconfirmed"
      ? "The response may have been sent. It will not be sent again automatically."
      : item.status === "telegram_reconciliation_required"
        ? "The response will not be sent again automatically."
        : item.status === "pending" && !canStop
          ? "Alphonse may still be working, or this request cannot be stopped safely here."
          : item.status === "started_unconfirmed"
            ? "An answer may already have been shown; its delivery is not confirmed."
            : item.status === "continuation_declined"
              ? "Earlier work remains recorded."
              : item.status === "continuation_expired"
                ? "Earlier work needs review."
                : "";
  return <li ref={cardRef} className="owner-request-card" id={`owner-recovery-${item.message_id}`} tabIndex={-1} aria-describedby={description ? detailsId : undefined}>
    <div className="owner-request-card-main">
      <h3 id={`owner-recovery-prompt-${item.message_id}`}>{title}</h3>
      {Array.from(item.prompt).length >= 240 && <p className="owner-request-excerpt-note">Request text may be shortened.</p>}
      <p className="owner-request-status"><strong>{status}</strong></p>
      {description && <p id={detailsId}>{description}</p>}
      {item.committed_effect && <p>Scheduled task: {item.committed_effect.name}.{item.committed_effect.execution_may_have_started && " The task may already have started."}</p>}
      {!actionsEnabled && <p>Check requests again before acting.</p>}
      {actionsEnabled && !canStop && !canShow && item.status !== "action_unconfirmed" && item.status !== "pending" && <p>No stop action is available here.</p>}
    </div>
    <div className="owner-request-card-actions">
      {canShow && <button type="button" disabled={Boolean(stopping)} aria-label={`Show answer for request ${ordinal}: ${title}`} onClick={() => onShow(item.message_id)}>Show answer</button>}
      {canStop && !confirming && <button ref={stopButtonRef} type="button" disabled={Boolean(stopping)} aria-label={`Stop request ${ordinal}: ${title}`} onClick={() => setConfirming(true)}>Stop request</button>}
    </div>
    {workLogRequest && <ConversationWorkLog request={workLogRequest} messageId={item.message_id} active={false} summary="Review recorded activity" showEvidenceLimits labelContextId={`owner-recovery-prompt-${item.message_id}`} />}
    {showConfirmation && <div className="owner-request-confirmation" role="group" aria-label={`Confirm stopping request ${ordinal}: ${title}`}
      onFocusCapture={() => { focusInConfirmation.current = true; }}
      onBlurCapture={(event) => { if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) focusInConfirmation.current = false; }}>
      <h4 ref={confirmationHeadingRef} tabIndex={-1}>Stop this request?</h4>
      <p>Stopping prevents Alphonse from continuing this request. Earlier records are kept; anything that may already have happened is not undone. An answer may already have been shown.</p>
      <div className="owner-request-confirmation-actions">
        <button type="button" disabled={Boolean(stopping)} aria-label={`Confirm stop request ${ordinal}: ${title}`} onClick={() => onStop(item.message_id)}>{stopping === item.message_id ? "Stopping…" : "Stop request"}</button>
        <button type="button" disabled={Boolean(stopping)} aria-label={`Keep request ${ordinal}: ${title}`} onClick={() => { restoreStopFocus.current = true; setConfirming(false); }}>Keep request</button>
      </div>
    </div>}
  </li>;
}

export function focusOwnerRecovery(messageId: string): boolean {
  const target = document.getElementById(`owner-recovery-${messageId}`);
  if (!target) return false;
  let parent = target.parentElement;
  while (parent) { if (parent instanceof HTMLDetailsElement) parent.open = true; parent = parent.parentElement; }
  target.focus();
  return document.activeElement === target;
}

export function focusConversationRequest(messageId: string): boolean {
  const target = document.getElementById(`owner-recovery-trigger-${messageId}`) || document.getElementById("conversation-timeline");
  if (!target) return false;
  let parent = target.parentElement;
  while (parent) { if (parent instanceof HTMLDetailsElement) parent.open = true; parent = parent.parentElement; }
  target.focus();
  return document.activeElement === target;
}

export function OwnerRecovery({ items, stopping, actionsEnabled = true, onStop, onShow, onReturnToConversation, workLogRequest }: {
  items: OwnerRecoveryItem[];
  stopping: string;
  actionsEnabled?: boolean;
  onStop: (id: string) => void;
  onShow: (id: string) => void;
  onReturnToConversation?: () => void;
  workLogRequest?: OwnerRequest;
}) {
  if (!items.length) return null;
  // Surface possible effects first without changing the order within each group.
  const important = items.filter(item => noStopStatuses.has(item.status) || item.status === "started_unconfirmed" || (item.status === "ready" && !item.can_abandon));
  const earlier = items.filter(item => !important.includes(item));
  const renderCard = (item: OwnerRecoveryItem) => <RequestCard key={item.message_id} item={item} ordinal={items.indexOf(item) + 1} stopping={stopping} actionsEnabled={actionsEnabled} onStop={onStop} onShow={onShow} workLogRequest={workLogRequest} />;
  return <section aria-label="Earlier requests" className="owner-recovery">
    <h2>Earlier requests</h2>
    {onReturnToConversation && <button type="button" onClick={onReturnToConversation}>Back to conversation</button>}
    <p>These requests have unfinished or unconfirmed results. Alphonse will not retry them automatically.</p>
    {important.length > 0 && <><h3>Results or effects to review</h3><ul className="owner-request-list">{important.map(renderCard)}</ul></>}
    {earlier.length > 0 && <details><summary>Other earlier requests ({earlier.length})</summary><p>These records may overlap with the paused work count.</p><ul className="owner-request-list">{earlier.map(renderCard)}</ul></details>}
  </section>;
}
