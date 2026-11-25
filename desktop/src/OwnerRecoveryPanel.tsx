import type { OwnerRecoveryItem } from "./ownerRecovery";
import type { OwnerRequest } from "./ownerConversation";
import { ConversationWorkLog } from "./ConversationWorkLog";

const labels: Record<string, string> = {
  pending: "Waiting to be processed",
  ready: "An answer is ready",
  failed: "Needs attention",
  started_unconfirmed: "Answer delivery is unconfirmed",
  action_unconfirmed: "An action may have run; its effects are unconfirmed",
  telegram_delivery_unconfirmed: "Telegram delivery is unconfirmed",
  telegram_reconciliation_required: "Telegram accepted the response; saving its history is unconfirmed",
  effect_committed: "The scheduled task was cancelled; the response is unavailable",
  unavailable: "Result unavailable",
  continuation_declined: "You chose not to continue; earlier work remains recorded",
  continuation_accepted: "Continuation accepted; waiting for the worker",
  continuation_expired: "Continuation expired before the worker resumed; earlier work needs review",
};

/** Focus the matching recovery item after the attention workspace has rendered. */
export function focusOwnerRecovery(messageId: string): boolean {
  const target = document.getElementById(`owner-recovery-${messageId}`);
  if (!target) return false;
  target.focus();
  return document.activeElement === target;
}

/** Restore the originating request control after returning to the conversation. */
export function focusConversationRequest(messageId: string): boolean {
  const target = document.getElementById(`owner-recovery-trigger-${messageId}`) || document.getElementById("conversation-timeline");
  if (!target) return false;
  target.focus();
  return document.activeElement === target;
}

export function OwnerRecovery({ items, stopping, onStop, onShow, onReturnToConversation, workLogRequest }: {
  items: OwnerRecoveryItem[];
  stopping: string;
  onStop: (id: string) => void;
  onShow: (id: string) => void;
  onReturnToConversation?: () => void;
  workLogRequest?: OwnerRequest;
}) {
  if (!items.length) return null;
  return <section aria-label="Unfinished owner requests" className="owner-recovery">
    <h3>Unfinished requests</h3>
    {onReturnToConversation && <button type="button" onClick={onReturnToConversation}>Back to conversation</button>}
    <p>Requests are not retried automatically. You can stop an unfinished request after it is no longer running. An unconfirmed answer may already have been shown.</p>
    <ul>{items.map((item) => <li key={item.message_id} id={`owner-recovery-${item.message_id}`} tabIndex={-1}>
      <span id={`owner-recovery-prompt-${item.message_id}`}>{item.prompt || "Untitled request"}</span>{" — "}
      <span>{item.status === "ready" && item.can_abandon ? "Answer unavailable; request expired" : (labels[item.status] || "Needs attention")}</span>{" "}
      {item.committed_effect && <span>{item.committed_effect.name} </span>}
      {item.committed_effect?.execution_may_have_started && <span>The task may already have started. </span>}
      {item.status === "ready" && !item.can_abandon && <button type="button" disabled={Boolean(stopping)} onClick={() => onShow(item.message_id)}>Show answer</button>}{" "}
      <button type="button" disabled={!item.can_abandon || Boolean(stopping)} onClick={() => onStop(item.message_id)}>
        {stopping === item.message_id ? "Stopping…" : "Stop request"}
      </button>
      {!item.can_abandon && <span> {item.status === "action_unconfirmed"
        ? "(This record cannot confirm what happened. Check the relevant service or device directly before taking further action.)" : item.status === "telegram_delivery_unconfirmed"
          ? "(the response may have been sent; it will not be sent again automatically)"
          : item.status === "telegram_reconciliation_required" ? "(it will not be sent again automatically)"
          : "(still running or unavailable to stop)"}</span>}
      {workLogRequest && <ConversationWorkLog request={workLogRequest} messageId={item.message_id} active={false} summary="Review recorded activity" showEvidenceLimits labelContextId={`owner-recovery-prompt-${item.message_id}`} />}
    </li>)}</ul>
  </section>;
}
