import type { OwnerRecoveryItem } from "./ownerRecovery";

const labels: Record<string, string> = {
  pending: "Waiting to be processed",
  ready: "An answer is ready",
  failed: "Needs attention",
  started_unconfirmed: "Answer delivery is unconfirmed",
  action_unconfirmed: "An action may have run; its effects are unconfirmed",
  effect_committed: "The scheduled task was cancelled; the response is unavailable",
  unavailable: "Result unavailable",
};

export function OwnerRecovery({ items, stopping, onStop, onShow }: {
  items: OwnerRecoveryItem[];
  stopping: string;
  onStop: (id: string) => void;
  onShow: (id: string) => void;
}) {
  if (!items.length) return null;
  return <section aria-label="Unfinished owner requests" className="owner-recovery">
    <h3>Unfinished requests</h3>
    <p>Requests are not retried automatically. You can stop an unfinished request after it is no longer running. An unconfirmed answer may already have been shown.</p>
    <ul>{items.map((item) => <li key={item.message_id}>
      <span>{item.prompt || "Untitled request"}</span>{" — "}
      <span>{item.status === "ready" && item.can_abandon ? "Answer unavailable; request expired" : (labels[item.status] || "Needs attention")}</span>{" "}
      {item.committed_effect && <span>{item.committed_effect.name} </span>}
      {item.committed_effect?.execution_may_have_started && <span>The task may already have started. </span>}
      {item.status === "ready" && !item.can_abandon && <button type="button" disabled={Boolean(stopping)} onClick={() => onShow(item.message_id)}>Show answer</button>}{" "}
      <button type="button" disabled={!item.can_abandon || Boolean(stopping)} onClick={() => onStop(item.message_id)}>
        {stopping === item.message_id ? "Stopping…" : "Stop request"}
      </button>
      {!item.can_abandon && <span> {item.status === "action_unconfirmed"
        ? "(check the result before trying again)" : "(still running or unavailable to stop)"}</span>}
    </li>)}</ul>
  </section>;
}
