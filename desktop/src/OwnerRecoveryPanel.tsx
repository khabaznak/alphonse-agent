import type { OwnerRecoveryItem } from "./ownerRecovery";

const labels: Record<string, string> = {
  pending: "Waiting to be processed",
  ready: "An answer is ready",
  failed: "Needs attention",
  started_unconfirmed: "Answer delivery is unconfirmed",
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
      {item.status === "ready" && !item.can_abandon && <button type="button" disabled={Boolean(stopping)} onClick={() => onShow(item.message_id)}>Show answer</button>}{" "}
      <button type="button" disabled={!item.can_abandon || Boolean(stopping)} onClick={() => onStop(item.message_id)}>
        {stopping === item.message_id ? "Stopping…" : "Stop request"}
      </button>
      {!item.can_abandon && <span> (still running or unavailable to stop)</span>}
    </li>)}</ul>
  </section>;
}
