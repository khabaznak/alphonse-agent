import { useState } from "react";
import type { OwnerRequest } from "./ownerConversation";
import { archiveLegacyNotification, listLegacyNotifications, type LegacyNotification } from "./legacyNotifications";

/** Explicit local-owner review; the server rechecks identity and state. */
export function LegacyNotificationReview({ request, onChanged }: { request: OwnerRequest; onChanged: () => void }) {
  const [items, setItems] = useState<LegacyNotification[] | null>(null);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setBusy(true);
    try {
      const result = await listLegacyNotifications(request);
      setItems(result);
      setNotice(result.length ? "" : "No owner-assigned notifications are available for review. Unassigned or ambiguous items need separate review.");
      onChanged();
    } catch {
      setItems(null);
      setNotice("Earlier notifications could not be checked. Nothing was changed.");
    } finally { setBusy(false); }
  }

  async function archive(item: LegacyNotification) {
    if (!item.can_archive || !window.confirm("Archive this notification without sending it? This does not tell us whether it was previously delivered.")) return;
    setBusy(true);
    try {
      await archiveLegacyNotification(request, item);
      setItems((current) => current?.filter((row) => row.outbox_message_id !== item.outbox_message_id) ?? null);
      setNotice("Notification archived without resending. Any earlier delivery remains unconfirmed.");
      onChanged();
    } catch {
      setNotice("The notification could not be archived. Its state may have changed; review it again.");
    } finally { setBusy(false); }
  }

  return <section className="composer-hint" aria-label="Earlier notifications">
    <button type="button" disabled={busy} onClick={() => void refresh()}>Review earlier notifications</button>
    {notice && <p role="status">{notice}</p>}
    {items?.map((item) => <article key={item.outbox_message_id}>
      {item.review_restriction ? <p>This linked or protected item needs separate recovery. Its contents are not shown here.</p> : <p>{item.message}</p>}
      <p>Type: {item.kind}. Recorded status: {item.status}. Previous delivery: {item.delivery_certainty === "not_attempted" ? "no attempt recorded" : "unknown"}. Attempts: {item.attempt_count}.</p>
      {!item.review_restriction && <button type="button" disabled={busy || !item.can_archive} onClick={() => void archive(item)}>Archive without sending</button>}
    </article>)}
  </section>;
}
