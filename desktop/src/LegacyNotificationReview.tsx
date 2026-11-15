import { useState } from "react";
import type { OwnerRequest } from "./ownerConversation";
import { archiveLegacyNotification, listLegacyNotifications, type LegacyNotification } from "./legacyNotifications";

export function legacyArchiveConfirmation(item: LegacyNotification): string {
  const prompt = item.review_restriction === "opaque_response"
    ? "This response's contents cannot be shown here and may be important. Archive it without resending? The record will be kept. Alphonse will not resend it; any previous delivery remains unknown."
    : "Archive this notification without sending it? This does not tell us whether it was previously delivered.";
  return `${prompt}\n\nRecorded: ${item.created_at || "unknown"}\nReference: ${item.outbox_message_id}`;
}

export function LegacyNotificationRow({ item, busy, onArchive }: { item: LegacyNotification; busy: boolean; onArchive: (item: LegacyNotification) => void }) {
  return <article>
    {item.review_restriction === "opaque_response" ? <p>This response's contents are hidden because their current source access cannot be verified. They may be important. The record will be kept if you choose to archive it without sending.</p> : item.review_restriction ? <p>This linked or protected item needs separate recovery. Its contents are not shown here.</p> : <p>{item.message}</p>}
    <p>Recorded: {item.created_at || "unknown"}. Reference: {item.outbox_message_id}.</p>
    <p>Type: {item.kind}. Recorded status: {item.status}. Previous delivery: {item.delivery_certainty === "not_attempted" ? "no attempt recorded" : "unknown"}. Attempts: {item.attempt_count}.</p>
    {item.review_restriction !== "linked_or_protected" && <button type="button" disabled={busy || !item.can_archive} onClick={() => onArchive(item)}>Archive without sending</button>}
  </article>;
}

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
    if (!item.can_archive || !window.confirm(legacyArchiveConfirmation(item))) return;
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
    {items?.map((item) => <LegacyNotificationRow key={item.outbox_message_id} item={item} busy={busy} onArchive={(row) => void archive(row)} />)}
  </section>;
}
