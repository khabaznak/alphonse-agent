import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { OwnerRequest } from "./ownerConversation";
import { readPendingReminder, reconcilePendingReminder, submitPendingReminder, type PendingReminder, type ReminderState } from "./ownerReminderSubmission";

type Offered = { id: string; message: string; due_at: string; acknowledgement_token: string };
type OpenReminder = { id: string; message: string; due_at: string; state: "pending" | "offered" };

export function reminderStateMessage(state: ReminderState): string {
  if (state === "cancelled") return "This reminder was already cancelled; it will not appear.";
  if (state === "acknowledged") return "This reminder was already reviewed; it will not repeat.";
  if (state === "offered") return "This reminder was already offered; check unconfirmed reminders below.";
  return "Reminder saved for this Desktop. It appears while Desktop is open.";
}

export async function takeVisibleReminder(request: OwnerRequest, active: boolean): Promise<Offered | null> {
  if (!active) return null;
  return (await request<{ reminder: Offered | null }>("owner_reminder_take", {})).reminder;
}

export async function dismissOfferedReminder(request: OwnerRequest, offered: Offered): Promise<void> {
  await request("owner_reminder_ack", { id: offered.id, token: offered.acknowledgement_token });
}

export function OwnerReminderOffer({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return <div role="alert"><strong>Reminder:</strong> {message}
    <button type="button" onClick={onDismiss}>Dismiss</button></div>;
}

export function OwnerReminderRecovery({ reminders, onResolve, onCancel }: {
  reminders: OpenReminder[]; onResolve: (id: string) => void; onCancel: (id: string) => void;
}) {
  if (!reminders.length) return null;
  return <details><summary>Upcoming and unconfirmed Desktop reminders</summary>
    {reminders.map(item => <p key={item.id}>{item.message} — {new Date(item.due_at).toLocaleString()}
      {item.state === "offered" ? <> · Pickup unconfirmed; it will not be automatically repeated.
        <button type="button" onClick={() => onResolve(item.id)}>Mark reviewed</button></> :
        <button type="button" onClick={() => onCancel(item.id)}>Cancel</button>}
    </p>)}
  </details>;
}

/** Explicit local-owner reminders. Due reminders are offered only once. */
export function OwnerReminders({ request, active, ownerId, managementTarget, onAttentionChange }: { request: OwnerRequest; active: boolean; ownerId: string; managementTarget?: HTMLElement | null; onAttentionChange?: (needed: boolean) => void }) {
  const [notice, setNotice] = useState("");
  const [offered, setOffered] = useState<Offered | null>(null);
  const [reminders, setReminders] = useState<OpenReminder[]>([]);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<PendingReminder | null>(null);
  const pendingKey = `alphonse.desktop.ownerReminder.pending.${ownerId}`;
  const polling = useRef(false);
  const shownRef = useRef(false);

  useEffect(() => {
    if (!active || !ownerId) return;
    try { setPending(readPendingReminder(window.localStorage, pendingKey)); }
    catch { setNotice("A pending reminder request cannot be checked. It may already be saved."); }
  }, [active, ownerId, pendingKey]);

  useEffect(() => {
    if (!active) return;
    let mounted = true;
    const check = async () => {
      if (polling.current || shownRef.current) return;
      polling.current = true;
      try {
        const result = await takeVisibleReminder(request, active);
        if (mounted && result) {
          shownRef.current = true;
          setOffered(result);
        }
        const open = await request<{ reminders: OpenReminder[] }>("owner_reminder_list", {});
        if (mounted) setReminders(open.reminders.filter(item => item.id !== result?.id));
      } catch {
        if (mounted) setNotice("Reminders are unavailable. Try again when Desktop is connected.");
      } finally { polling.current = false; }
    };
    void check();
    const timer = window.setInterval(() => void check(), 5000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, [active, request]);

  const needsAttention = Boolean(active && (pending || notice || reminders.some(item => item.state === "offered")));
  useEffect(() => {
    onAttentionChange?.(needsAttention);
    return () => onAttentionChange?.(false);
  }, [needsAttention, onAttentionChange]);

  const submit = async (item: PendingReminder) => {
    setBusy(true);
    try {
      const state = await submitPendingReminder(request, window.localStorage, pendingKey, item);
      setPending(null);
      setNotice(reminderStateMessage(state));
    } catch {
      setNotice("Reminder could not be confirmed. Check it or retry the exact saved request.");
    } finally { setBusy(false); }
  };

  const checkPending = async () => {
    if (!pending || busy) return;
    setBusy(true);
    try {
      const found = await reconcilePendingReminder(request, window.localStorage, pendingKey, pending);
      if (found) { setPending(null); setNotice(reminderStateMessage(found)); }
      else setNotice("The request was not found yet. You can safely retry the exact saved request.");
    } catch { setNotice("The pending reminder could not be checked; it has not been repeated."); }
    finally { setBusy(false); }
  };

  if (!active) return null;
  const management = <section aria-label="Desktop reminder recovery">
    {pending && <div role="status"><p>A reminder request may already be saved: {pending.message} — {new Date(pending.due_at).toLocaleString()}.</p>
      <button type="button" disabled={busy} onClick={() => void checkPending()}>Check saved reminder</button>
      <button type="button" disabled={busy} onClick={() => void submit(pending)}>Retry same request</button>
      <button type="button" disabled={busy} onClick={() => {
        try { window.localStorage.removeItem(pendingKey); setPending(null); setNotice("Tracking stopped. If it was saved, the reminder may still appear later."); }
        catch { setNotice("The pending request could not be cleared."); }
      }}>Stop tracking; saved reminder may still appear</button>
    </div>}
    {notice && <p role="status">{notice}</p>}
    <OwnerReminderRecovery reminders={reminders}
      onResolve={id => void request("owner_reminder_resolve", { id })
        .then(() => setReminders(current => current.filter(row => row.id !== id)))
        .catch(() => setNotice("That reminder could not be cleared."))}
      onCancel={id => void request("owner_reminder_cancel", { id })
        .then(() => setReminders(current => current.filter(row => row.id !== id)))
        .catch(() => setNotice("That reminder could not be cancelled; it may already be due."))} />
  </section>;
  return <>
    {managementTarget && createPortal(management, managementTarget)}
    {offered && typeof document !== "undefined" && createPortal(<div className="owner-reminder-alert">
      <OwnerReminderOffer message={offered.message} onDismiss={() => void dismissOfferedReminder(request, offered)
        .then(() => { setOffered(null); shownRef.current = false; })
        .catch(() => setNotice("Reminder acknowledgement could not be confirmed; the reminder remains visible."))} />
      {notice && <p role="status">{notice}</p>}
    </div>, document.body)}
  </>;
}
