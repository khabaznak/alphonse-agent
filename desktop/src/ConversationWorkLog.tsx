import { useEffect, useRef, useState } from "react";
import { ownerFailureMessage, type OwnerRequest } from "./ownerConversation";

type WorkLog = { status: string; failure_code: string | null; notice: string; events: { label: string; outcome: string }[] };

/** Only fixed, bounded descriptions from the owner-scoped endpoint are rendered. */
export function ConversationWorkLog({ request, messageId, active }: { request: OwnerRequest; messageId: string; active: boolean }) {
  const [open, setOpen] = useState(active);
  const wasActive = useRef(active);
  const [log, setLog] = useState<WorkLog | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    if (active && !wasActive.current) setOpen(true);
    if (!active && wasActive.current) setOpen(false);
    wasActive.current = active;
  }, [active]);
  useEffect(() => {
    if (!open && !active) return;
    let cancelled = false;
    const load = async () => {
      try {
        const result = await request<WorkLog>("experimental_owner_work_log", { message_id: messageId });
        if (!result || !Array.isArray(result.events) || result.events.length > 101
            || !result.events.every((event) => typeof event.label === "string" && event.label.length <= 100
              && typeof event.outcome === "string" && event.outcome.length <= 100)
            || typeof result.notice !== "string" || result.notice.length > 200
            || !["running", "blocked", "completed", "waiting_user", "cancelled"].includes(result.status)
            || (result.failure_code !== null && typeof result.failure_code !== "string")) throw new Error("invalid");
        if (!cancelled) { setLog(result); setUnavailable(false); }
      } catch { if (!cancelled) { setLog(null); setUnavailable(true); } }
    };
    void load();
    const timer = active ? window.setInterval(() => void load(), 2000) : undefined;
    return () => { cancelled = true; if (timer) window.clearInterval(timer); };
  }, [request, messageId, active, open]);
  return <details className="conversation-work-log" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Work log{active ? " · In progress" : ""}</summary>
    {unavailable ? <p>Work log is unavailable or not recorded for this request.</p> : !log ? <p>Checking recorded work…</p> : <>
      {log.status === "blocked" && <p>{ownerFailureMessage(log.failure_code || undefined)}</p>}
      {log.events.length ? <ol>{log.events.map((event, index) => <li key={index}>{event.label} · {event.outcome}</li>)}</ol>
        : <p>No tool observations recorded yet.</p>}
      <small>{log.notice}</small>
    </>}
  </details>;
}
