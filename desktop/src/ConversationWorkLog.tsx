import { useEffect, useRef, useState } from "react";
import { ownerFailureMessage, type OwnerRequest } from "./ownerConversation";

type WorkLog = { status: string; failure_code: string | null; notice: string; events: { label: string; outcome: string; gmail_label?: string }[] };

function validGmailLabel(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value === value.trim() && Array.from(value).length <= 256;
}

export function focusRequestWorkLog(messageId: string): boolean {
  const details = document.getElementById(`conversation-work-log-${messageId}`);
  if (!(details instanceof HTMLDetailsElement)) return false;
  details.open = true;
  const summary = details.querySelector<HTMLElement>("summary");
  summary?.focus();
  return document.activeElement === summary;
}

/** Render only bounded descriptions returned by the owner-scoped endpoint. */
export function ConversationWorkLog({ request, messageId, active, summary = "Work log", showEvidenceLimits = false, labelContextId, disclosureId }: { request: OwnerRequest; messageId: string; active: boolean; summary?: string; showEvidenceLimits?: boolean; labelContextId?: string; disclosureId?: string }) {
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
              && typeof event.outcome === "string" && event.outcome.length <= 100
              && (event.gmail_label === undefined || validGmailLabel(event.gmail_label)))
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
  const gmailLabels = (log?.events ?? []).flatMap(({ gmail_label }) => gmail_label ? [gmail_label] : []);
  const eventList = <ol>{log?.events.map((event, index) =>
    <li key={index}>{event.label}{event.gmail_label && (" · " + event.gmail_label)} · {event.outcome}</li>)}</ol>;
  return <details id={disclosureId} className="conversation-work-log" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary id={labelContextId ? `work-log-summary-${messageId}` : undefined} aria-labelledby={labelContextId ? `work-log-summary-${messageId} ${labelContextId}` : undefined}>{summary}{active ? " · In progress" : ""}</summary>
    {showEvidenceLimits && <p>These are recorded observations, not source content or proof of an action’s effects.</p>}
    {unavailable ? <p>Work log is unavailable or not recorded for this request.</p> : !log ? <p>Checking recorded work…</p> : <>
      {log.status === "blocked" && <p>{ownerFailureMessage(log.failure_code || undefined)}</p>}
      {gmailLabels.length > 0 ? <>
        <p>Gmail search · {gmailLabels.join(" → ")}</p>
        <details className="conversation-work-log-details">
          <summary>Technical details</summary>
          {eventList}
        </details>
      </> : log.events.length ? eventList : <p>No tool observations recorded yet.</p>}
      <small>{log.notice}</small>
    </>}
  </details>;
}
