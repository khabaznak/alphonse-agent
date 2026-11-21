import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { OwnerRequest } from "./ownerConversation";
import type { ChatMessage } from "./types";

export type ScheduledResult = {
  id: string; message: string; kind: "answer" | "question"; created_at: string; sequence: number;
};
export type PresentedResult = ScheduledResult & { acknowledged: boolean };

export function scheduledResultMessage(result: ScheduledResult): ChatMessage {
  return { id: `scheduled:${result.id}`, role: "assistant", content: result.message,
    source: "scheduled", created_at: result.created_at, sequence: result.sequence };
}

export function mergeScheduledResults(current: PresentedResult[], pending: ScheduledResult[],
  dismissed: ReadonlySet<string>): PresentedResult[] {
  const known = new Set(current.map(item => item.id));
  const next = [...current];
  for (const result of pending) {
    if (!known.has(result.id) && !dismissed.has(result.id)) {
      next.push({ ...result, acknowledged: false });
      known.add(result.id);
    }
  }
  return next;
}

export function awaitingPresentation(results: PresentedResult[], visible: boolean,
  inFlight: ReadonlySet<string>): PresentedResult[] {
  return visible ? results.filter(item => !item.acknowledged && !inFlight.has(item.id)) : [];
}

/** Acknowledgement follows rendering, and does not assert that anyone read it. */
export function OwnerScheduledResults({ request, active, onPresented }: {
  request: OwnerRequest; active: boolean; onPresented: (message: ChatMessage) => void;
}) {
  const [results, setResults] = useState<PresentedResult[]>([]);
  const [notice, setNotice] = useState("");
  const [tick, setTick] = useState(0);
  const polling = useRef(false);
  const inFlight = useRef(new Set<string>());
  const presented = useRef(new Set<string>());
  const dismissed = useRef(new Set<string>());

  useEffect(() => {
    if (!active) return;
    let mounted = true;
    const poll = async () => {
      if (polling.current) return;
      polling.current = true;
      try {
        const response = await request<{ results: ScheduledResult[] }>("owner_scheduled_results_list", {});
        if (mounted) setResults(current => mergeScheduledResults(current, response.results, dismissed.current));
      } catch {
        // Reconnect/poll later; no acknowledgement or scheduled work is replayed.
      } finally { polling.current = false; }
    };
    const refresh = () => { if (mounted) setTick(value => value + 1); };
    void poll();
    const timer = window.setInterval(() => { void poll(); refresh(); }, 5000);
    document.addEventListener("visibilitychange", refresh);
    return () => { mounted = false; window.clearInterval(timer); document.removeEventListener("visibilitychange", refresh); };
  }, [active, request]);

  useEffect(() => {
    if (!active || typeof document === "undefined") return;
    for (const result of awaitingPresentation(results, document.visibilityState === "visible", inFlight.current)) {
      inFlight.current.add(result.id);
      if (!presented.current.has(result.id)) {
        onPresented(scheduledResultMessage(result));
        presented.current.add(result.id);
      }
      void request<{ status: string }>("owner_scheduled_result_ack", { id: result.id })
        .then(reply => {
          if (reply.status !== "client_acknowledged") throw new Error("unconfirmed");
          setResults(current => current.map(item => item.id === result.id ? { ...item, acknowledged: true } : item));
          setNotice("");
        })
        .catch(() => setNotice("Delivery could not be confirmed. Reconnecting…"))
        .finally(() => inFlight.current.delete(result.id));
    }
  }, [active, results, request, onPresented, tick]);

  if (!active || !results.length || typeof document === "undefined") return null;
  return createPortal(<div className="owner-reminder-alert owner-scheduled-results" role="status">
    {results.map(result => <div key={result.id}>
      <strong>Scheduled task:</strong> {result.message}
      <button type="button" disabled={!result.acknowledged} onClick={() => {
        dismissed.current.add(result.id);
        setResults(current => current.filter(item => item.id !== result.id));
      }}>Dismiss</button>
    </div>)}
    {notice && <p>{notice}</p>}
  </div>, document.body);
}
