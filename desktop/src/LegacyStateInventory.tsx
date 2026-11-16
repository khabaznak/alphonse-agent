import { useEffect, useRef, useState } from "react";
import type { OwnerRequest } from "./ownerConversation";

type Counts = Record<string, number>;
type WorkItem = { id: string; status: string; engine: string; origin_marker: string };
type OutboxItem = { id: string; status: string; attempts: string; owner_task_id: string | null };
type ScheduleItem = { id: string; status: string };
type OccurrenceItem = { position: number; task_id: string; status: string; attempts: string; owner_queue_id: string | null; owner_outbox_id: string | null };
type ScheduledItems = { status: "unavailable" } | { status: "available"; tasks: { items: ScheduleItem[]; truncated: boolean }; occurrences: { items: OccurrenceItem[]; truncated: boolean } };
type WorkItems = { queue: { items: WorkItem[]; truncated: boolean }; checkpoints: { items: WorkItem[]; truncated: boolean }; outbox: { items: OutboxItem[]; truncated: boolean }; scheduled: ScheduledItems };
type Inventory = {
  queue_other_ingress: { statuses: Counts };
  checkpoints_other_ingress: { statuses: Counts };
  queue_owner_origin_marked: { statuses: Counts };
  checkpoints_owner_origin_marked: { statuses: Counts };
  owner_addressed_outbox: { statuses: Counts; attempt_counter: Counts };
  owner_scheduled_work: { status: "unavailable" } | { status: "available"; tasks: Counts; occurrences: Counts };
};

function checkedCounts(value: unknown, keys: string[]): Counts {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("invalid inventory");
  const counts = value as Record<string, unknown>;
  if (Object.keys(counts).length !== keys.length || keys.some((key) =>
    typeof counts[key] !== "number" || !Number.isSafeInteger(counts[key]) || (counts[key] as number) < 0)) {
    throw new Error("invalid inventory");
  }
  return counts as Counts;
}

export async function loadLegacyStateItems(request: OwnerRequest, actor: string): Promise<WorkItems> {
  if (!actor.trim()) throw new Error("invalid actor");
  const response = await request<{ status: string; items?: WorkItems }>("legacy_state_items", { actor_user_id: actor });
  if (response?.status !== "available" || !response.items) throw new Error("items unavailable");
  function checked(group: WorkItems["queue"], allowed: string[]) {
    if (!group || !Array.isArray(group.items) || group.items.length > 50 || typeof group.truncated !== "boolean") throw new Error("invalid items");
    const items = group.items.map((item) => {
      if (!item || typeof item.id !== "string" || !item.id || item.id.length > 256 ||
          !allowed.includes(item.status) || !["tactical_v2", "hierarchical_v3", "single_loop_experimental", "missing", "malformed", "other"].includes(item.engine) ||
          !["present", "absent", "unavailable"].includes(item.origin_marker)) throw new Error("invalid item");
      return { id: item.id, status: item.status, engine: item.engine, origin_marker: item.origin_marker };
    });
    return { items, truncated: group.truncated };
  }
  const outbox = response.items.outbox;
  if (!outbox || !Array.isArray(outbox.items) || outbox.items.length > 50 || typeof outbox.truncated !== "boolean") throw new Error("invalid items");
  const deliveries = outbox.items.map((item) => {
    if (!item || typeof item.id !== "string" || !item.id || item.id.length > 256 ||
        !["pending", "claimed", "retry_wait", "delivered", "failed", "unconfirmed", "other"].includes(item.status) ||
        !["zero", "positive", "invalid"].includes(item.attempts) ||
        (item.owner_task_id !== null && (typeof item.owner_task_id !== "string" || !item.owner_task_id || item.owner_task_id.length > 256))) throw new Error("invalid item");
    return { id: item.id, status: item.status, attempts: item.attempts, owner_task_id: item.owner_task_id };
  });
  const scheduled = response.items.scheduled;
  if (!scheduled || !["available", "unavailable"].includes(scheduled.status)) throw new Error("invalid items");
  let scheduleItems: ScheduledItems = { status: "unavailable" };
  if (scheduled.status === "available") {
    const { tasks, occurrences } = scheduled;
    if (!tasks || !occurrences || !Array.isArray(tasks.items) || !Array.isArray(occurrences.items) ||
        tasks.items.length > 50 || occurrences.items.length > 50 || typeof tasks.truncated !== "boolean" ||
        typeof occurrences.truncated !== "boolean") throw new Error("invalid items");
    const validId = (id: unknown): id is string => typeof id === "string" && id.length > 0 && id.length <= 256;
    const taskItems = tasks.items.map(item => {
      if (!item || !validId(item.id) || !["active", "paused", "completed", "cancelled", "failed", "other"].includes(item.status)) throw new Error("invalid item");
      return { id: item.id, status: item.status };
    });
    const occurrenceItems = occurrences.items.map((item, index) => {
      if (!item || item.position !== index + 1 || !validId(item.task_id) ||
          !["pending", "claimed", "enqueued", "processing", "response_pending", "retry_wait", "delivered", "failed", "delivery_failed", "unconfirmed", "cancelled", "completed", "other"].includes(item.status) ||
          !["zero", "positive", "invalid"].includes(item.attempts) ||
          (item.owner_queue_id !== null && !validId(item.owner_queue_id)) || (item.owner_outbox_id !== null && !validId(item.owner_outbox_id))) throw new Error("invalid item");
      return { position: item.position, task_id: item.task_id, status: item.status, attempts: item.attempts, owner_queue_id: item.owner_queue_id, owner_outbox_id: item.owner_outbox_id };
    });
    scheduleItems = { status: "available", tasks: { items: taskItems, truncated: tasks.truncated },
      occurrences: { items: occurrenceItems, truncated: occurrences.truncated } };
  }
  return { queue: checked(response.items.queue, ["pending", "processing", "retry_wait", "failed", "other"]),
    checkpoints: checked(response.items.checkpoints, ["queued", "running", "waiting_user", "done", "failed", "cancelled", "other"]),
    outbox: { items: deliveries, truncated: outbox.truncated }, scheduled: scheduleItems };
}

export async function loadLegacyStateInventory(request: OwnerRequest, actor: string): Promise<Inventory> {
  if (!actor.trim()) throw new Error("inventory unavailable");
  const response = await request<{ status: string; inventory?: Inventory }>("legacy_state_inventory", { actor_user_id: actor });
  if (response?.status !== "available") throw new Error("inventory unavailable");
  const result = response.inventory;
  if (!result || !result.queue_other_ingress || !result.checkpoints_other_ingress || !result.queue_owner_origin_marked || !result.checkpoints_owner_origin_marked || !result.owner_addressed_outbox || !result.owner_scheduled_work) throw new Error("invalid inventory");
  const scheduled = result.owner_scheduled_work;
  if (scheduled.status !== "available" && scheduled.status !== "unavailable") throw new Error("invalid inventory");
  const queueKeys = ["pending", "processing", "retry_wait", "failed", "other"];
  const checkpointKeys = ["queued", "running", "waiting_user", "done", "failed", "cancelled", "other"];
  return {
    queue_other_ingress: { statuses: checkedCounts(result.queue_other_ingress.statuses, queueKeys) },
    checkpoints_other_ingress: { statuses: checkedCounts(result.checkpoints_other_ingress.statuses, checkpointKeys) },
    queue_owner_origin_marked: { statuses: checkedCounts(result.queue_owner_origin_marked.statuses, queueKeys) },
    checkpoints_owner_origin_marked: { statuses: checkedCounts(result.checkpoints_owner_origin_marked.statuses, checkpointKeys) },
    owner_addressed_outbox: {
      statuses: checkedCounts(result.owner_addressed_outbox.statuses, ["pending", "claimed", "retry_wait", "delivered", "failed", "unconfirmed", "other"]),
      attempt_counter: checkedCounts(result.owner_addressed_outbox.attempt_counter, ["zero", "positive", "invalid", "other"]),
    },
    owner_scheduled_work: scheduled.status === "available" ? {
      status: "available",
      tasks: checkedCounts(scheduled.tasks, ["active", "paused", "completed", "cancelled", "failed", "other"]),
      occurrences: checkedCounts(scheduled.occurrences, ["pending", "claimed", "enqueued", "processing", "response_pending", "retry_wait", "delivered", "failed", "delivery_failed", "unconfirmed", "cancelled", "completed", "other"]),
    } : { status: "unavailable" },
  };
}

function total(counts: Counts): number {
  const n = Object.values(counts).reduce((sum, value) => sum + value, 0);
  if (!Number.isSafeInteger(n)) throw new Error("invalid inventory");
  return n;
}

export function scheduledWorkSummary(work: Inventory["owner_scheduled_work"]): string {
  if (work.status === "unavailable") return "Scheduled work could not be checked.";
  return `Scheduled task records: ${total(work.tasks)}; linked occurrence records: ${total(work.occurrences)} (${work.occurrences.unconfirmed} marked unconfirmed; ${work.occurrences.delivery_failed} marked delivery failed).`;
}

export function LegacyStateInventory({ request, actor }: { request: OwnerRequest; actor: string }) {
  return <LegacyStateInventoryForActor key={actor} request={request} actor={actor} />;
}

function LegacyStateInventoryForActor({ request, actor }: { request: OwnerRequest; actor: string }) {
  const [summary, setSummary] = useState<{ queued: number; checkpoints: number; ownerQueued: number; ownerCheckpoints: number; notifications: number; unconfirmed: number; scheduled: string } | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "failed">("idle");
  const [details, setDetails] = useState<WorkItems | null>(null);
  const [detailsState, setDetailsState] = useState<"idle" | "loading" | "failed">("idle");
  const actorRef = useRef(actor);
  actorRef.current = actor;
  useEffect(() => { setSummary(null); setDetails(null); setState("idle"); setDetailsState("idle"); }, [actor]);
  const inspect = async () => {
    setSummary(null);
    setState("loading");
    try {
      const value = await loadLegacyStateInventory(request, actor);
      if (actorRef.current !== actor) return;
      setSummary({ queued: total(value.queue_other_ingress.statuses), checkpoints: total(value.checkpoints_other_ingress.statuses),
        ownerQueued: total(value.queue_owner_origin_marked.statuses), ownerCheckpoints: total(value.checkpoints_owner_origin_marked.statuses),
        notifications: total(value.owner_addressed_outbox.statuses), unconfirmed: value.owner_addressed_outbox.statuses.unconfirmed,
        scheduled: scheduledWorkSummary(value.owner_scheduled_work) });
      setState("idle");
    } catch {
      if (actorRef.current !== actor) return;
      setSummary(null);
      setState("failed");
    }
  };
  const inspectItems = async () => {
    setDetails(null);
    setDetailsState("loading");
    try { const value = await loadLegacyStateItems(request, actor); if (actorRef.current !== actor) return; setDetails(value); setDetailsState("idle"); }
    catch { if (actorRef.current === actor) setDetailsState("failed"); }
  };
  return <section className="owner-recovery" aria-label="Earlier work summary">
    <button type="button" disabled={state === "loading"} onClick={() => void inspect()}>{state === "loading" ? "Checking…" : "Check earlier work"}</button>
    {state === "failed" && <p role="status">Earlier work could not be checked.</p>}
    {summary && <div role="status"><p>{summary.queued} queue records and {summary.checkpoints} saved task records without a matching owner-origin marker; {summary.ownerQueued} queue records and {summary.ownerCheckpoints} saved task records with one; {summary.notifications} outbound records addressed to you ({summary.unconfirmed} marked unconfirmed).</p>
      <p>{summary.scheduled}</p>
      <p>This is a snapshot from the last check. Markers have not been verified. Counts can overlap and may include current items. They do not establish whether an action occurred or a message was received. Nothing was retried.</p></div>}
    <button type="button" disabled={detailsState === "loading"} onClick={() => void inspectItems()}>{detailsState === "loading" ? "Checking records…" : "List earlier work records"}</button>
    {detailsState === "failed" && <p role="status">Earlier work records could not be checked.</p>}
    {details && <div role="status"><p>Stored record identifiers and statuses only. These may include current work. A marker has not been verified; a status does not prove an action or delivery. No work was retried.</p>
      {([ ["Queue", details.queue], ["Saved tasks", details.checkpoints] ] as const).map(([label, group]) => <div key={label}>
        <p>{label}{group.truncated ? " (first 50; more records exist)" : ""}</p>
        {group.items.length === 0 ? <p>No records in this snapshot.</p> : <ul>{group.items.map(item => <li key={item.id}><code>{item.id}</code> — {item.status}; engine: {item.engine}; origin marker: {item.origin_marker}</li>)}</ul>}
      </div>)}
      <div><p>Outbound records addressed to you{details.outbox.truncated ? " (first 50; more records exist)" : ""}</p>
        <p>Stored attempt counters and task associations are unverified; they do not establish an effect or delivery.</p>
        {details.outbox.items.length === 0 ? <p>No records in this snapshot.</p> : <ul>{details.outbox.items.map(item => <li key={item.id}>
          <code>{item.id}</code> — {item.status}; stored attempts: {item.attempts}; {item.owner_task_id ? <>stored task association: <code>{item.owner_task_id}</code></> : "no owner-attributed task association"}
        </li>)}</ul>}
      </div>
      {details.scheduled.status === "unavailable" ? <p>Scheduled records could not be checked.</p> : <div>
        <p>Scheduled tasks{details.scheduled.tasks.truncated ? " (first 50; more records exist)" : ""}</p>
        {details.scheduled.tasks.items.length === 0 ? <p>No records in this snapshot.</p> : <ul>{details.scheduled.tasks.items.map(item => <li key={item.id}><code>{item.id}</code> — {item.status}</li>)}</ul>}
        <p>Scheduled occurrences{details.scheduled.occurrences.truncated ? " (first 50; more records exist)" : ""}. Stored links and statuses are unverified.</p>
        {details.scheduled.occurrences.items.length === 0 ? <p>No records in this snapshot.</p> : <ul>{details.scheduled.occurrences.items.map(item => <li key={item.position}>
          Snapshot item {item.position} — task <code>{item.task_id}</code>; {item.status}; stored attempts: {item.attempts};
          {item.owner_queue_id && <> queue: <code>{item.owner_queue_id}</code>;</>}
          {item.owner_outbox_id && <> outbound: <code>{item.owner_outbox_id}</code>;</>}
        </li>)}</ul>}
      </div>}
    </div>}
  </section>;
}
