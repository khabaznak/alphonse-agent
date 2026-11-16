import { useState } from "react";
import type { OwnerRequest } from "./ownerConversation";

type Counts = Record<string, number>;
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
  const [summary, setSummary] = useState<{ queued: number; checkpoints: number; ownerQueued: number; ownerCheckpoints: number; notifications: number; unconfirmed: number; scheduled: string } | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "failed">("idle");
  const inspect = async () => {
    setSummary(null);
    setState("loading");
    try {
      const value = await loadLegacyStateInventory(request, actor);
      setSummary({ queued: total(value.queue_other_ingress.statuses), checkpoints: total(value.checkpoints_other_ingress.statuses),
        ownerQueued: total(value.queue_owner_origin_marked.statuses), ownerCheckpoints: total(value.checkpoints_owner_origin_marked.statuses),
        notifications: total(value.owner_addressed_outbox.statuses), unconfirmed: value.owner_addressed_outbox.statuses.unconfirmed,
        scheduled: scheduledWorkSummary(value.owner_scheduled_work) });
      setState("idle");
    } catch {
      setSummary(null);
      setState("failed");
    }
  };
  return <section className="owner-recovery" aria-label="Earlier work summary">
    <button type="button" disabled={state === "loading"} onClick={() => void inspect()}>{state === "loading" ? "Checking…" : "Check earlier work"}</button>
    {state === "failed" && <p role="status">Earlier work could not be checked.</p>}
    {summary && <div role="status"><p>{summary.queued} queue records and {summary.checkpoints} saved task records without a matching owner-origin marker; {summary.ownerQueued} queue records and {summary.ownerCheckpoints} saved task records with one; {summary.notifications} outbound records addressed to you ({summary.unconfirmed} marked unconfirmed).</p>
      <p>{summary.scheduled}</p>
      <p>This is a snapshot from the last check. Markers have not been verified. Counts can overlap and may include current items. They do not establish whether an action occurred or a message was received. Nothing was retried.</p></div>}
  </section>;
}
