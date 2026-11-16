import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LegacyStateInventory, loadLegacyStateInventory, loadLegacyStateItems, scheduledWorkSummary } from "./LegacyStateInventory";
import type { OwnerRequest } from "./ownerConversation";

const inventory = {
  queue_other_ingress: { statuses: { pending: 1, processing: 0, retry_wait: 0, failed: 2, other: 0 } },
  checkpoints_other_ingress: { statuses: { queued: 0, running: 0, waiting_user: 0, done: 1, failed: 0, cancelled: 0, other: 0 } },
  queue_owner_origin_marked: { statuses: { pending: 0, processing: 1, retry_wait: 0, failed: 0, other: 0 } },
  checkpoints_owner_origin_marked: { statuses: { queued: 0, running: 1, waiting_user: 0, done: 0, failed: 0, cancelled: 0, other: 0 } },
  owner_addressed_outbox: { statuses: { pending: 0, claimed: 0, retry_wait: 0, delivered: 1, failed: 0, unconfirmed: 1, other: 0 },
    attempt_counter: { zero: 0, positive: 2, invalid: 0, other: 0 } },
  owner_scheduled_work: { status: "available", tasks: { active: 1, paused: 0, completed: 0, cancelled: 0, failed: 0, other: 0 },
    occurrences: { pending: 0, claimed: 0, enqueued: 0, processing: 0, response_pending: 0, retry_wait: 0, delivered: 0, failed: 0, delivery_failed: 0, unconfirmed: 1, cancelled: 0, completed: 0, other: 0 } },
};

describe("earlier work inventory", () => {
  it("loads only bounded record metadata on explicit request", async () => {
    const items = { queue: { items: [{ id: "fixture-id", status: "pending", engine: "hierarchical_v3", origin_marker: "absent" }], truncated: false },
      checkpoints: { items: [], truncated: true },
      outbox: { items: [{ id: "delivery-id", status: "unconfirmed", attempts: "positive", owner_task_id: "fixture-id" }], truncated: false },
      scheduled: { status: "available", tasks: { items: [{ id: "job", status: "active" }], truncated: false },
        occurrences: { items: [{ position: 1, task_id: "job", status: "unconfirmed", attempts: "positive", owner_queue_id: "fixture-id", owner_outbox_id: "delivery-id" }], truncated: false } } };
    const request = vi.fn(async () => ({ status: "available", items }));
    expect(await loadLegacyStateItems(request as OwnerRequest, "fixture-owner")).toEqual(items);
    expect(request).toHaveBeenCalledExactlyOnceWith("legacy_state_items", { actor_user_id: "fixture-owner" });
    const invalid = vi.fn(async () => ({ status: "available", items: { ...items, queue: { items: [{ ...items.queue.items[0], status: "arbitrary" }], truncated: false } } }));
    await expect(loadLegacyStateItems(invalid as OwnerRequest, "fixture-owner")).rejects.toThrow();
    const invalidOutbox = vi.fn(async () => ({ status: "available", items: { ...items, outbox: {
      items: [{ ...items.outbox.items[0], attempts: "delivered" }], truncated: false } } }));
    await expect(loadLegacyStateItems(invalidOutbox as OwnerRequest, "fixture-owner")).rejects.toThrow();
    const invalidOccurrence = vi.fn(async () => ({ status: "available", items: { ...items, scheduled: { ...items.scheduled,
      occurrences: { items: [{ ...items.scheduled.occurrences.items[0], position: -1 }], truncated: false } } } }));
    await expect(loadLegacyStateItems(invalidOccurrence as OwnerRequest, "fixture-owner")).rejects.toThrow();
  });
  it("only requests the read-only owner inventory on demand", async () => {
    const request = vi.fn(async () => ({ status: "available", inventory }));
    const html = renderToStaticMarkup(<LegacyStateInventory request={request as OwnerRequest} actor="fixture-owner" />);
    expect(html).toContain("Check earlier work");
    expect(request).not.toHaveBeenCalled();
    const result = await loadLegacyStateInventory(request as OwnerRequest, "fixture-owner");
    expect(result).toEqual(inventory);
    expect(scheduledWorkSummary(result.owner_scheduled_work)).toBe("Scheduled task records: 1; linked occurrence records: 1 (1 marked unconfirmed; 0 marked delivery failed).");
    expect(request).toHaveBeenCalledExactlyOnceWith("legacy_state_inventory", { actor_user_id: "fixture-owner" });
  });

  it("rejects unavailable, malformed, or incomplete counts", async () => {
    const unavailable = vi.fn(async () => ({ status: "unavailable" }));
    await expect(loadLegacyStateInventory(unavailable as OwnerRequest, "fixture-owner")).rejects.toThrow();
    const invalid = vi.fn(async () => ({ status: "available", inventory: {
      ...inventory, queue_other_ingress: { statuses: { pending: -1 } },
    } }));
    await expect(loadLegacyStateInventory(invalid as OwnerRequest, "fixture-owner")).rejects.toThrow();
    await expect(loadLegacyStateInventory(invalid as OwnerRequest, "")).rejects.toThrow();
    const invalidSchedules = vi.fn(async () => ({ status: "available", inventory: { ...inventory,
      owner_scheduled_work: { status: "available", tasks: {}, occurrences: {} } } }));
    await expect(loadLegacyStateInventory(invalidSchedules as OwnerRequest, "fixture-owner")).rejects.toThrow();
    const unavailableSchedules = vi.fn(async () => ({ status: "available", inventory: { ...inventory,
      owner_scheduled_work: { status: "unavailable" } } }));
    const missing = (await loadLegacyStateInventory(unavailableSchedules as OwnerRequest, "fixture-owner")).owner_scheduled_work;
    expect(missing).toEqual({ status: "unavailable" });
    expect(scheduledWorkSummary(missing)).toBe("Scheduled work could not be checked.");
  });
});
