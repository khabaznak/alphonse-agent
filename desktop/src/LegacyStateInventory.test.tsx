import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LegacyStateInventory, loadLegacyStateInventory } from "./LegacyStateInventory";
import type { OwnerRequest } from "./ownerConversation";

const inventory = {
  queue_other_ingress: { statuses: { pending: 1, processing: 0, retry_wait: 0, failed: 2, other: 0 } },
  checkpoints_other_ingress: { statuses: { queued: 0, running: 0, waiting_user: 0, done: 1, failed: 0, cancelled: 0, other: 0 } },
  queue_owner_origin_marked: { statuses: { pending: 0, processing: 1, retry_wait: 0, failed: 0, other: 0 } },
  checkpoints_owner_origin_marked: { statuses: { queued: 0, running: 1, waiting_user: 0, done: 0, failed: 0, cancelled: 0, other: 0 } },
  owner_addressed_outbox: { statuses: { pending: 0, claimed: 0, retry_wait: 0, delivered: 1, failed: 0, unconfirmed: 1, other: 0 },
    attempt_counter: { zero: 0, positive: 2, invalid: 0, other: 0 } },
};

describe("earlier work inventory", () => {
  it("only requests the read-only owner inventory on demand", async () => {
    const request = vi.fn(async () => ({ status: "available", inventory }));
    const html = renderToStaticMarkup(<LegacyStateInventory request={request as OwnerRequest} actor="fixture-owner" />);
    expect(html).toContain("Check earlier work");
    expect(request).not.toHaveBeenCalled();
    expect(await loadLegacyStateInventory(request as OwnerRequest, "fixture-owner")).toEqual(inventory);
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
  });
});
