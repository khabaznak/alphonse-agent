import type { OwnerRequest } from "./ownerConversation";
import type { CommittedEffect } from "./types";

/** Only the peer-verified server can return the current owner's requests. */
export type OwnerRecoveryItem = {
  message_id: string;
  prompt: string;
  status: string;
  can_abandon: boolean;
  committed_effect?: CommittedEffect;
};

export async function listOwnerRecovery(request: OwnerRequest): Promise<OwnerRecoveryItem[]> {
  const result = await request<{ items: OwnerRecoveryItem[] }>("experimental_owner_recovery_list", {});
  if (!result || !Array.isArray(result.items) || result.items.some((item) =>
    !item || !/^[0-9a-f]{32}$/.test(item.message_id) || typeof item.prompt !== "string"
      || typeof item.status !== "string" || typeof item.can_abandon !== "boolean"
      || (item.committed_effect !== undefined && (!item.committed_effect
        || item.committed_effect.kind !== "scheduled_task_cancelled"
        || typeof item.committed_effect.schedule_id !== "string"
        || typeof item.committed_effect.name !== "string"
        || typeof item.committed_effect.execution_may_have_started !== "boolean")))) {
    throw new Error("Unfinished requests could not be checked.");
  }
  return result.items;
}

/** This runs only in response to the owner's explicit button action. */
export async function abandonOwnerRequest(request: OwnerRequest, messageId: string): Promise<string> {
  if (!/^[0-9a-f]{32}$/.test(messageId)) throw new Error("This request could not be stopped.");
  const result = await request<{ status: string; delivery_status: string }>("experimental_owner_abandon", { message_id: messageId });
  if (result?.status !== "abandoned" || !["unconfirmed", "no_pickup_recorded"].includes(result.delivery_status)) {
    throw new Error("This request could not be stopped.");
  }
  return result.delivery_status;
}
