import type { OwnerRequest } from "./ownerConversation";

export type LegacyNotification = {
  outbox_message_id: string; kind: string; status: string; message: string;
  created_at: string; attempt_count: number; delivery_certainty: "not_attempted" | "unknown";
  review_token: string; can_archive: boolean; review_restriction?: "" | "opaque_response" | "linked_or_protected" | null;
};

export async function listLegacyNotifications(request: OwnerRequest): Promise<LegacyNotification[]> {
  const result = await request<{ items?: unknown }>("legacy_notification_list", {});
  if (!result || !Array.isArray(result.items) || !result.items.every((item: unknown) => {
    if (!item || typeof item !== "object") return false;
    const row = item as Record<string, unknown>;
    return typeof row.outbox_message_id === "string" && !!row.outbox_message_id
      && typeof row.kind === "string" && typeof row.status === "string" && typeof row.message === "string"
      && typeof row.created_at === "string" && typeof row.attempt_count === "number"
      && Number.isSafeInteger(row.attempt_count) && row.attempt_count >= 0
      && (row.delivery_certainty === "not_attempted" || row.delivery_certainty === "unknown")
      && typeof row.review_token === "string" && /^[0-9a-f]{64}$/.test(row.review_token)
      && typeof row.can_archive === "boolean"
      && (row.review_restriction === undefined || row.review_restriction === null || row.review_restriction === "" || row.review_restriction === "opaque_response" || row.review_restriction === "linked_or_protected")
      && (row.review_restriction !== "opaque_response" || (row.kind === "response" && row.message === ""))
      && (row.review_restriction !== "linked_or_protected" || (row.can_archive === false && row.message === ""));
  })) throw new Error("legacy notification review unavailable");
  return result.items as LegacyNotification[];
}

export async function archiveLegacyNotification(request: OwnerRequest, item: LegacyNotification): Promise<void> {
  if (!item.can_archive || !item.outbox_message_id || !/^[0-9a-f]{64}$/.test(item.review_token)) {
    throw new Error("legacy notification review unavailable");
  }
  const result = await request<{ status?: string; delivery_certainty?: string }>("legacy_notification_archive", {
    message_id: item.outbox_message_id, review_token: item.review_token,
  });
  if (result?.status !== "archived_without_resending"
      || (result.delivery_certainty !== "not_attempted" && result.delivery_certainty !== "unknown")) {
    throw new Error("legacy notification review unavailable");
  }
}
