import type { OwnerRequest } from "./ownerConversation";

export type PendingReminder = { message: string; due_at: string; request_id: string };
export type ReminderState = "pending" | "offered" | "acknowledged" | "cancelled";
type ReminderStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function reminderState(value: unknown): ReminderState {
  if (value !== "pending" && value !== "offered" && value !== "acknowledged" && value !== "cancelled") {
    throw new Error("owner_reminder_result_unconfirmed");
  }
  return value;
}

export function readPendingReminder(storage: ReminderStorage, key: string): PendingReminder | null {
  const raw = storage.getItem(key);
  if (raw === null) return null;
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== "object") throw new Error("pending_reminder_invalid");
  const item = value as Record<string, unknown>;
  if (Object.keys(item).sort().join(",") !== "due_at,message,request_id"
      || typeof item.message !== "string" || !item.message.trim() || item.message.length > 500
      || typeof item.due_at !== "string" || Number.isNaN(Date.parse(item.due_at))
      || typeof item.request_id !== "string" || !/^[0-9a-f]{32}$/.test(item.request_id)) {
    throw new Error("pending_reminder_invalid");
  }
  return { message: item.message, due_at: item.due_at, request_id: item.request_id };
}

export function persistPendingReminder(storage: ReminderStorage, key: string, item: PendingReminder): void {
  storage.setItem(key, JSON.stringify(item));
  const recorded = readPendingReminder(storage, key);
  if (!recorded || JSON.stringify(recorded) !== JSON.stringify(item)) throw new Error("pending_reminder_unavailable");
}

export async function submitPendingReminder(request: OwnerRequest, storage: ReminderStorage,
                                            key: string, item: PendingReminder): Promise<ReminderState> {
  // Store the exact operation before dispatch; a restart only reveals the
  // pending request and never automatically repeats it.
  persistPendingReminder(storage, key, item);
  const result = await request<{ id?: unknown; due_at?: unknown; state?: unknown }>("owner_reminder_create", item);
  if (typeof result?.id !== "string" || !/^[0-9a-f]{32}$/.test(result.id)
      || typeof result.due_at !== "string" || Date.parse(result.due_at) !== Date.parse(item.due_at)) {
    throw new Error("owner_reminder_result_unconfirmed");
  }
  const state = reminderState(result.state);
  storage.removeItem(key);
  return state;
}

export async function reconcilePendingReminder(request: OwnerRequest, storage: ReminderStorage,
                                               key: string, item: PendingReminder): Promise<ReminderState | null> {
  const result = await request<{ reminder?: { id?: unknown; message?: unknown; due_at?: unknown; state?: unknown } | null }>(
    "owner_reminder_lookup", { request_id: item.request_id });
  const found = result?.reminder;
  if (!found) return null;
  if (typeof found.id !== "string" || !/^[0-9a-f]{32}$/.test(found.id)
      || found.message !== item.message || typeof found.due_at !== "string"
      || Date.parse(found.due_at) !== Date.parse(item.due_at)) {
    throw new Error("owner_reminder_result_conflict");
  }
  const state = reminderState(found.state);
  storage.removeItem(key);
  return state;
}
