/** Locally observed terminal failures; these never authorize a retry or change daemon state. */
export type ConversationFailureRecord = { messageId: string; failureCode: string; dismissed: boolean };
const storageKey = (user: string) => `alphonse.desktop.conversationFailures.${encodeURIComponent(user)}`;

export function readConversationFailures(storage: Storage, user: string): ConversationFailureRecord[] {
  if (!user) return [];
  try {
    const parsed: unknown = JSON.parse(storage.getItem(storageKey(user)) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is ConversationFailureRecord => item !== null && typeof item === "object"
      && typeof item.messageId === "string" && /^[0-9a-f]{32}$/.test(item.messageId)
      && typeof item.failureCode === "string" && /^[a-z_]{0,80}$/.test(item.failureCode)
      && typeof item.dismissed === "boolean");
  } catch { return []; }
}

export function saveConversationFailures(storage: Storage, user: string, records: ConversationFailureRecord[]): boolean {
  if (!user) return false;
  try { storage.setItem(storageKey(user), JSON.stringify(records)); return true; }
  catch { return false; }
}
