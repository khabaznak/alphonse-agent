/** Locally observed terminal failures; these never authorize a retry or change daemon state. */
export type FailureRecovery = { reply: string; language: string; activity?: { label: string; detail: string } };
export type ConversationFailureRecord = { messageId: string; failureCode: string; dismissed: boolean; recovery?: FailureRecovery };
const storageKey = (user: string) => `alphonse.desktop.conversationFailures.${encodeURIComponent(user)}`;

function boundedText(value: unknown, max: number): value is string {
  return typeof value === "string" && value.trim().length > 0 && Array.from(value).length <= max;
}

/** The checkpoint supplies contextual prose and its display language. */
export function decodeFailureRecovery(value: unknown): FailureRecovery | undefined {
  if (!value || typeof value !== "object") return;
  const { reply, language, activity } = value as Record<string, unknown>;
  if (!boundedText(reply, 1000) || typeof language !== "string" || !/^[a-z]{2,8}(?:-[a-z0-9]{1,8}){0,4}$/i.test(language)) return;
  if (activity && typeof activity === "object") {
    const { label, detail } = activity as Record<string, unknown>;
    if (boundedText(label, 100) && boundedText(detail, 240)) return { reply, language, activity: { label, detail } };
  }
  return { reply, language };
}

export function readConversationFailures(storage: Storage, user: string): ConversationFailureRecord[] {
  if (!user) return [];
  try {
    const parsed: unknown = JSON.parse(storage.getItem(storageKey(user)) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is ConversationFailureRecord => item !== null && typeof item === "object"
      && typeof item.messageId === "string" && /^[0-9a-f]{32}$/.test(item.messageId)
      && typeof item.failureCode === "string" && /^[a-z_]{0,80}$/.test(item.failureCode)
      && typeof item.dismissed === "boolean").map(({ messageId, failureCode, dismissed }) => ({ messageId, failureCode, dismissed }));
  } catch { return []; }
}

export function saveConversationFailures(storage: Storage, user: string, records: ConversationFailureRecord[]): boolean {
  if (!user) return false;
  try {
    // Recovery can contain household context. Persist only the local notice state.
    storage.setItem(storageKey(user), JSON.stringify(records.map(({ messageId, failureCode, dismissed }) => ({ messageId, failureCode, dismissed }))));
    return true;
  }
  catch { return false; }
}
