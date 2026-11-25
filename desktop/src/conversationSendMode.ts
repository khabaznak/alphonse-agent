export type ConversationSendMode = "queue" | "steer";
export function conversationSendLabel(mode: ConversationSendMode): string {
  return mode === "queue" ? "Send after this task" : "Send as update to this task";
}
export function readConversationSendMode(storage: Pick<Storage, "getItem">, user: string): ConversationSendMode {
  if (!user) return "queue";
  return storage.getItem(`alphonse.desktop.conversationSendMode.${encodeURIComponent(user)}`) === "steer" ? "steer" : "queue";
}
export function writeConversationSendMode(storage: Pick<Storage, "setItem">, user: string, mode: ConversationSendMode): void {
  if (user) storage.setItem(`alphonse.desktop.conversationSendMode.${encodeURIComponent(user)}`, mode);
}

/** null represents an idle composer; undefined means no draft has been started. */
export type DraftTask = string | null | undefined;
export function draftTaskChanged(draftTask: DraftTask, currentTask: string | undefined): boolean {
  return draftTask !== undefined && draftTask !== (currentTask ?? null);
}
export function primarySendMode(task: string | undefined, preferred: ConversationSendMode): ConversationSendMode | "normal" {
  return task ? preferred : "normal";
}
export function alternateSendMode(preferred: ConversationSendMode): ConversationSendMode {
  return preferred === "queue" ? "steer" : "queue";
}
export function shouldSendOnKey(key: string, shiftKey: boolean, enterToSend: boolean): boolean {
  return enterToSend && key === "Enter" && !shiftKey;
}
