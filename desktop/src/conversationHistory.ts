import type { ChatMessage } from "./types";

export function mergeFreshConversationHistory(
  history: ChatMessage[],
  messagesReceivedDuringReload: ChatMessage[],
): ChatMessage[] {
  const merged = [...history];
  for (const message of messagesReceivedDuringReload) {
    const alreadyPresent = merged.some((candidate) =>
      candidate.id === message.id
      || (candidate.role === message.role && candidate.content === message.content),
    );
    if (!alreadyPresent) merged.push(message);
  }
  return merged;
}
