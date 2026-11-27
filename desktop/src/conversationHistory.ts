import type { ChatMessage } from "./types";
import { decodePrivateProviderSurface } from "./a2ui";

type HistoryMessage = ChatMessage & { private_card?: { provider?: unknown; envelopes?: unknown } };
export type HistoryResponse = { messages: HistoryMessage[]; private_cards_checked?: boolean };

export function desktopHistoryMessages(history: HistoryResponse, owner: string): ChatMessage[] {
  return history.messages.map(({ private_card: card, ...message }) => {
    if (history.private_cards_checked !== true || !owner || message.role !== "assistant" || message.source !== "desktop"
      || !/^owner:[0-9a-f]{32}$/.test(message.id) || (card?.provider !== "gmail" && card?.provider !== "calendar")) return message;
    const surface = decodePrivateProviderSurface(card.envelopes, message.id.slice(6), card.provider);
    return surface ? { ...message, private_reply_owner: owner, private_reply_surface: surface } : message;
  });
}

export function mergeFreshConversationHistory(
  history: ChatMessage[],
  messagesReceivedDuringReload: ChatMessage[],
): ChatMessage[] {
  const merged = new Map(history.map((message) => [message.id, message]));
  for (const message of messagesReceivedDuringReload) {
    const previous = merged.get(message.id);
    if (!previous) merged.set(message.id, message);
    else if (message.private_reply_owner && previous.role === "assistant") {
      // A live, owner-scoped provider pickup can include private details. History stores the receipt.
      merged.set(message.id, { ...previous, content: message.content, private_reply_owner: message.private_reply_owner,
        ...(message.private_reply_plain_text ? { private_reply_plain_text: true } : {}),
        ...(message.private_reply_surface ? { private_reply_surface: message.private_reply_surface } : {}),
        ...(message.private_outlook_inbox ? { private_outlook_inbox: message.private_outlook_inbox } : {}) });
    }
  }
  return orderConversationMessages([...merged.values()]);
}

export function orderConversationMessages(messages: ChatMessage[]): ChatMessage[] {
  return [...messages].sort(conversationOrder);
}

// Invalid or absent timestamps sort after dated events. Sequence and ID make
// ties deterministic across polling and reloads, regardless of input order.
function conversationOrder(left: ChatMessage, right: ChatMessage): number {
  const leftTime = Date.parse(left.created_at || "");
  const rightTime = Date.parse(right.created_at || "");
  const a = Number.isFinite(leftTime) ? leftTime : Infinity;
  const b = Number.isFinite(rightTime) ? rightTime : Infinity;
  if (a !== b) return a < b ? -1 : 1;
  const leftSequence = left.sequence && left.sequence > 0 ? left.sequence : Infinity;
  const rightSequence = right.sequence && right.sequence > 0 ? right.sequence : Infinity;
  if (leftSequence !== rightSequence) return leftSequence < rightSequence ? -1 : 1;
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}
