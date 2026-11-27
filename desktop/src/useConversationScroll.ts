import { useCallback, useLayoutEffect, useRef } from "react";
import type { ChatMessage } from "./types";

const BOTTOM_GAP = 24;
const REPLY_INSET = 12;
const MESSAGE_SELECTOR = "[data-conversation-message-id]";

type ReadingAnchor = { messageId: string; offset: number };
type ScrollPosition = { following: boolean; anchor?: ReadingAnchor };

function messageElement(timeline: HTMLElement, messageId: string): HTMLElement | undefined {
  return Array.from(timeline.querySelectorAll<HTMLElement>(MESSAGE_SELECTOR)).find(
    (element) => element.dataset.conversationMessageId === messageId,
  );
}

function readingAnchor(timeline: HTMLElement): ReadingAnchor | undefined {
  const viewport = timeline.getBoundingClientRect();
  for (const message of timeline.querySelectorAll<HTMLElement>(MESSAGE_SELECTOR)) {
    const bounds = message.getBoundingClientRect();
    if (bounds.bottom > viewport.top && bounds.top < viewport.bottom && message.dataset.conversationMessageId) {
      return { messageId: message.dataset.conversationMessageId, offset: bounds.top - viewport.top };
    }
  }
}

export function useConversationScroll({ scope, visible, messages, working, progressIds }: {
  scope: string;
  visible: boolean;
  messages: readonly ChatMessage[];
  working: boolean;
  progressIds: readonly string[];
}) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const position = useRef<ScrollPosition>({ following: true });
  const previous = useRef({ scope, messages, working, progressIds });
  const activityFrame = useRef<number>(undefined);

  const rememberPosition = useCallback(() => {
    const timeline = timelineRef.current;
    if (!timeline || !timeline.clientHeight) return;
    const following = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight <= BOTTOM_GAP;
    position.current = { following, anchor: following ? undefined : readingAnchor(timeline) };
  }, []);

  useLayoutEffect(() => {
    const before = previous.current;
    previous.current = { scope, messages, working, progressIds };
    if (before.scope !== scope) position.current = { following: true };
    const timeline = timelineRef.current;
    if (!visible || !timeline || !timeline.clientHeight || before.scope !== scope) return;

    const last = messages.at(-1);
    const priorLast = before.messages.at(-1);
    const newTail = last && last.id !== priorLast?.id;
    const newActivity = (working && !before.working)
      || progressIds.some((id) => !before.progressIds.includes(id));
    const { following, anchor } = position.current;
    if (!newTail && !newActivity && !following) {
      const element = anchor && messageElement(timeline, anchor.messageId);
      if (element && anchor) {
        const offset = element.getBoundingClientRect().top - timeline.getBoundingClientRect().top;
        if (Math.abs(offset - anchor.offset) >= 1) timeline.scrollTop += offset - anchor.offset;
      }
      return;
    }

    if (!newTail && !newActivity && working === before.working) return;

    const reply = newTail && priorLast && last.role === "assistant" ? messageElement(timeline, last.id) : undefined;
    const bounds = reply?.getBoundingClientRect();
    if (bounds && bounds.height > timeline.clientHeight - BOTTOM_GAP) {
      timeline.scrollTop += bounds.top - timeline.getBoundingClientRect().top - REPLY_INSET;
      // The reader can now continue down the reply. A pending work-log frame
      // or a receipt refresh must not move them to the bottom of the card.
      position.current = { following: false, anchor: readingAnchor(timeline) };
      return;
    }
    timeline.scrollTop = timeline.scrollHeight;
    position.current = { following: true };
  }, [messages, progressIds, scope, visible, working]);

  const followActivity = useCallback(() => {
    if (!position.current.following || activityFrame.current !== undefined) return;
    activityFrame.current = window.requestAnimationFrame(() => {
      activityFrame.current = undefined;
      const timeline = timelineRef.current;
      if (timeline?.clientHeight && position.current.following) timeline.scrollTop = timeline.scrollHeight;
    });
  }, []);

  useLayoutEffect(() => () => {
    if (activityFrame.current !== undefined) {
      window.cancelAnimationFrame(activityFrame.current);
      activityFrame.current = undefined;
    }
  }, [scope, visible]);

  return { timelineRef, rememberPosition, followActivity };
}
