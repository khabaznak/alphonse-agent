// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatMessage } from "./types";
import { useConversationScroll } from "./useConversationScroll";

function viewMessage(id: string, role: ChatMessage["role"] = "user"): ChatMessage {
  return { id, role, content: id };
}

function Timeline({ messages, progressIds, working }: {
  messages: ChatMessage[]; progressIds: string[]; working: boolean;
}) {
  const { timelineRef, rememberPosition, followActivity } = useConversationScroll({
    scope: "owner/day", visible: true, messages, progressIds, working,
  });
  return <div ref={timelineRef} onScroll={rememberPosition}>
    {messages.map((message) => <article data-conversation-message-id={message.id} key={message.id}>{message.content}</article>)}
    <button onClick={followActivity}>Work log update</button>
  </div>;
}

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("conversation scroll", () => {
  const messages = [viewMessage("one"), viewMessage("two", "assistant"), viewMessage("three")];
  let host: HTMLElement;
  let root: ReturnType<typeof createRoot>;
  let pane: HTMLElement;
  let mounts: Map<string, { top: number; height: number }>;
  let contentHeight: number;
  let scrollTop: number;

  async function render(entries: ChatMessage[], working = false, progressIds: string[] = []) {
    await act(async () => root.render(<Timeline messages={entries} working={working} progressIds={progressIds} />));
  }

  beforeEach(async () => {
    host = document.body.appendChild(document.createElement("div"));
    root = createRoot(host);
    await render([]);
    pane = host.firstElementChild as HTMLElement;
    contentHeight = 650;
    scrollTop = 0;
    mounts = new Map([
      ["one", { top: 0, height: 220 }], ["two", { top: 220, height: 200 }], ["three", { top: 420, height: 230 }],
    ]);
    Object.defineProperties(pane, {
      clientHeight: { value: 300 },
      scrollHeight: { get: () => contentHeight },
      scrollTop: { get: () => scrollTop, set: (value) => { scrollTop = Math.max(0, Math.min(value, contentHeight - 300)); } },
    });
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
      const mount = mounts.get(this.dataset.conversationMessageId ?? "");
      return mount ? new DOMRect(0, mount.top - scrollTop, 400, mount.height) : new DOMRect(0, 0, 400, 300);
    });
    await render(messages, true);
    expect(pane.scrollTop).toBe(350);
  });

  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); });

  it("keeps the earlier message anchored when history grows above it", async () => {
    pane.scrollTop = 125;
    pane.dispatchEvent(new Event("scroll"));
    for (const mount of mounts.values()) mount.top += 80;
    mounts.set("earlier", { top: 0, height: 80 });
    contentHeight = 730;
    await render([viewMessage("earlier"), ...messages], true);
    expect(pane.scrollTop).toBe(205);
    expect(host.querySelector("article[data-conversation-message-id='one']")?.getBoundingClientRect().top).toBe(-125);
  });

  it("brings a new question into view while the user is reading earlier messages", async () => {
    pane.scrollTop = 125;
    pane.dispatchEvent(new Event("scroll"));
    const question = viewMessage("question:clarify", "assistant");
    mounts.set(question.id, { top: 650, height: 100 });
    contentHeight = 750;
    await render([...messages, question]);
    expect(pane.scrollTop).toBe(450);
  });

  it("reveals the start of a tall reply once and keeps it there through a receipt refresh and work-log callback", async () => {
    const reply = viewMessage("reply", "assistant");
    mounts.set(reply.id, { top: 650, height: 700 });
    contentHeight = 1350;
    await render([...messages, reply]);
    expect(pane.scrollTop).toBe(638);
    await render([...messages, { ...reply, sequence: 4 }]);
    host.querySelector("button")?.click();
    await act(async () => { await new Promise<void>((resolve) => requestAnimationFrame(() => resolve())); });
    expect(pane.scrollTop).toBe(638);
  });

  it("continues to follow compact replies at the bottom", async () => {
    const reply = viewMessage("reply", "assistant");
    mounts.set(reply.id, { top: 650, height: 100 });
    contentHeight = 750;
    await render([...messages, reply]);
    expect(pane.scrollTop).toBe(450);
  });

  it("follows a status-only completion at the bottom", async () => {
    contentHeight = 760;
    await render(messages, false);
    expect(pane.scrollTop).toBe(460);
  });
});
