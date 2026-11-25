// @vitest-environment jsdom
import { act, createRef } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationComposer } from "./ConversationComposer";
import { ConversationSuggestion } from "./ConversationSuggestion";
import { draftTaskChanged, type ConversationSendMode } from "./conversationSendMode";
import type { OwnerRequest } from "./ownerConversation";
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("conversation composer routing", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());
  it.each([["queue", "steer"], ["steer", "queue"]] as const)("sends the %s preference or the %s alternative", async (mode, alternative) => {
    const element = document.createElement("div"), root = createRoot(element), send = vi.fn();
    const render = (task: string | undefined) => <ConversationComposer prompt="draft" task={task} mode={mode} enterToSend reviewDraft={false} textareaRef={createRef()} commandMenuRef={createRef()} suggestions={[]} commands={[]} context="Everyday assistant" suggestion={null} onChange={() => {}} onSend={send} onReview={() => {}} />;
    await act(async () => root.render(render("task-1")));
    await act(async () => element.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    expect(send).toHaveBeenLastCalledWith(mode, "task-1");
    await act(async () => (element.querySelector(".conversation-send-alternative") as HTMLButtonElement).click());
    expect(send).toHaveBeenLastCalledWith(alternative, "task-1");
    await act(async () => element.querySelector("textarea")!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true })));
    expect(send).toHaveBeenLastCalledWith(mode, "task-1");
    const count = send.mock.calls.length;
    await act(async () => element.querySelector("textarea")!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", shiftKey: true, bubbles: true, cancelable: true })));
    expect(send).toHaveBeenCalledTimes(count);
    await act(async () => root.render(render(undefined)));
    expect(element.querySelector(".conversation-send-alternative")).toBeNull();
    await act(async () => element.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    expect(send).toHaveBeenLastCalledWith("normal", undefined);
    await act(async () => root.unmount());
  });
  it("preserves a draft and blocks sends after the task changes until reviewed", async () => {
    const element = document.createElement("div"), root = createRoot(element), send = vi.fn(), review = vi.fn();
    const render = (task: string | undefined, draftTask: string | null) => <ConversationComposer prompt="keep this draft" task={task} mode="queue" enterToSend reviewDraft={draftTaskChanged(draftTask, task)} textareaRef={createRef()} commandMenuRef={createRef()} suggestions={[]} commands={[]} context={null} suggestion={null} onChange={() => {}} onSend={send} onReview={review} />;
    await act(async () => root.render(render(undefined, "old-task")));
    expect(element.querySelector("textarea")!.value).toBe("keep this draft");
    expect(element.textContent).toContain("The active task changed");
    await act(async () => element.querySelector("textarea")!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true })));
    expect(send).not.toHaveBeenCalled();
    await act(async () => [...element.querySelectorAll("button")].find(b => b.textContent?.includes("reviewed"))!.click());
    expect(review).toHaveBeenCalledOnce();
    await act(async () => root.render(render(undefined, null)));
    await act(async () => element.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    expect(send).toHaveBeenCalledWith("normal", undefined);
    await act(async () => root.unmount());
  });
  it("sends the suggested route and exact task only when its bubble button is pressed", async () => {
    const element = document.createElement("div"), root = createRoot(element), send = vi.fn();
    const request = vi.fn(async () => ({ suggestion: "steer" })) as unknown as OwnerRequest;
    const onSend = (mode: ConversationSendMode, task: string) => send(mode, task);
    const bubble = <ConversationSuggestion request={request} task="task-1" prompt="draft" user="owner" defaultMode="queue" onSend={onSend} />;
    await act(async () => root.render(<ConversationComposer prompt="draft" task="task-1" mode="queue" enterToSend reviewDraft={false} textareaRef={createRef()} commandMenuRef={createRef()} suggestions={[]} commands={[]} context={null} suggestion={bubble} onChange={() => {}} onSend={send} onReview={() => {}} />));
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(send).not.toHaveBeenCalled();
    await act(async () => element.querySelector<HTMLButtonElement>(".conversation-suggestion button")!.click());
    expect(send).toHaveBeenCalledExactlyOnceWith("steer", "task-1");
    await act(async () => root.unmount());
  });
});
