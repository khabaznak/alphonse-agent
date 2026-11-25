// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationSuggestion } from "./ConversationSuggestion";
import type { OwnerRequest } from "./ownerConversation";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
describe("conversation suggestion", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());
  it.each([["queue", "steer", "update to the current task"], ["steer", "queue", "separate follow-up"]] as const)("shows the opposing action for %s default", async (defaultMode, suggested, text) => {
    const element = document.createElement("div"); const root = createRoot(element);
    const onSend = vi.fn();
    const request = vi.fn(async () => ({ suggestion: suggested })) as unknown as OwnerRequest;
    await act(async () => root.render(<ConversationSuggestion request={request} task="task" prompt="new text" user="owner" defaultMode={defaultMode} onSend={onSend} />));
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(element.textContent).toContain(text);
    expect(onSend).not.toHaveBeenCalled();
    await act(async () => element.querySelector("button")!.click());
    expect(onSend).toHaveBeenCalledWith(suggested, "task");
    await act(async () => root.unmount());
  });
  it("drops stale responses when draft or task changes", async () => {
    const element = document.createElement("div"); const root = createRoot(element);
    let resolve!: (value: {suggestion: string}) => void;
    const request = vi.fn(() => new Promise((done) => { resolve = done; })) as unknown as OwnerRequest;
    const render = (task: string, prompt: string) => <ConversationSuggestion request={request} task={task} prompt={prompt} user="owner" defaultMode="queue" onSend={() => {}} />;
    await act(async () => root.render(render("task", "first")));
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    await act(async () => root.render(render("other", "changed")));
    await act(async () => resolve({suggestion: "steer"}));
    expect(element.textContent).toBe("");
    await act(async () => root.unmount());
  });
  it("does not show a suggestion that matches the preference", async () => {
    const element = document.createElement("div"); const root = createRoot(element);
    const request = vi.fn(async () => ({ suggestion: "queue" })) as unknown as OwnerRequest;
    await act(async () => root.render(<ConversationSuggestion request={request} task="task" prompt="test" user="owner" defaultMode="queue" onSend={() => {}} />));
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(element.textContent).toBe("");
    await act(async () => root.unmount());
  });
});
