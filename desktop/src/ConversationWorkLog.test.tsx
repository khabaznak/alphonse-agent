// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { ConversationWorkLog } from "./ConversationWorkLog";
import { ConversationFailure } from "./ConversationFailure";
import type { OwnerRequest } from "./ownerConversation";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("shows recorded activity during work and keeps it collapsible afterward", async () => {
  const container = document.createElement("div"); document.body.appendChild(container);
  const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "blocked", failure_code: "time_limit", notice: "May be incomplete.",
    events: [{ label: "Searched the web", outcome: "No results returned" }] })) as unknown as OwnerRequest;
  await act(async () => { root.render(<ConversationWorkLog request={request} messageId={"a".repeat(32)} active />); });
  expect(container.textContent).toContain("Searched the web · No results returned");
  expect(container.textContent).toContain("time limit");
  expect(request).toHaveBeenCalledWith("experimental_owner_work_log", { message_id: "a".repeat(32) });
  await act(async () => { root.unmount(); }); container.remove();
});

it("offers only explicitly available recovery controls", () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const review = vi.fn();
  act(() => root.render(<ConversationFailure message="Stopped" dismissed={false} onDismissedChange={() => {}} onReviewRecovery={review} />));
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === "Review unfinished work");
  expect(button).toBeTruthy(); act(() => button!.click()); expect(review).toHaveBeenCalledOnce();
  expect(container.textContent).not.toContain("Open model settings");
  act(() => root.unmount());
});
