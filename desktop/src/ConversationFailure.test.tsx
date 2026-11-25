// @vitest-environment jsdom
import { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { ConversationFailure } from "./ConversationFailure";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("collapses and restores details without resolving the failed request", async () => {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const changed = vi.fn();
  function Example() {
    const [dismissed, setDismissed] = useState(false);
    return <ConversationFailure message="The source was unavailable." dismissed={dismissed} onDismissedChange={(value) => { changed(value); setDismissed(value); }} />;
  }
  await act(async () => root.render(<Example />));
  expect(container.textContent).toContain("The source was unavailable.");
  await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Dismiss notice")!.click());
  expect(container.textContent).not.toContain("The source was unavailable.");
  expect(changed).toHaveBeenCalledWith(true);
  await act(async () => container.querySelector("button")!.click());
  expect(container.textContent).toContain("The source was unavailable.");
  expect(changed).toHaveBeenCalledWith(false);
  await act(async () => root.unmount());
  container.remove();
});

it("keeps technical recovery and conditional effects guidance behind a disclosure", async () => {
  const container = document.createElement("div"); document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => root.render(<ConversationFailure message="The specific reason was not recorded." dismissed={false} onDismissedChange={() => {}} onReviewRecovery={() => {}} onViewWorkLog={() => {}} onDraftQuestion={() => {}} hasWorkLog />));
  const details = container.querySelector("details")!;
  expect(details.open).toBe(false);
  expect(details.querySelector("summary")?.textContent).toBe("More about this request");
  expect(details.textContent).toContain("If the request involved a change");
  expect(details.textContent).toContain("Review request status");
  expect(container.querySelector("section > .conversation-failure-actions")?.textContent).not.toContain("Review request status");
  await act(async () => root.unmount()); container.remove();
});
