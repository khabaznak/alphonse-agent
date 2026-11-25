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
