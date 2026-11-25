// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { OwnerRecovery } from "./OwnerRecoveryPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const id = "a".repeat(32);
async function openEarlierRequests(element: HTMLElement) {
  const details = [...element.querySelectorAll("details")].find(item => item.querySelector("summary")?.textContent?.startsWith("Other earlier requests"));
  expect(details?.open).toBe(false);
  await act(async () => { details!.open = true; });
}
it("confirms stopping the specific eligible request and allows keeping it", async () => {
  const element = document.createElement("div"), root = createRoot(element), stop = vi.fn();
  const item = { message_id: id, prompt: "A prior request", status: "unavailable", can_abandon: true };
  await act(async () => root.render(<OwnerRecovery items={[item]} stopping="" onStop={stop} onShow={() => {}} />));
  await openEarlierRequests(element);
  expect(element.querySelector(`#owner-recovery-${id}`)).not.toBeNull();
  await act(async () => element.querySelector<HTMLButtonElement>('button[aria-label="Stop request 1: A prior request"]')!.click());
  expect(stop).not.toHaveBeenCalled();
  expect(element.textContent).toContain("anything that may already have happened is not undone");
  await act(async () => element.querySelector<HTMLButtonElement>('button[aria-label="Keep request 1: A prior request"]')!.click());
  expect(stop).not.toHaveBeenCalled();
  await act(async () => element.querySelector<HTMLButtonElement>('button[aria-label="Stop request 1: A prior request"]')!.click());
  await act(async () => element.querySelector<HTMLButtonElement>('button[aria-label="Confirm stop request 1: A prior request"]')!.click());
  expect(stop).toHaveBeenCalledExactlyOnceWith(id);
  await act(async () => root.unmount());
});
it("withdraws the stop confirmation if eligibility changes", async () => {
  const element = document.createElement("div"), root = createRoot(element), stop = vi.fn();
  const render = (can_abandon: boolean) => <OwnerRecovery items={[{ message_id: id, prompt: "A prior request", status: "unavailable", can_abandon }]} stopping="" onStop={stop} onShow={() => {}} />;
  await act(async () => root.render(render(true)));
  await openEarlierRequests(element);
  await act(async () => element.querySelector<HTMLButtonElement>("button")!.click());
  await act(async () => root.render(render(false)));
  expect(element.textContent).toContain("No stop action is available here");
  expect(element.querySelector("button")).toBeNull();
  expect(stop).not.toHaveBeenCalled();
  await act(async () => root.unmount());
});

it("distinguishes actions for requests with identical excerpts", async () => {
  const element = document.createElement("div"), root = createRoot(element), stop = vi.fn();
  const items = ["a", "b"].map(value => ({ message_id: value.repeat(32), prompt: "The same request", status: "unavailable", can_abandon: true }));
  await act(async () => root.render(<OwnerRecovery items={items} stopping="" onStop={stop} onShow={() => {}} />));
  await openEarlierRequests(element);
  const buttons = [...element.querySelectorAll("button")];
  expect(buttons.map(button => button.getAttribute("aria-label"))).toEqual([
    "Stop request 1: The same request", "Stop request 2: The same request",
  ]);
  await act(async () => buttons[1].click());
  const confirm = element.querySelector<HTMLButtonElement>('button[aria-label="Confirm stop request 2: The same request"]')!;
  await act(async () => confirm.click());
  expect(stop).toHaveBeenCalledExactlyOnceWith("b".repeat(32));
  await act(async () => root.unmount());
});

it("moves focus into confirmation and back to Stop when the user keeps the request", async () => {
  const element = document.createElement("div"); document.body.appendChild(element);
  const root = createRoot(element), stop = vi.fn();
  const item = { message_id: id, prompt: "A prior request", status: "unavailable", can_abandon: true };
  await act(async () => root.render(<OwnerRecovery items={[item]} stopping="" onStop={stop} onShow={() => {}} />));
  await openEarlierRequests(element);
  const initial = element.querySelector<HTMLButtonElement>('button[aria-label="Stop request 1: A prior request"]')!;
  initial.focus();
  await act(async () => initial.click());
  expect(document.activeElement?.textContent).toBe("Stop this request?");
  await act(async () => element.querySelector<HTMLButtonElement>('button[aria-label="Keep request 1: A prior request"]')!.click());
  expect(document.activeElement).toBe(element.querySelector('button[aria-label="Stop request 1: A prior request"]'));
  expect(stop).not.toHaveBeenCalled();
  await act(async () => root.unmount()); element.remove();
});

it("moves focus to the request when stop eligibility disappears during confirmation", async () => {
  const element = document.createElement("div"); document.body.appendChild(element);
  const root = createRoot(element), stop = vi.fn();
  const render = (can_abandon: boolean) => <OwnerRecovery items={[{ message_id: id, prompt: "A prior request", status: "unavailable", can_abandon }]} stopping="" onStop={stop} onShow={() => {}} />;
  await act(async () => root.render(render(true)));
  await openEarlierRequests(element);
  await act(async () => element.querySelector<HTMLButtonElement>("button")!.click());
  expect(document.activeElement?.textContent).toBe("Stop this request?");
  await act(async () => root.render(render(false)));
  expect(document.activeElement).toBe(element.querySelector(`#owner-recovery-${id}`));
  expect(element.textContent).toContain("No stop action is available here");
  expect(stop).not.toHaveBeenCalled();
  await act(async () => root.render(render(true)));
  expect(element.querySelector('[role="group"]')).toBeNull();
  await act(async () => root.unmount()); element.remove();
});

it("withholds stale stop and answer actions until requests are checked again", async () => {
  const element = document.createElement("div"); document.body.appendChild(element);
  const root = createRoot(element), stop = vi.fn(), show = vi.fn();
  const items = [
    { message_id: id, prompt: "Earlier request", status: "unavailable", can_abandon: true },
    { message_id: "b".repeat(32), prompt: "Ready answer", status: "ready", can_abandon: false },
  ];
  const render = (actionsEnabled: boolean) => <OwnerRecovery items={items} stopping="" actionsEnabled={actionsEnabled} onStop={stop} onShow={show} />;
  await act(async () => root.render(render(true)));
  await openEarlierRequests(element);
  await act(async () => element.querySelector<HTMLButtonElement>('button[aria-label^="Stop request 1"]')!.click());
  expect(document.activeElement?.textContent).toBe("Stop this request?");
  await act(async () => root.render(render(false)));
  expect(element.querySelectorAll("button")).toHaveLength(0);
  expect(element.textContent?.match(/Check requests again before acting/g)).toHaveLength(2);
  expect(document.activeElement).toBe(element.querySelector(`#owner-recovery-${id}`));
  expect(stop).not.toHaveBeenCalled(); expect(show).not.toHaveBeenCalled();
  await act(async () => root.render(render(true)));
  expect(element.querySelector('[role="group"]')).toBeNull();
  expect(element.querySelector('button[aria-label^="Show answer for request 2"]')).not.toBeNull();
  await act(async () => root.unmount()); element.remove();
});
