// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OwnerFollowups } from "./OwnerFollowups";
import { OwnerSavedDrafts, type OwnerSavedDraft } from "./OwnerSavedDrafts";
import { OwnerSteering } from "./OwnerSteering";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const containers: HTMLElement[] = [];
afterEach(() => { containers.forEach(c => c.remove()); containers.length = 0; });

function mount() {
  const container = document.createElement("div"); document.body.appendChild(container); containers.push(container);
  return { container, root: createRoot(container) };
}

describe("interactive follow-up controls", () => {
  it("requires checking a recovered unresolved submission before deletion or restoration", async () => {
    const { container, root } = mount();
    const check = vi.fn(), restore = vi.fn(), remove = vi.fn();
    const saved: OwnerSavedDraft = {id: "a".repeat(32), active_message_id: "b".repeat(32), prompt: "Add apples", resolution: "unresolved"};
    const render = (item: OwnerSavedDraft, prompt = "") => <OwnerSavedDrafts items={[item]} prompt={prompt} onCheck={check} onRestore={restore} onDelete={remove} />;
    await act(async () => root.render(render(saved)));
    expect(container.textContent).toContain("Submission needs checking");
    expect([...container.querySelectorAll("button")].map(b => b.textContent)).toEqual(["Check submission"]);
    await act(async () => container.querySelector("button")!.click());
    expect(check).toHaveBeenCalledExactlyOnceWith(saved);
    await act(async () => root.render(render({...saved, resolution: "not_submitted"}, "existing draft")));
    const buttons = [...container.querySelectorAll("button")];
    expect(buttons[0].disabled).toBe(true);
    await act(async () => buttons[1].click());
    expect(remove).toHaveBeenCalledOnce();
    expect(restore).not.toHaveBeenCalled();
    await act(async () => root.unmount());
  });

  it("only offers removal for a held follow-up", async () => {
    const {container, root} = mount();
    const remove = vi.fn();
    await act(async () => root.render(<OwnerFollowups items={[
      {id: "first", prompt: "Add apples", status: "queued", can_remove: true},
      {id: "second", prompt: "Make them green", status: "released", can_remove: false},
    ]} onRemove={remove} />));
    expect(container.textContent).toContain("Queued");
    expect(container.textContent).toContain("Processing");
    expect(container.querySelectorAll("button")).toHaveLength(1);
    await act(async () => container.querySelector("button")!.click());
    expect(remove).toHaveBeenCalledExactlyOnceWith("first");
    await act(async () => root.unmount());
  });
  it("only offers steering removal when the daemon marks it safe", async () => {
    const {container, root} = mount();
    const remove = vi.fn();
    await act(async () => root.render(<OwnerSteering items={[
      {id: "busy", prompt: "Use green apples", status: "accepted", can_remove: false, remove_blocked_reason: "dependent_followups"},
      {id: "done", prompt: "Use pears", status: "included_in_planning", can_remove: true},
    ]} onRemove={remove} />));
    expect(container.textContent).toContain("Included in planning");
    expect(container.textContent).toContain("Review dependent follow-ups before clearing.");
    expect(container.querySelectorAll("button")).toHaveLength(1);
    await act(async () => container.querySelector("button")!.click());
    expect(remove).toHaveBeenCalledExactlyOnceWith("done");
    await act(async () => root.unmount());
  });

});
