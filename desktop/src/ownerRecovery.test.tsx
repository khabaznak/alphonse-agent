import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { OwnerRecovery } from "./OwnerRecoveryPanel";
import { abandonOwnerRequest, listOwnerRecovery } from "./ownerRecovery";
import type { OwnerRequest } from "./ownerConversation";

const id = "a".repeat(32);

describe("owner request recovery", () => {
  it("lists server-owned requests without taking or stopping anything", async () => {
    const items = [{ message_id: id, prompt: "Find my note", status: "started_unconfirmed", can_abandon: true }];
    const request = vi.fn(async () => ({ items }));
    expect(await listOwnerRecovery(request as OwnerRequest)).toEqual(items);
    expect(request).toHaveBeenCalledOnce();
    expect(request).toHaveBeenCalledWith("experimental_owner_recovery_list", {});
  });

  it("requires an explicit stop action and carries no claim or token from the client", async () => {
    const request = vi.fn(async () => ({ status: "abandoned", delivery_status: "unconfirmed" }));
    expect(await abandonOwnerRequest(request as OwnerRequest, id)).toBe("unconfirmed");
    expect(request).toHaveBeenCalledExactlyOnceWith("experimental_owner_abandon", { message_id: id });
    await expect(abandonOwnerRequest(request as OwnerRequest, "wrong")).rejects.toThrow();
    expect(request).toHaveBeenCalledOnce();
  });

  it("explains uncertainty and disables stopping an active request", () => {
    const html = renderToStaticMarkup(<OwnerRecovery items={[
      { message_id: id, prompt: "Find my note", status: "pending", can_abandon: false },
    ]} stopping="" onStop={() => {}} onShow={() => {}} />);
    expect(html).toContain("not retried automatically");
    expect(html).toContain("may already have been shown");
    expect(html).toContain("disabled");
    expect(html).toContain("Find my note");
  });

  it("offers an explicit show action for a listed ready answer", () => {
    const html = renderToStaticMarkup(<OwnerRecovery items={[
      { message_id: id, prompt: "Find my note", status: "ready", can_abandon: false },
    ]} stopping="" onStop={() => {}} onShow={() => {}} />);
    expect(html).toContain("Show answer");
    expect(html).toContain("An answer is ready");
  });

  it("does not offer a stale ready answer after its claim expired", () => {
    const html = renderToStaticMarkup(<OwnerRecovery items={[
      { message_id: id, prompt: "Find my note", status: "ready", can_abandon: true },
    ]} stopping="" onStop={() => {}} onShow={() => {}} />);
    expect(html).toContain("request expired");
    expect(html).not.toContain("Show answer");
  });

  it("shows an unresolved action without offering replay or stop", () => {
    const html = renderToStaticMarkup(<OwnerRecovery items={[
      { message_id: id, prompt: "Synthetic request", status: "action_unconfirmed", can_abandon: false },
    ]} stopping="" onStop={() => {}} onShow={() => {}} />);
    expect(html).toContain("An action may have run; its effects are unconfirmed");
    expect(html).toContain("check the result before trying again");
    expect(html).not.toContain("Show answer");
    expect(html).toContain("disabled");
  });

  it("shows a committed cancellation and possible earlier execution", () => {
    const html = renderToStaticMarkup(<OwnerRecovery items={[{
      message_id: id, prompt: "Cancel the check", status: "effect_committed", can_abandon: false,
      committed_effect: { kind: "scheduled_task_cancelled", schedule_id: "fixture", name: "Pantry", execution_may_have_started: true },
    }]} stopping="" onStop={() => {}} onShow={() => {}} />);
    expect(html).toContain("scheduled task was cancelled");
    expect(html).toContain("may already have started");
    expect(html).toContain("disabled");
  });

  it("rejects malformed discovery results instead of offering a stop", async () => {
    const request = vi.fn(async () => ({ items: [{ message_id: id, prompt: "x", status: "pending", can_abandon: "yes" }] }));
    await expect(listOwnerRecovery(request as OwnerRequest)).rejects.toThrow("could not be checked");
  });
});
