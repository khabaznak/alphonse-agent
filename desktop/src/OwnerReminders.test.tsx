import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { dismissOfferedReminder, OwnerReminderOffer, OwnerReminderRecovery, OwnerReminders, reminderStateMessage, takeVisibleReminder } from "./OwnerReminders";
import { readPendingReminder, reconcilePendingReminder, submitPendingReminder } from "./ownerReminderSubmission";
import type { OwnerRequest } from "./ownerConversation";

describe("owner Desktop reminders", () => {
  it("does not take or render reminders when the Desktop owner is inactive", async () => {
    const request = vi.fn();
    expect(await takeVisibleReminder(request as OwnerRequest, false)).toBeNull();
    expect(renderToStaticMarkup(<OwnerReminders request={request as OwnerRequest} active={false} ownerId="owner" />)).toBe("");
    expect(request).not.toHaveBeenCalled();
  });

  it("keeps the reminder surface active outside Conversation while hiding its management form", () => {
    const request = vi.fn();
    const html = renderToStaticMarkup(<OwnerReminders request={request as OwnerRequest} active ownerId="owner" showManagement={false} />);
    expect(html).toContain('aria-label="Desktop reminders"');
    expect(html).toContain('<div hidden="">');
    expect(html).toContain('Create a Desktop reminder');
  });

  it("requires explicit dismiss before acknowledging an offered reminder", async () => {
    const offered = { id: "id", message: "Fixture", due_at: "2026-11-15T12:00:00+00:00", acknowledgement_token: "secret" };
    const request = vi.fn().mockResolvedValueOnce({ reminder: offered }).mockResolvedValueOnce({ status: "client_acknowledged" });
    expect(await takeVisibleReminder(request as OwnerRequest, true)).toEqual(offered);
    expect(request).toHaveBeenCalledTimes(1);
    expect(request).toHaveBeenCalledWith("owner_reminder_take", {});
    const html = renderToStaticMarkup(<OwnerReminderOffer message={offered.message} onDismiss={() => {}} />);
    expect(html).toContain("Fixture");
    expect(html).toContain("Dismiss");
    expect(html).not.toContain("secret");
    expect(request).toHaveBeenCalledTimes(1);
    await dismissOfferedReminder(request as OwnerRequest, offered);
    expect(request).toHaveBeenLastCalledWith("owner_reminder_ack", { id: "id", token: "secret" });
  });

  it("shows recovery as unconfirmed with explicit review and no automatic repeat", () => {
    const resolve = vi.fn();
    const cancel = vi.fn();
    const html = renderToStaticMarkup(<OwnerReminderRecovery reminders={[
      { id: "id", message: "Fixture", due_at: "2026-11-15T12:00:00+00:00", state: "offered" },
    ]} onResolve={resolve} onCancel={cancel} />);
    expect(html).toContain("Pickup unconfirmed");
    expect(html).toContain("not be automatically repeated");
    expect(html).toContain("Mark reviewed");
    expect(resolve).not.toHaveBeenCalled();
    expect(cancel).not.toHaveBeenCalled();
  });

  it("preserves the exact creation request across a lost response and restart without replay", async () => {
    let saved: string | null = null;
    const storage = { getItem: vi.fn(() => saved), setItem: vi.fn((_key: string, value: string) => { saved = value; }),
      removeItem: vi.fn(() => { saved = null; }) };
    const item = { message: "Private fixture", due_at: "2026-11-15T12:00:00.000Z", request_id: "a".repeat(32) };
    const request = vi.fn().mockRejectedValueOnce(new Error("lost response"));
    await expect(submitPendingReminder(request as OwnerRequest, storage, "owner", item)).rejects.toThrow();
    const restored = readPendingReminder(storage, "owner");
    expect(restored).toEqual(item);
    expect(request).toHaveBeenCalledTimes(1); // Loading never repeats the mutation.
    request.mockResolvedValueOnce({ reminder: { id: "b".repeat(32), message: item.message, due_at: "2026-11-15T12:00:00+00:00", state: "cancelled" } });
    expect(await reconcilePendingReminder(request as OwnerRequest, storage, "owner", restored!)).toBe("cancelled");
    expect(reminderStateMessage("cancelled")).toContain("will not appear");
    expect(reminderStateMessage("acknowledged")).toContain("will not repeat");
    expect(request).toHaveBeenLastCalledWith("owner_reminder_lookup", { request_id: item.request_id });
    expect(saved).toBeNull();
  });

  it("never dispatches if the pending request cannot be persisted", async () => {
    const request = vi.fn();
    const storage = { getItem: () => null, setItem: () => { throw new Error("storage unavailable"); }, removeItem: () => {} };
    await expect(submitPendingReminder(request as OwnerRequest, storage, "owner", {
      message: "Fixture", due_at: "2026-11-15T12:00:00.000Z", request_id: "a".repeat(32),
    })).rejects.toThrow("storage unavailable");
    expect(request).not.toHaveBeenCalled();
  });
});
