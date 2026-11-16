import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LegacyNotificationReview, LegacyNotificationRow, legacyArchiveConfirmation } from "./LegacyNotificationReview";
import { archiveLegacyNotification, listLegacyNotifications, type LegacyNotification } from "./legacyNotifications";
import { legacyNotificationReviewMessage, type OwnerRequest } from "./ownerConversation";

const item: LegacyNotification = { outbox_message_id: "message-1", kind: "identity_resolution", status: "unconfirmed",
  message: "Synthetic old identity notice", created_at: "2026-01-01", attempt_count: 1,
  delivery_certainty: "unknown", review_token: "a".repeat(64), can_archive: true, review_restriction: "" };

describe("legacy notification review", () => {
  it("reports owner and ambiguous counts without claiming automatic redelivery", () => {
    expect(legacyNotificationReviewMessage({owner_scoped: 1, ambiguous: 1, unassigned: 0})).toContain("2 earlier notifications need review");
    expect(legacyNotificationReviewMessage({owner_scoped: 0, ambiguous: 0, unassigned: 0})).toBe("");
    expect(legacyNotificationReviewMessage(null)).toContain("could not be checked");
    expect(legacyNotificationReviewMessage({owner_scoped: -1, ambiguous: 0, unassigned: 0})).toBe("");
  });
  it("lists only on explicit review and renders no data before requesting", async () => {
    const request = vi.fn(async () => ({ items: [item] }));
    const html = renderToStaticMarkup(<LegacyNotificationReview request={request as OwnerRequest} onChanged={() => {}} />);
    expect(html).toContain("Review earlier notifications");
    expect(request).not.toHaveBeenCalled();
    expect(await listLegacyNotifications(request as OwnerRequest)).toEqual([item]);
    expect(request).toHaveBeenCalledExactlyOnceWith("legacy_notification_list", {});
  });
  it("requires an explicit exact-row archive and rejects unconfirmed responses", async () => {
    const request = vi.fn(async () => ({ status: "archived_without_resending", delivery_certainty: "unknown" }));
    await archiveLegacyNotification(request as OwnerRequest, item);
    expect(request).toHaveBeenCalledExactlyOnceWith("legacy_notification_archive", {message_id: item.outbox_message_id, review_token: item.review_token});
    await expect(archiveLegacyNotification(request as OwnerRequest, {...item, can_archive: false})).rejects.toThrow();
    await expect(archiveLegacyNotification((async () => ({ status: "unknown" })) as OwnerRequest, item)).rejects.toThrow();
  });
  it("rejects malformed list data rather than presenting an unsafe action", async () => {
    await expect(listLegacyNotifications((async () => ({items: [{...item, can_archive: "yes"}]})) as OwnerRequest)).rejects.toThrow();
    await expect(listLegacyNotifications((async () => ({items: [{...item, review_restriction: "linked_or_protected", can_archive: false}]})) as OwnerRequest)).rejects.toThrow();
    const restricted = {...item, message: "", review_restriction: "linked_or_protected", can_archive: false};
    expect(await listLegacyNotifications((async () => ({items: [restricted]})) as OwnerRequest)).toEqual([restricted]);
  });
  it("allows an opaque owner response only with its contents hidden", async () => {
    const opaque = {...item, kind: "response", message: "", review_restriction: "opaque_response"};
    expect(await listLegacyNotifications((async () => ({items: [opaque]})) as OwnerRequest)).toEqual([opaque]);
    await expect(listLegacyNotifications((async () => ({items: [{...opaque, message: "private"}]})) as OwnerRequest)).rejects.toThrow();
    await expect(listLegacyNotifications((async () => ({items: [{...opaque, kind: "finding"}]})) as OwnerRequest)).rejects.toThrow();
    const confirmation = legacyArchiveConfirmation(opaque as LegacyNotification);
    expect(confirmation).toContain("contents cannot be shown");
    expect(confirmation).toContain("will not resend");
    expect(confirmation).toContain("previous delivery remains unknown");
    expect(confirmation).not.toContain("will not be delivered");
  });
  it("distinguishes similar opaque records in review and confirmation without showing their contents", () => {
    const opaque: LegacyNotification = {...item, kind: "response", message: "", review_restriction: "opaque_response"};
    const second = {...opaque, outbox_message_id: "message-2", created_at: "2026-01-02"};
    const html = renderToStaticMarkup(<><LegacyNotificationRow item={opaque} busy={false} onArchive={() => {}} />
      <LegacyNotificationRow item={second} busy={false} onArchive={() => {}} /></>);
    expect(html).toContain("Reference: message-1");
    expect(html).toContain("Reference: message-2");
    expect(html).toContain("Recorded: 2026-01-01");
    expect(html).toContain("Recorded: 2026-01-02");
    expect(html).not.toContain(item.message);
    expect(legacyArchiveConfirmation(second)).toContain("Reference: message-2");
    expect(legacyArchiveConfirmation(second)).toContain("Recorded: 2026-01-02");
  });
});
