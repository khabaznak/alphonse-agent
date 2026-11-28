// @vitest-environment jsdom
import { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";

const management = vi.hoisted(() => vi.fn());
vi.mock("./api", () => ({ optionalManagement: management }));
import { CalendarApprovalCard, CalendarEditReviews, useCalendarApprovalHistory, type CalendarApproval } from "./CalendarEditReviews";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ordinaryReview = {
  review_token: "proposal", expires_at: "2099-01-01T00:00:00Z", calendar_id: "selected",
  event_id: "fixture", event_title: "Fixture", event_start: "2099-01-15", recurrence_scope: "instance",
  before: { location: "Old room" }, after: { location: "New room" },
};

const inviteReview = {
  ...ordinaryReview,
  review_token: "review_cq70g",
  event_title: "Pottery handoff",
  before: { description: "Meet by the side entrance.\nBring the receipt.", location: "Workshop A" },
  after: { description: "Meet by the shelving in Workshop B.\nBring the receipt.", location: "Workshop B" },
  guests: {
    current: [{ name: "Riley Chen", email: "riley.chen@example.com" }],
    added: [{ name: "Maya Santos", email: "maya.santos@example.com" }],
    notification_level: "ALL",
  },
};

const savedReceipt = {
  id: "saved-edit-1", approved_at: "2026-11-29T02:03:00+00:00",
  event_title: inviteReview.event_title, event_start: "2026-12-01T09:15:00-06:00",
  event_end: "2026-12-01T10:00:00-06:00", recurrence_scope: "instance",
  changes: inviteReview.after, invited_guests: inviteReview.guests.added, notification_level: "ALL",
} as const;

async function mountReviews() {
  const element = document.body.appendChild(document.createElement("div"));
  const root = createRoot(element);
  await act(async () => root.render(<CalendarEditReviews active />));
  return {
    element,
    button: (label: string) => Array.from(element.querySelectorAll("button")).find(button => button.textContent?.trim() === label)!,
    async unmount() {
      await act(async () => root.unmount());
      element.remove();
    },
  };
}

it("shows exact Calendar changes and commits only after a click", async () => {
  management.mockReset();
  management.mockImplementation(async (action: string) => action === "calendar_edit_reviews" ? { reviews: [ordinaryReview] } : { status: "conflict" });
  const { element, button, unmount } = await mountReviews();
  try {
    expect(element.textContent).toContain("Current");
    expect(element.textContent).toContain("Old room");
    expect(element.textContent).toContain("Proposed");
    expect(element.textContent).toContain("New room");
    expect(element.textContent).toContain("This occurrence");
    expect(element.querySelector("details")?.open).toBe(false);
    expect(element.querySelector("details")?.textContent).toContain("selected");
    expect(element.textContent).toMatch(/overwrit/i);
    expect(element.textContent).not.toContain("Approve and notify");
    expect(management.mock.calls.every(([action]) => action === "calendar_edit_reviews")).toBe(true);
    const approve = button("Approve and update");
    expect(approve.closest(".calendar-edit-details")).toBeNull();
    approve.focus();
    await act(async () => approve.click());
    expect(management).toHaveBeenLastCalledWith("calendar_edit_commit", { review_token: "proposal" });
    expect(element.textContent).toContain("The event changed before this update");
    expect(element.textContent).not.toContain("saved and verified");
    expect(document.activeElement).toBe(element.querySelector('[role="status"]'));
  } finally {
    await unmount();
  }
});

it.each([
  { readback: { status: "updated", guests_verified: true }, outcome: /confirm.*new guests/i },
  { readback: { status: "updated" }, outcome: /outcome is unknown/i },
])("discloses the exact invite changes and reports the outcome from $readback", async ({ readback, outcome }) => {
  management.mockReset();
  management.mockImplementation(async (action: string) => action === "calendar_edit_reviews"
    ? { reviews: [inviteReview] }
    : readback);
  const { element, button, unmount } = await mountReviews();
  try {
    const description = Array.from(element.querySelectorAll("h4")).find(heading => heading.textContent === "Description");
    expect(Array.from(description!.parentElement!.querySelectorAll("dt, dd"), node => node.textContent))
      .toEqual(["Current", inviteReview.before.description, "Proposed", inviteReview.after.description]);
    const guests = element.querySelector('[aria-label="Guests"]');
    expect(guests?.textContent).toMatch(/Riley Chen.*riley\.chen@example\.com.*Already invited/);
    expect(guests?.textContent).toMatch(/Maya Santos.*maya\.santos@example\.com.*New guest/);
    expect(element.textContent).toMatch(/(?:email|notify).*?(?:both|all).*?guests/i);
    expect(element.textContent).not.toContain(inviteReview.review_token);
    expect(button("Approve and update")).toBeUndefined();
    expect(management.mock.calls.every(([action]) => action === "calendar_edit_reviews")).toBe(true);
    await act(async () => button("Approve and notify").click());
    expect(management).toHaveBeenLastCalledWith("calendar_edit_commit", { review_token: inviteReview.review_token, notification_level: "ALL" });
    expect(element.querySelector('[role="status"]')?.textContent).toMatch(outcome);
    expect(element.querySelector('[role="status"]')?.textContent).not.toMatch(/delivered|inboxes/i);
  } finally {
    await unmount();
  }
});

it("reports an expired review separately from an unknown provider outcome", async () => {
  management.mockReset();
  management.mockImplementation(async (action: string) => {
    if (action === "calendar_edit_reviews") return { reviews: [inviteReview] };
    throw new Error("calendar_edit_review_expired_or_denied");
  });
  const { element, button, unmount } = await mountReviews();
  try {
    await act(async () => button("Approve and notify").click());
    const status = element.querySelector('[role="status"]')?.textContent;
    expect(status).toMatch(/expired|review.*(?:unavailable|no longer available)/i);
    expect(status).toMatch(/new|fresh/i);
    expect(status).not.toMatch(/outcome is unknown|confirmed/i);
  } finally {
    await unmount();
  }
});

it("offers a keyboard-focusable description jump only while the field comparison overflows", async () => {
  let measure = () => {};
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { measure = callback; }
    observe() {}
    disconnect() {}
  });
  management.mockReset();
  management.mockResolvedValue({ reviews: [inviteReview] });
  const { element, button, unmount } = await mountReviews();
  try {
    expect(button("Scroll to description")).toBeUndefined();
    const region = element.querySelector<HTMLDivElement>('[aria-label="Proposed field changes"]')!;
    const description = region.querySelector(".calendar-review-description")!;
    let height = 120;
    Object.defineProperties(region, {
      clientHeight: { get: () => height },
      scrollHeight: { get: () => 240 },
    });
    vi.spyOn(region, "getBoundingClientRect").mockReturnValue(new DOMRect(0, 100, 320, 120));
    vi.spyOn(description, "getBoundingClientRect").mockReturnValue(new DOMRect(0, 185, 320, 100));
    await act(async () => measure());

    await act(async () => button("Scroll to description").click());
    expect(region.scrollTop).toBe(85);
    expect(document.activeElement).toBe(region);
    expect(management.mock.calls.every(([action]) => action === "calendar_edit_reviews")).toBe(true);

    region.scrollTop = 0;
    vi.spyOn(description.previousElementSibling!, "getBoundingClientRect")
      .mockReturnValue(new DOMRect(0, 100, 320, 147));
    await act(async () => measure());
    expect(button("Scroll to description")).toBeUndefined();
    await act(async () => button("Scroll field changes").click());
    expect(region.scrollTop).toBe(96);

    height = 240;
    await act(async () => measure());
    expect(button("Scroll to description")).toBeUndefined();
    expect(button("Scroll field changes")).toBeUndefined();
  } finally {
    await unmount();
    vi.unstubAllGlobals();
  }
});

it("keeps the approved invite visibly frozen until the provider responds even if polling drops it", async () => {
  vi.useFakeTimers();
  management.mockReset();
  let listed = [inviteReview, ordinaryReview];
  let settle!: (result: { status: string }) => void;
  management.mockImplementation((action: string) => action === "calendar_edit_reviews"
    ? Promise.resolve({ reviews: listed })
    : new Promise(resolve => { settle = resolve; }));
  const { element, button, unmount } = await mountReviews();
  try {
    const approve = button("Approve and notify");
    approve.focus();
    await act(async () => { approve.click(); approve.click(); });
    expect(approve.disabled).toBe(true);
    expect(button("Dismiss").disabled).toBe(true);
    expect(element.querySelector("select")?.disabled).toBe(true);

    listed = [ordinaryReview];
    await act(async () => vi.advanceTimersByTimeAsync(4_000));
    expect(element.textContent).toContain("maya.santos@example.com");
    expect(button("Dismiss").disabled).toBe(true);
    expect(button("Approve and update")).toBeUndefined();
    expect(management.mock.calls.filter(([action]) => action !== "calendar_edit_reviews"))
      .toEqual([["calendar_edit_commit", { review_token: inviteReview.review_token, notification_level: "ALL" }]]);

    await act(async () => settle({ status: "unknown" }));
    const status = element.querySelector('[role="status"]');
    expect(status?.textContent).toMatch(/unknown|check.*google calendar/i);
    expect(document.activeElement).toBe(status);
    expect(button("Approve and update").disabled).toBe(false);
  } finally {
    await unmount();
    vi.useRealTimers();
  }
});

it("preserves selector focus when moving between pending reviews", async () => {
  management.mockReset();
  management.mockResolvedValue({ reviews: [inviteReview, ordinaryReview] });
  const { element, unmount } = await mountReviews();
  try {
    const picker = element.querySelector("select")!;
    picker.focus();
    await act(async () => {
      picker.value = "1";
      picker.dispatchEvent(new Event("change", { bubbles: true }));
    });
    const nextPicker = element.querySelector("select");
    expect(nextPicker?.value).toBe("1");
    expect(document.activeElement).toBe(nextPicker);
    expect(element.querySelector("article h3")?.textContent).toBe(ordinaryReview.event_title);
  } finally {
    await unmount();
  }
});

it("reveals a verified read-only card and restores it on the approval day after remount", async () => {
  management.mockReset();
  management.mockImplementation(async (action: string, values: Record<string, unknown>) => {
    if (action === "calendar_edit_reviews") return { reviews: [inviteReview] };
    if (action === "calendar_edit_receipts") {
      return { receipts: applied && values.local_date === "2026-11-28" ? [savedReceipt] : [] };
    }
    applied = true;
    return { status: "updated", guests_verified: true, receipt: savedReceipt };
  });
  const element = document.body.appendChild(document.createElement("div"));
  let applied = false;
  const root = createRoot(element);
  const timezone = "America/Mexico_City";
  function History({ day, latest }: { day: string; latest?: CalendarApproval }) {
    const history = useCalendarApprovalHistory({ active: true, owner: "owner", timezone, day, latest });
    return history.receipts.map(receipt => <CalendarApprovalCard key={receipt.id} receipt={receipt} timezone={timezone} latest={history.latest} />);
  }
  function Conversation() {
    const [latest, setLatest] = useState<CalendarApproval>();
    return <>
      <div id="timeline"><History day="2026-11-28" latest={latest} /></div>
      <CalendarEditReviews active onSaved={setLatest} />
      <textarea aria-label="Draft" defaultValue="Keep this draft" />
    </>;
  }
  try {
    await act(async () => root.render(<Conversation />));
    const approve = Array.from(element.querySelectorAll("button")).find(btn => btn.textContent === "Approve and notify")!;
    approve.focus();
    await act(async () => approve.click());
    const card = element.querySelector("#timeline article");
    expect(element.querySelectorAll("#timeline article")).toHaveLength(1);
    expect(document.activeElement).toBe(card);
    expect(card?.textContent).toMatch(/Approved and saved.*Pottery handoff.*This occurrence only/);
    expect(card?.textContent).toContain(savedReceipt.changes.description);
    expect(card?.textContent).toContain(savedReceipt.changes.location);
    expect(card?.textContent).toContain(savedReceipt.invited_guests[0].email);
    expect(card?.textContent).not.toContain(inviteReview.guests.current[0].email);
    expect(card?.textContent).not.toMatch(/deliver|received/i);
    expect(card?.querySelector("button, input, a, textarea")).toBeNull();
    expect(element.querySelector("textarea")?.value).toBe("Keep this draft");

    await act(async () => root.render(<History day="2026-12-01" />));
    expect(element.querySelector("article")).toBeNull();
    await act(async () => root.render(<History day="2026-11-28" />));
    expect(element.querySelector("article")?.textContent).toContain("Pottery handoff");
  } finally {
    await act(async () => root.unmount());
    element.remove();
  }
});
