import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { CalendarCheck2, ChevronDown, CircleAlert, CircleCheck, Info, Mail } from "lucide-react";
import { optionalManagement } from "./api";
import { dayInZone, shiftDay } from "./agendaDates";

type CalendarGuest = { name?: string; email: string };
type GuestChanges = {
  current: CalendarGuest[];
  added: CalendarGuest[];
  notification_level: "ALL";
};
type CalendarReview = {
  review_token: string;
  event_title?: string;
  event_start?: string;
  expires_at: string;
  calendar_id: string;
  event_id: string;
  recurrence_scope: string;
  before: Record<string, unknown>;
  after: Record<string, unknown>;
  guests?: GuestChanges;
};

export type CalendarEditReceipt = {
  id: string;
  approved_at: string;
  event_title: string;
  event_start: string | null;
  event_end: string | null;
  recurrence_scope: string;
  changes: Record<string, string | null>;
  invited_guests?: CalendarGuest[];
  notification_level?: "ALL";
};
export type CalendarApproval = { receipt: CalendarEditReceipt; recorded: boolean; focus: boolean };
type ReviewResult = {
  status: string;
  reason?: string;
  guests_verified?: boolean;
  receipt?: CalendarEditReceipt;
  receipt_recorded?: boolean;
};
type Notice = { text: string; verified: boolean };

const fields: Record<string, string> = {
  summary: "Title", description: "Description", location: "Location",
  start: "Starts", end: "Ends", time_zone: "Time zone", timeZone: "Time zone",
};
const resultCopy: Record<string, string> = {
  updated: "Google Calendar saved and verified the change.",
  conflict: "The event changed before this update. Ask Alphonse to prepare a new review.",
  unavailable: "This change could not be applied. Check Calendar access and request a new review.",
  calendar_edit_google_approval_required: "Google requires an additional approval for this change. The update is blocked.",
  calendar_edit_api_disabled: "Google Calendar MCP is disabled for this connection. Enable it and request a new review.",
  calendar_edit_notification_approval_required: "Guest notifications still need to be approved. Reopen the review in the current Desktop app.",
  unknown: "The update outcome is unknown. Check the event in Google Calendar before retrying.",
};

function approvalNotice(result: ReviewResult, includesGuests: boolean): Notice {
  if (result.status !== "updated") {
    return { text: resultCopy[result.reason || ""] || resultCopy[result.status] || resultCopy.unknown, verified: false };
  }
  if (!includesGuests) return { text: resultCopy.updated, verified: true };
  if (result.guests_verified !== true) return { text: resultCopy.unknown, verified: false };
  return { text: "Alphonse read back the event and confirmed the approved changes, including the new guests.", verified: true };
}

function displayField(value: unknown, field?: string): string {
  if (typeof value === "string" && value.trim()) return value;
  if (value && typeof value === "object") {
    const time = value as { date?: string; dateTime?: string; timeZone?: string };
    if (typeof time.date === "string") return time.date;
    if (typeof time.dateTime === "string") return [time.dateTime, time.timeZone].filter(Boolean).join(" · ");
  }
  if (field === "description") return "No description set";
  if (field === "location") return "No location set";
  return "Empty";
}

function eventDate(value: string, timezone?: string, timeOnly = false) {
  const allDay = /^\d{4}-\d{2}-\d{2}$/.test(value);
  const instant = new Date(value);
  if (Number.isNaN(instant.valueOf())) return value;
  const date: Intl.DateTimeFormatOptions = { weekday: "short", month: "short", day: "numeric", year: "numeric" };
  if (allDay) return new Intl.DateTimeFormat(undefined, { ...date, timeZone: "UTC" }).format(instant);
  const format: Intl.DateTimeFormatOptions = { ...(timeOnly ? {} : date), hour: "numeric", minute: "2-digit", timeZoneName: "short" };
  try {
    return new Intl.DateTimeFormat(undefined, { ...format, timeZone: timezone }).format(instant);
  } catch (error) {
    if (error instanceof RangeError) return value;
    throw error;
  }
}

function eventScope(scope: string) {
  if (scope === "series") return "Entire recurring series";
  if (scope === "instance") return "This occurrence only";
  return "This event only";
}

function notificationText(count: number) {
  if (count === 1) return "Requests an update email for this guest.";
  if (count === 2) return "Requests update emails for both guests.";
  return `Requests update emails for all ${count} guests.`;
}

function CalendarGuests({ guests }: { guests: GuestChanges }) {
  return <section className="calendar-review-guests" aria-label="Guests">
    <h4>Guests</h4>
    <ul tabIndex={guests.current.length + guests.added.length > 2 ? 0 : undefined}>
      {guests.current.map(guest => <li key={guest.email}>
        <div>{guest.name && <span>{guest.name}</span>}<span className={guest.name ? "calendar-review-email" : undefined}>{guest.email}</span></div>
        <span className="calendar-review-guest-status">Already invited</span>
      </li>)}
      {guests.added.map(guest => <li key={guest.email}>
        <div>{guest.name && <span>{guest.name}</span>}<span className={guest.name ? "calendar-review-email" : undefined}>{guest.email}</span></div>
        <span className="calendar-review-guest-status calendar-review-guest-added">New guest</span>
      </li>)}
    </ul>
  </section>;
}

function ReceiptEventTime({ receipt, timezone }: { receipt: CalendarEditReceipt; timezone: string }) {
  const { event_start: start, event_end: end } = receipt;
  if (!start) return null;
  const allDay = /^\d{4}-\d{2}-\d{2}$/.test(start);
  const endDate = end ? new Date(end) : null;
  const startDate = new Date(start);
  const validEnd = endDate && !Number.isNaN(endDate.valueOf());
  const lastDay = allDay && end && validEnd && /^\d{4}-\d{2}-\d{2}$/.test(end) ? shiftDay(end, -1) : null;
  const endsSameDay = !allDay && validEnd && !Number.isNaN(startDate.valueOf())
    && dayInZone(timezone, startDate) === dayInZone(timezone, endDate);
  return <><time dateTime={start}>{eventDate(start, timezone)}</time>
    {allDay ? <>{lastDay && lastDay > start && <> – <time dateTime={lastDay}>{eventDate(lastDay, timezone)}</time></>} · All day</>
      : end && <> – <time dateTime={end}>{eventDate(end, timezone, Boolean(endsSameDay))}</time></>}</>;
}

export function useCalendarApprovalHistory({ active, owner, day, timezone, latest }: {
  active: boolean; owner: string; day: string; timezone: string; latest?: CalendarApproval;
}) {
  const [history, setHistory] = useState<{ scope: string; receipts: CalendarEditReceipt[]; truncated: boolean } | null>(null);
  const scope = JSON.stringify([owner, day, timezone]);
  const latestForDay = latest && dayInZone(timezone, new Date(latest.receipt.approved_at)) === day ? latest : undefined;
  const current = history?.scope === scope ? history : null;
  useEffect(() => {
    if (!active) return;
    let live = true;
    const load = async () => {
      try {
        const result = await optionalManagement<{ receipts: CalendarEditReceipt[]; truncated: boolean }>("calendar_edit_receipts", { local_date: day });
        if (live && Array.isArray(result?.receipts)) setHistory({ ...result, scope });
      } catch {
        // Keep the current day's records if the daemon becomes unavailable.
      }
    };
    void load();
    return () => { live = false; };
  }, [active, day, scope, latestForDay?.receipt.id]);
  const receipts = useMemo(() => {
    if (!active) return [];
    const stored = current?.receipts ?? [];
    return latestForDay && !stored.some(receipt => receipt.id === latestForDay.receipt.id)
      ? [...stored, latestForDay.receipt] : stored;
  }, [active, current, latestForDay]);
  return { receipts, latest: latestForDay, truncated: active && current?.truncated };
}

export function CalendarApprovalCard({ receipt, timezone, latest, onFocused }: {
  receipt: CalendarEditReceipt; timezone: string; latest?: CalendarApproval; onFocused?: (id: string) => void;
}) {
  const focusCard = useRef<HTMLElement>(null);
  const isLatest = latest?.receipt.id === receipt.id;
  const focus = isLatest && latest.focus;
  useLayoutEffect(() => {
    const card = focusCard.current;
    if (!focus || !card) return;
    card.focus({ preventScroll: true });
    card.scrollIntoView?.({ block: "start" });
    onFocused?.(receipt.id);
  }, [focus, onFocused, receipt.id]);
  return <article ref={focusCard} data-conversation-message-id={`calendar-receipt:${receipt.id}`}
    tabIndex={-1} className="calendar-approval-card" aria-label={`Approved Calendar change for ${receipt.event_title || "Untitled event"}`}>
      <header className="calendar-review-heading">
        <span className="calendar-review-symbol"><CalendarCheck2 aria-hidden="true" size={17} /></span>
        <div>
          <p className="calendar-approval-label">Approved and saved</p>
          <h3>{receipt.event_title || "Untitled event"}</h3>
          <p className="calendar-review-label"><ReceiptEventTime receipt={receipt} timezone={timezone} />
            {receipt.event_start && " · "}{eventScope(receipt.recurrence_scope)}</p>
        </div>
      </header>
      {Object.keys(receipt.changes).length > 0 && <dl className="calendar-approval-fields">
        {Object.entries(receipt.changes).filter(([field, value]) => field !== "time_zone" || value).map(([field, value]) =>
          <div key={field}><dt>{fields[field] || field}</dt><dd>{value && (field === "start" || field === "end")
            ? <time dateTime={value}>{eventDate(value, timezone)}</time> : displayField(value, field)}</dd></div>)}
      </dl>}
      {Boolean(receipt.invited_guests?.length) && <section className="calendar-approval-guests" aria-label="Newly invited guests">
        <h4>Newly invited guests</h4>
        <ul>{receipt.invited_guests?.map(guest => <li key={guest.email}>
          {guest.name && <span>{guest.name}</span>}<span className={guest.name ? "calendar-review-email" : undefined}>{guest.email}</span>
        </li>)}</ul>
        {receipt.notification_level === "ALL" && <p>Alphonse asked Google to notify all guests.</p>}
      </section>}
      <footer className="calendar-approval-footer">
        Approved <time dateTime={receipt.approved_at}>{eventDate(receipt.approved_at, timezone)}</time>
        {isLatest && !latest.recorded &&
          <p role="status">Google saved the change. Desktop could not keep this receipt in history.</p>}
      </footer>
    </article>;
}

export function CalendarEditReviews({ active, timezone, onSaved }: {
  active: boolean; timezone?: string; onSaved?: (approval: CalendarApproval) => void;
}) {
  const notificationId = useId();
  const [reviews, setReviews] = useState<CalendarReview[]>([]);
  const [selectedToken, setSelectedToken] = useState("");
  const [pending, setPending] = useState<{ review: CalendarReview; choices: CalendarReview[]; approve: boolean } | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [scrollTarget, setScrollTarget] = useState<"description" | "fields" | null>(null);
  const pendingAction = useRef("");
  const dismissed = useRef(new Set<string>());
  const listRequest = useRef(0);
  const reviewCard = useRef<HTMLElement>(null);
  const reviewPicker = useRef<HTMLSelectElement>(null);
  const restorePickerFocus = useRef(false);
  const changeRegion = useRef<HTMLDivElement>(null);
  const resultNotice = useRef<HTMLParagraphElement>(null);
  const focusResult = useRef(false);
  useLayoutEffect(() => {
    if (notice && focusResult.current) {
      focusResult.current = false;
      resultNotice.current?.focus();
    }
  }, [notice]);
  useEffect(() => {
    if (!active) { setReviews([]); return; }
    let current = true;
    const refresh = async () => {
      const request = ++listRequest.current;
      try {
        const result = await optionalManagement<{ reviews: CalendarReview[] }>("calendar_edit_reviews");
        if (current && request === listRequest.current && Array.isArray(result?.reviews)) {
          setReviews(result.reviews.filter(review => !dismissed.current.has(review.review_token)));
        }
      } catch {
        if (current && request === listRequest.current) setReviews([]);
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4_000);
    return () => { current = false; window.clearInterval(timer); };
  }, [active]);

  async function decide(review: CalendarReview, approve: boolean) {
    if (!active || pendingAction.current || dismissed.current.has(review.review_token)) return;
    const startedWithFocus = reviewCard.current?.contains(document.activeElement);
    pendingAction.current = review.review_token;
    setPending({ review, choices: reviews, approve });
    setNotice(null);
    try {
      const values = approve && review.guests
        ? { review_token: review.review_token, notification_level: review.guests.notification_level }
        : { review_token: review.review_token };
      const result = await optionalManagement<ReviewResult>(
        approve ? "calendar_edit_commit" : "calendar_edit_cancel", values,
      );
      const outcome = approve ? approvalNotice(result, Boolean(review.guests))
        : { text: result.status === "cancelled" ? "Calendar edit dismissed." : resultCopy.unknown, verified: false };
      if (outcome.verified && result.receipt && onSaved) {
        onSaved({ receipt: result.receipt, recorded: result.receipt_recorded !== false,
          focus: document.activeElement === document.body || Boolean(reviewCard.current?.contains(document.activeElement)) });
      } else {
        setNotice(outcome);
      }
    } catch (error) {
      const unavailable = error instanceof Error
        && /^(?:PermissionError: )?calendar_edit_review_expired_or_denied$/.test(error.message);
      const text = unavailable ? "This review is no longer available. Check the event before requesting a fresh review."
        : approve ? resultCopy.unknown : "The review could not be dismissed. Refresh before reviewing it again.";
      setNotice({ text, verified: false });
    } finally {
      focusResult.current = Boolean(startedWithFocus && (document.activeElement === document.body
        || reviewCard.current?.contains(document.activeElement)));
      dismissed.current.add(review.review_token);
      setReviews(current => current.filter(item => item.review_token !== review.review_token));
      pendingAction.current = "";
      setPending(null);
    }
  }

  const choices = pending?.choices ?? reviews;
  const review = pending?.review ?? choices.find(item => item.review_token === selectedToken) ?? choices[0];
  useLayoutEffect(() => {
    if (!restorePickerFocus.current) return;
    restorePickerFocus.current = false;
    reviewPicker.current?.focus({ preventScroll: true });
  }, [review?.review_token]);
  useLayoutEffect(() => {
    const region = changeRegion.current;
    const measure = () => {
      if (!region || region.scrollHeight <= region.clientHeight + 1) {
        setScrollTarget(null);
        return;
      }
      const description = region.querySelector(".calendar-review-description");
      const previousChange = description?.previousElementSibling;
      const visibleBottom = region.getBoundingClientRect().top + region.clientHeight - region.scrollTop;
      setScrollTarget(description && (!previousChange || previousChange.getBoundingClientRect().bottom <= visibleBottom)
        ? "description" : "fields");
    };
    measure();
    if (!region || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(region);
    return () => observer.disconnect();
  }, [active, review]);

  if (!active || (!review && !notice)) return null;
  const guests = review?.guests;
  const changes = review ? Object.entries(review.after)
    .filter(([field, value]) => JSON.stringify(review.before[field]) !== JSON.stringify(value))
    .sort(([left], [right]) => Number(left === "description") - Number(right === "description")) : [];
  function showMoreChanges() {
    const region = changeRegion.current;
    if (!region) return;
    const description = scrollTarget === "description" && region.querySelector(".calendar-review-description");
    if (description) {
      region.scrollTop += description.getBoundingClientRect().top - region.getBoundingClientRect().top;
    } else {
      const atEnd = region.scrollTop + region.clientHeight >= region.scrollHeight - 1;
      region.scrollTop = atEnd ? 0 : region.scrollTop + region.clientHeight * 0.8;
    }
    region.scrollIntoView?.({ block: "nearest" });
    region.focus({ preventScroll: true });
  }

  return <section className="calendar-edit-reviews" aria-label="Calendar edit reviews">
    {review && <article key={review.review_token} ref={reviewCard} className={guests ? "has-calendar-review-guests" : undefined} aria-label="Proposed Calendar edit">
      <div className="calendar-review-body" role="region" tabIndex={0} aria-label="Calendar review details">
      <header className="calendar-review-heading">
        <span className="calendar-review-symbol"><CalendarCheck2 aria-hidden="true" size={17} /></span>
        <div>
          <p className="calendar-review-label">Calendar edit review</p>
          <h3>{review.event_title || displayField(review.before.summary)}</h3>
          <p className="calendar-review-label">{review.event_start && <><time dateTime={review.event_start}>{eventDate(review.event_start, timezone)}</time> · </>}{eventScope(review.recurrence_scope)}</p>
        </div>
      </header>
      {choices.length > 1 && <select ref={reviewPicker} aria-label="Calendar edit to review" value={choices.indexOf(review)} disabled={pending !== null} onChange={event => {
        const selected = choices[Number(event.target.value)];
        if (selected && selected.review_token !== review.review_token) {
          restorePickerFocus.current = document.activeElement === event.currentTarget;
          setSelectedToken(selected.review_token);
        }
      }}>
        {choices.map((item, index) => <option key={item.review_token} value={index}>{item.event_title || displayField(item.before.summary)}{item.event_start ? ` · ${eventDate(item.event_start, timezone)}` : ""}</option>)}
      </select>}
      {changes.length > 0 && <div className="calendar-edit-details" ref={changeRegion} role="region" tabIndex={0} aria-label="Proposed field changes">
        {changes.map(([field, value]) => <section className={field === "description" ? "calendar-review-change calendar-review-description" : "calendar-review-change"} key={field}>
            <h4>{fields[field] || field}</h4>
            <dl>
              <div><dt>Current</dt><dd>{displayField(review.before[field], field)}</dd></div>
              <div><dt>Proposed</dt><dd>{displayField(value, field)}</dd></div>
            </dl>
          </section>)}
      </div>}
      {guests && <CalendarGuests guests={guests} />}
      </div>
      <div className="calendar-review-context">
        <details className="calendar-review-event-details"><summary>Event details</summary><dl><dt>Calendar ID</dt><dd>{review.calendar_id}</dd></dl></details>
        {scrollTarget && <button type="button" className="calendar-review-scroll" onClick={showMoreChanges}>
          <ChevronDown size={13} aria-hidden="true" />{scrollTarget === "description" ? "Scroll to description" : "Scroll field changes"}
        </button>}
      </div>
      <footer className="calendar-review-footer">
        {guests && <p id={notificationId} className="calendar-review-notification"><Mail size={15} aria-hidden="true" />{notificationText(guests.current.length + guests.added.length)}</p>}
        <p className="calendar-review-warning"><Info size={15} aria-hidden="true" /><span>{guests
          ? "Wait until Alphonse finishes. Simultaneous Google Calendar edits can overwrite changes or notify other guests."
          : "Wait for Alphonse to finish before editing this event in Google Calendar. A last-second change in Google Calendar can still be overwritten."}</span></p>
        <div className="dialog-actions">
          <button type="button" disabled={pending !== null} onClick={() => void decide(review, false)}>{pending && !pending.approve ? "Dismissing…" : "Dismiss"}</button>
          <button type="button" className="calendar-review-approve" disabled={pending !== null} aria-describedby={guests ? notificationId : undefined} onClick={() => void decide(review, true)}>{pending?.approve ? "Checking Google Calendar…" : guests ? "Approve and notify" : "Approve and update"}</button>
        </div>
      </footer>
    </article>}
    {notice && <p className={notice.verified ? "calendar-review-result is-verified" : "calendar-review-result"} ref={resultNotice} role="status" tabIndex={-1}>
      {notice.verified ? <CircleCheck size={17} aria-hidden="true" /> : <CircleAlert size={17} aria-hidden="true" />}{notice.text}
    </p>}
  </section>;
}
