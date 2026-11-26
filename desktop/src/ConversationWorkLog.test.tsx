// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { ConversationWorkLog } from "./ConversationWorkLog";
import { ConversationFailure } from "./ConversationFailure";
import type { OwnerRequest } from "./ownerConversation";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("summarizes recorded observations while preserving grouped technical details and uncertainty", async () => {
  const container = document.createElement("div"); document.body.appendChild(container);
  const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "running", failure_code: null, notice: "May be incomplete.",
    activity: { recorded_action_count: 3, unconfirmed_effect_count: 2, pending_tool: true, latest: { tool_kind: "web_search", outcome: "no_results", effects: "not_reported" } },
    events: [{ label: "Local command", outcome: "Command returned exit code 1; effects are unconfirmed" },
      { label: "Local command", outcome: "Command returned exit code 1; effects are unconfirmed" },
      { label: "Searched the web", outcome: "No results returned" }] })) as unknown as OwnerRequest;
  await act(async () => { root.render(<ConversationWorkLog request={request} messageId={"a".repeat(32)} active />); });
  expect(container.textContent).toContain("3 tool observations recorded.");
  expect(container.textContent).toContain("A tool request was recorded; whether it ran is unconfirmed.");
  expect(container.textContent).toContain("Web search: no results were returned.");
  expect(container.textContent).toContain("The log cannot confirm whether the recorded local commands changed anything.");
  expect(container.textContent).toContain("Command returned exit code 1; effects are unconfirmed (2 times)");
  const technical = container.querySelector<HTMLDetailsElement>(".conversation-work-log-details")!;
  expect(technical.open).toBe(false);
  expect(technical.textContent).toContain("Searched the web · No results returned");
  expect(request).toHaveBeenCalledWith("experimental_owner_work_log", { message_id: "a".repeat(32) });
  await act(async () => { root.unmount(); }); container.remove();
});

it("supports older work logs without inventing effects or pending work", async () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "completed", failure_code: null, notice: "May be incomplete.", events: [] })) as unknown as OwnerRequest;
  await act(async () => { root.render(<ConversationWorkLog request={request} messageId={"a".repeat(32)} active />); });
  expect(container.textContent).toContain("No tool observations recorded yet.");
  expect(container.textContent).not.toContain("whether it ran");
  expect(container.textContent).not.toContain("Effects of local commands");
  await act(async () => { root.unmount(); });
});

it("shows each Gmail label in search order while technical details are closed", async () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "completed", failure_code: null, notice: "May be incomplete.",
    activity: { recorded_action_count: 3, unconfirmed_effect_count: 0, pending_tool: false,
      latest: { tool_kind: "mail_search", outcome: "recorded", effects: "not_reported", gmail_label: "Family/School" } },
    events: ["Inbox", "Family/School", "Family/School"].map((gmail_label) => ({
      label: "Searched Gmail", outcome: "Observation recorded", gmail_label,
    })),
  })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationWorkLog request={request} messageId={"a".repeat(32)} active />));
  expect(Array.from(container.querySelectorAll(".conversation-work-log > p"), (item) => item.textContent)).toEqual([
    "3 tool observations recorded.",
    "Gmail search · Inbox → Family/School → Family/School",
  ]);
  expect(container.querySelector<HTMLDetailsElement>(".conversation-work-log-details")!.open).toBe(false);
  expect(Array.from(container.querySelectorAll("li"), (item) => item.textContent)).toEqual([
    "Searched Gmail · Inbox · Observation recorded",
    "Searched Gmail · Family/School · Observation recorded (2 times)",
  ]);
  expect(container.textContent).not.toContain("Gmail Inbox search");
  await act(async () => root.unmount());
});

it("does not infer a Gmail label on a work log without a recorded scope", async () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "completed", failure_code: null, notice: "May be incomplete.", events: [],
    activity: { recorded_action_count: 1, unconfirmed_effect_count: 0, pending_tool: false,
      latest: { tool_kind: "mail_search", outcome: "source_failed", effects: "not_reported" } },
  })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationWorkLog request={request} messageId={"a".repeat(32)} active />));
  expect(container.textContent).toContain("Gmail search: could not be completed.");
  expect(container.textContent).not.toContain("Inbox");
  await act(async () => root.unmount());
});

it("opens when an existing request becomes active without closing when work ends", async () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "running", failure_code: null, notice: "May be incomplete.", events: [] })) as unknown as OwnerRequest;
  const render = (active: boolean) => root.render(<ConversationWorkLog request={request} messageId={"a".repeat(32)} active={active} />);
  await act(async () => render(false));
  expect(container.querySelector("details")!.open).toBe(false);
  await act(async () => render(true));
  expect(container.querySelector("details")!.open).toBe(true);
  await act(async () => render(false));
  expect(container.querySelector("details")!.open).toBe(true);
  await act(async () => { root.unmount(); });
});

it("offers only explicitly available recovery controls", () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const review = vi.fn();
  act(() => root.render(<ConversationFailure message="Stopped" dismissed={false} onDismissedChange={() => {}} onReviewRecovery={review} />));
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === "Review request status");
  expect(button).toBeTruthy(); act(() => button!.click()); expect(review).toHaveBeenCalledOnce();
  expect(container.textContent).not.toContain("Open model settings");
  expect(container.textContent).toContain("check the relevant service or device before trying that change again");
  expect(container.textContent).not.toContain("The work log above");
  act(() => root.unmount());
});

it("opens recorded evidence when a model format failure is restored", async () => {
  const { ConversationTimelineEntry } = await import("./ConversationTimelineEntry");
  const container = document.createElement("div");
  const root = createRoot(container);
  const id = "f".repeat(32);
  const request = vi.fn(async () => ({ status: "blocked", failure_code: "model_response_invalid", notice: "May be incomplete.",
    model_parse_stages: ["arguments_invalid_json"],
    events: [{ label: "Local command", outcome: "Command timed out; effects are unconfirmed" }],
    activity: { recorded_action_count: 1, unconfirmed_effect_count: 1, pending_tool: false,
      outcome_counts: [{ tool_kind: "local_command", outcome: "timed_out", count: 1 }] },
  })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationTimelineEntry message={{ id, role: "user", content: "Check the fixture" }}
    questions={[]} user="fixture" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false}
    failures={[{ messageId: id, failureCode: "model_response_invalid", dismissed: false }]}
    onFailureDismissedChange={() => {}} workLogRequest={request} />));
  const workLog = container.querySelector<HTMLDetailsElement>(".conversation-work-log")!;
  expect(workLog.open).toBe(true);
  expect(workLog.textContent).toContain("1 timed out");
  expect(workLog.textContent).toContain("cannot confirm whether the recorded local commands changed anything");
  const technical = container.querySelector<HTMLDetailsElement>(".conversation-work-log-details")!;
  expect(technical.open).toBe(false);
  expect(technical.textContent).toContain("Tool arguments could not be decoded as JSON");
  expect(request).toHaveBeenCalledWith("experimental_owner_work_log", { message_id: id });
  await act(async () => root.unmount());
});

it("sends the failed request ID to recovery and matches the recovery destination", async () => {
  const { ConversationTimelineEntry } = await import("./ConversationTimelineEntry");
  const { OwnerRecovery, focusConversationRequest, focusOwnerRecovery } = await import("./OwnerRecoveryPanel");
  const id = "b".repeat(32);
  const container = document.createElement("div"); document.body.appendChild(container);
  const root = createRoot(container); const review = vi.fn();
  await act(async () => root.render(<><ConversationTimelineEntry message={{ id, role: "user", content: "Find my update" }} questions={[]} user="fixture" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={[{ messageId: id, failureCode: "time_limit", dismissed: false }]} onFailureDismissedChange={() => {}} onReviewRecovery={review} /><OwnerRecovery items={[{ message_id: id, prompt: "Find my update", status: "action_unconfirmed", can_abandon: false }]} stopping="" onStop={() => {}} onShow={() => {}} /></>));
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === "Review request status")!;
  act(() => button.click());
  expect(review).toHaveBeenCalledWith(id);
  expect(focusOwnerRecovery(id)).toBe(true);
  expect(document.activeElement?.id).toBe(`owner-recovery-${id}`);
  expect(focusConversationRequest(id)).toBe(true);
  expect(document.activeElement?.id).toBe(`owner-recovery-trigger-${id}`);
  expect(container.textContent).toContain("Check the relevant device or service before trying again. No stop action is available here.");
  await act(async () => root.unmount()); container.remove();
});

it.each([
  ["web_source", "source_failed", "timeout", "Web source: could not be completed. The request timed out."],
  ["web_search", "source_failed", "upstream_search_failed", "Web search: could not be completed. Upstream search engines failed."],
  ["web_search", "partial", undefined, "Web search: returned partial results because some search engines failed."],
])("reports a %s observation with outcome %s", async (tool_kind, outcome, detail, expected) => {
  const container = document.createElement("div"); const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "blocked", failure_code: "source_unavailable", notice: "May be incomplete.", events: [],
    activity: { recorded_action_count: 1, unconfirmed_effect_count: 0, pending_tool: false, latest: { tool_kind, outcome, effects: "not_reported", detail } } })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationWorkLog request={request} messageId={"c".repeat(32)} active />));
  expect(container.textContent).toContain(expected);
  expect(container.textContent).not.toContain("Read a web source");
  expect(container.textContent).not.toContain("changed anything");
  await act(async () => root.unmount());
});

it("explains when no recovery action is offered", () => {
  const container = document.createElement("div"); const root = createRoot(container);
  act(() => root.render(<ConversationFailure message="Stopped" dismissed={false} onDismissedChange={() => {}} />));
  expect(container.textContent).toContain("No recovery action is available here.");
  expect(container.textContent).not.toContain("work log above");
  act(() => root.unmount());
});

it("returns focus to the timeline if the originating recovery button is unavailable", async () => {
  const { focusConversationRequest } = await import("./OwnerRecoveryPanel");
  const timeline = document.createElement("div"); timeline.id = "conversation-timeline"; timeline.tabIndex = -1; document.body.appendChild(timeline);
  expect(focusConversationRequest("d".repeat(32))).toBe(true);
  expect(document.activeElement).toBe(timeline);
  timeline.remove();
});

it("loads recovery activity only for the selected request and handles unavailable logs", async () => {
  const { OwnerRecovery } = await import("./OwnerRecoveryPanel");
  const first = "e".repeat(32); const second = "f".repeat(32);
  const request = vi.fn(async (_method: string, args: { message_id: string }) => {
    if (args.message_id === second) throw new Error("unavailable");
    return { status: "blocked", failure_code: null, notice: "May be incomplete.", events: [],
      activity: { recorded_action_count: 1, unconfirmed_effect_count: 0, pending_tool: false,
        latest: { tool_kind: "web_search", outcome: "no_results", effects: "not_reported" } } };
  }) as unknown as OwnerRequest;
  const container = document.createElement("div"); document.body.appendChild(container); const root = createRoot(container);
  const items = [first, second].map((message_id) => ({ message_id, prompt: "Fixture", status: "action_unconfirmed", can_abandon: false }));
  await act(async () => root.render(<OwnerRecovery items={items} stopping="" onStop={() => {}} onShow={() => {}} workLogRequest={request} />));
  const logs = container.querySelectorAll<HTMLDetailsElement>(".conversation-work-log");
  expect(logs).toHaveLength(2); expect(request).not.toHaveBeenCalled();
  expect(logs[0].querySelector("summary")?.getAttribute("aria-labelledby")).toBe(`work-log-summary-${first} owner-recovery-prompt-${first}`);
  expect(logs[1].querySelector("summary")?.getAttribute("aria-labelledby")).toBe(`work-log-summary-${second} owner-recovery-prompt-${second}`);
  await act(async () => { logs[0].open = true; logs[0].dispatchEvent(new Event("toggle")); });
  expect(request).toHaveBeenCalledWith("experimental_owner_work_log", { message_id: first });
  expect(logs[0].textContent).toContain("Web search: no results were returned.");
  expect(logs[0].textContent).toContain("not source content or proof of an action’s effects");
  expect(logs[1].textContent).not.toContain("Web search: no results were returned.");
  await act(async () => { logs[1].open = true; logs[1].dispatchEvent(new Event("toggle")); });
  expect(request).toHaveBeenCalledWith("experimental_owner_work_log", { message_id: second });
  expect(logs[1].textContent).toContain("Work log is unavailable or not recorded");
  expect([...container.querySelectorAll("button")].every((button) => button.textContent !== "Retry")).toBe(true);
  await act(async () => root.unmount()); container.remove();
});

it("shows loading then empty activity, and omits the control without an authorized request", async () => {
  const { OwnerRecovery } = await import("./OwnerRecoveryPanel");
  const id = "1".repeat(32); const item = { message_id: id, prompt: "Fixture", status: "failed", can_abandon: false };
  const container = document.createElement("div"); const root = createRoot(container);
  let resolve!: (value: unknown) => void;
  const request = vi.fn(() => new Promise((done) => { resolve = done; })) as unknown as OwnerRequest;
  await act(async () => root.render(<OwnerRecovery items={[item]} stopping="" onStop={() => {}} onShow={() => {}} />));
  expect(container.querySelector(".conversation-work-log")).toBeNull();
  await act(async () => root.render(<OwnerRecovery items={[item]} stopping="" onStop={() => {}} onShow={() => {}} workLogRequest={request} />));
  const disclosure = container.querySelector<HTMLDetailsElement>(".conversation-work-log")!;
  await act(async () => { disclosure.open = true; disclosure.dispatchEvent(new Event("toggle")); });
  expect(disclosure.textContent).toContain("Checking recorded work");
  await act(async () => resolve({ status: "blocked", failure_code: null, notice: "May be incomplete.", events: [], activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: false, latest: null } }));
  expect(disclosure.textContent).toContain("No tool observations recorded yet.");
  await act(async () => root.unmount());
});

it("does not equate a command exit with the request's outcome", async () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "running", failure_code: null, notice: "May be incomplete.", events: [],
    activity: { recorded_action_count: 1, unconfirmed_effect_count: 1, pending_tool: false, latest: { tool_kind: "local_command", outcome: "exit_nonzero", effects: "unconfirmed" } } })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationWorkLog request={request} messageId={"2".repeat(32)} active />));
  expect(container.textContent).toContain("what that means for this request is unknown");
  expect(container.textContent).toContain("cannot confirm whether the recorded local commands changed anything");
  await act(async () => root.unmount());
});

it("drafts a question about the matching request without sending or overwriting an existing draft", async () => {
  const { ConversationTimelineEntry } = await import("./ConversationTimelineEntry");
  const { draftQuestionAboutRequest } = await import("./conversationFollowup");
  const id = "3".repeat(32); const container = document.createElement("div"); const root = createRoot(container);
  let draft = "My existing draft";
  const onDraftQuestion = vi.fn((_id: string, content: string) => { draft = draftQuestionAboutRequest(draft, content); });
  await act(async () => root.render(<ConversationTimelineEntry message={{ id, role: "user", content: "First line\nSecond line" }} questions={[]} user="fixture" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={[{ messageId: id, failureCode: "time_limit", dismissed: false }]} onFailureDismissedChange={() => {}} onDraftQuestion={onDraftQuestion} />));
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === "Reference this failed request in draft")!;
  act(() => button.click());
  expect(onDraftQuestion).toHaveBeenCalledWith(id, "First line\nSecond line", undefined);
  expect(draft).toBe("My existing draft\n\nI have a question about this earlier request text:\n\n> First line\n> Second line");
  act(() => button.click());
  expect(draft).toBe("My existing draft\n\nI have a question about this earlier request text:\n\n> First line\n> Second line");
  await act(async () => root.unmount());
});

it("opens and focuses only the matching request's recorded activity", async () => {
  const { ConversationTimelineEntry } = await import("./ConversationTimelineEntry");
  const id = "4".repeat(32); const container = document.createElement("div"); document.body.appendChild(container); const root = createRoot(container);
  const request = vi.fn(async () => ({ status: "blocked", failure_code: null, notice: "May be incomplete.", events: [] })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationTimelineEntry message={{ id, role: "user", content: "Fixture request" }} questions={[]} user="fixture" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={[{ messageId: id, failureCode: "time_limit", dismissed: false }]} onFailureDismissedChange={() => {}} workLogRequest={request} />));
  const details = container.querySelector<HTMLDetailsElement>(`#conversation-work-log-${id}`)!;
  expect(details.open).toBe(false); expect(request).not.toHaveBeenCalled();
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === "Review recorded activity")!;
  await act(async () => { button.click(); details.dispatchEvent(new Event("toggle")); });
  expect(details.open).toBe(true); expect(document.activeElement).toBe(details.querySelector("summary"));
  expect(request).toHaveBeenCalledWith("experimental_owner_work_log", { message_id: id });
  await act(async () => root.unmount()); container.remove();
});

it("routes each failure card to its own log and reports unavailable evidence", async () => {
  const { ConversationTimelineEntry } = await import("./ConversationTimelineEntry");
  const ids = ["5".repeat(32), "6".repeat(32)]; const container = document.createElement("div"); document.body.appendChild(container); const root = createRoot(container);
  const request = vi.fn(async () => { throw new Error("unavailable"); }) as unknown as OwnerRequest;
  await act(async () => root.render(<>{ids.map((id) => <ConversationTimelineEntry key={id} message={{ id, role: "user", content: `Request ${id[0]}` }} questions={[]} user="fixture" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={[{ messageId: id, failureCode: "time_limit", dismissed: false }]} onFailureDismissedChange={() => {}} workLogRequest={request} />)}</>));
  const logs = ids.map((id) => container.querySelector<HTMLDetailsElement>(`#conversation-work-log-${id}`)!);
  const buttons = [...container.querySelectorAll("button")].filter((item) => item.textContent === "Review recorded activity");
  await act(async () => { buttons[1].click(); logs[1].dispatchEvent(new Event("toggle")); });
  expect(logs[0].open).toBe(false); expect(logs[1].open).toBe(true);
  expect(request).toHaveBeenCalledWith("experimental_owner_work_log", { message_id: ids[1] });
  expect(request).not.toHaveBeenCalledWith("experimental_owner_work_log", { message_id: ids[0] });
  expect(logs[1].textContent).toContain("Work log is unavailable or not recorded");
  await act(async () => root.unmount()); container.remove();
});

it("attributes bounded intent only to a running pending reservation", async () => {
  const container = document.createElement("div"); const root = createRoot(container);
  let status = "running";
  const request = vi.fn(async () => ({ status, failure_code: null, notice: "May be incomplete.", events: [],
    activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: true, pending_intent: "check_state", latest: null } })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationWorkLog request={request} messageId={"7".repeat(32)} active />));
  expect(container.textContent).toContain("Alphonse says it plans to check a state. A tool request was recorded; whether it ran is unconfirmed.");
  status = "blocked";
  await act(async () => root.render(<ConversationWorkLog request={request} messageId={"7".repeat(32)} active={false} />));
  expect(container.textContent).not.toContain("plans to check a state");
  expect(container.textContent).toContain("whether it ran is unconfirmed");
  await act(async () => root.unmount());
});

it("does not replace newer activity with an older poll that finishes late", async () => {
  vi.useFakeTimers();
  const container = document.createElement("div"); document.body.appendChild(container); const root = createRoot(container);
  const pending: Array<(value: unknown) => void> = [];
  const request = vi.fn(() => new Promise((resolve) => pending.push(resolve))) as unknown as OwnerRequest;
  try {
    await act(async () => root.render(<ConversationWorkLog request={request} messageId={"8".repeat(32)} active />));
    await act(async () => vi.advanceTimersByTime(2000));
    expect(pending).toHaveLength(2);
    await act(async () => pending[1]({ status: "blocked", failure_code: "time_limit", notice: "May be incomplete.", events: [],
      activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: false, pending_intent: null, latest: null } }));
    expect(container.textContent).toContain("No tool observations recorded yet.");
    await act(async () => pending[0]({ status: "running", failure_code: null, notice: "May be incomplete.", events: [],
      activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: true, pending_intent: "check_state", latest: null } }));
    expect(container.textContent).not.toContain("plans to check a state");
    expect(container.textContent).not.toContain("whether it ran is unconfirmed");
  } finally {
    await act(async () => root.unmount()); container.remove(); vi.useRealTimers();
  }
});

it("summarizes mixed command outcomes without interpreting their purpose or losing chronology", async () => {
  const container = document.createElement("div"); const root = createRoot(container);
  const events = ["exit 1", "exit 0", "timed out", "exit 1", "unknown", "exit 0"].map((outcome) => ({ label: "Local command", outcome }));
  const request = vi.fn(async () => ({ status: "running", failure_code: null, notice: "May be incomplete.", events,
    activity: { recorded_action_count: 6, unconfirmed_effect_count: 6, pending_tool: false,
      outcome_counts: [{ tool_kind: "local_command", outcome: "exit_nonzero", count: 2 }, { tool_kind: "local_command", outcome: "exit_zero", count: 2 }, { tool_kind: "local_command", outcome: "timed_out", count: 1 }, { tool_kind: "local_command", outcome: "unknown", count: 1 }],
      latest: { tool_kind: "local_command", outcome: "exit_zero", effects: "unconfirmed" } } })) as unknown as OwnerRequest;
  await act(async () => root.render(<ConversationWorkLog request={request} messageId={"9".repeat(32)} active />));
  expect(container.textContent).toContain("Local commands: 2 returned a code that can indicate a problem; 2 reported a normal command exit; 1 timed out; 1 had an unknown outcome. Some commands use these codes for other reasons. They do not establish what happened with your request.");
  expect(container.textContent).toContain("cannot confirm whether the recorded local commands changed anything");
  expect([...container.querySelectorAll(".conversation-work-log-details li")].map((item) => item.textContent)).toEqual(events.map((event) => `Local command · ${event.outcome}`));
  expect(container.textContent).not.toContain("request failed");
  await act(async () => root.unmount());
});

it("quotes text without claiming to distinguish identical prior requests", async () => {
  const { draftQuestionAboutRequest } = await import("./conversationFollowup");
  const first = draftQuestionAboutRequest("", "Same question");
  const second = draftQuestionAboutRequest("", "Same question");
  expect(first).toBe(second);
  expect(first).toBe("I have a question about this earlier request text:\n\n> Same question");
  expect(first).not.toContain("failed attempt");
});
