// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const service = vi.hoisted(() => ({ ingress: vi.fn(), followup: vi.fn(), steer: vi.fn(), followupList: vi.fn(), history: vi.fn() }));
vi.mock("./api", () => ({
  ensureDaemon: async () => {}, playAlertSound: async () => {}, showInFinder: async () => {}, stopDaemon: async () => {},
  daemonRequest: async (method: string, params: Record<string, unknown> = {}) => {
    if (method === "current_user") return { onboarded: true, workspace_configured: true, user: { user_id: "fixture", display_name: "Fixture", role: "admin", is_active: true } };
    if (method === "desktop_poll") return { next_sequence: 0, next_ui_sequence: 0, daemon_id: "fixture", questions: [], events: [], deliveries: [], ui_events: [], status: { queue: { ready: 0, processing: 0 }, activity: { state: "idle" }, active_work: {} } };
    if (method === "experimental_owner_status") return params.message_id ? { status: "pending" } : { processing_ready: true, owner_only: true, paused_legacy_requests: 0 };
    if (method === "experimental_owner_ingress") return service.ingress(params);
    if (method === "experimental_owner_followup_list") return service.followupList();
    if (method === "experimental_owner_composer_save") return { id: params.id, status: "saved" };
    if (method === "experimental_owner_followup_add") return service.followup(params);
    if (method === "experimental_owner_steer_add") return service.steer(params);
    if (method === "desktop_conversation_history") return service.history();
    if (method === "owner_scheduled_results_list") return { results: [] };
    if (method === "timezone_settings") return { timezone: "UTC" };
    if (method === "manageable_projects") return { projects: [] };
    if (method.startsWith("experimental_owner_")) return { items: [] };
    return {};
  },
}));
import App from "./App";
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const flush = () => new Promise((resolve) => setTimeout(resolve, 20));
function typeValue(textarea: HTMLTextAreaElement, value: string) {
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(textarea, value);
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("confirmed conversation sends", () => {
  beforeEach(() => {
    const data = new Map<string, string>();
    const storage = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => data.set(key, String(value)), removeItem: (key: string) => data.delete(key), clear: () => data.clear() };
    Object.defineProperty(window, "localStorage", { configurable: true, value: storage });
    window.matchMedia = vi.fn().mockReturnValue({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() });
    HTMLElement.prototype.scrollTo = vi.fn();
    service.ingress.mockReset(); service.followup.mockReset(); service.steer.mockReset(); service.followupList.mockReset();
    service.followupList.mockResolvedValue({ items: [] }); service.history.mockReset(); service.history.mockResolvedValue({ messages: [] });
    service.followup.mockResolvedValue({ status: "queued" }); service.steer.mockResolvedValue({ status: "accepted" });
  });
  afterEach(() => vi.restoreAllMocks());
  it("clears the exact whitespace-padded draft after confirmed send", async () => {
    let confirm!: (value: unknown) => void;
    service.ingress.mockImplementation(() => new Promise((resolve) => { confirm = resolve; }));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "  A message with space  \n"));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith({ prompt: "A message with space" });
    await act(async () => { confirm({ message_id: "a".repeat(32) }); await flush(); });
    expect(textarea.value).toBe("");
    expect(element.textContent).not.toContain("The active task changed while you were writing.");
    await act(async () => root.unmount()); element.remove();
  });
  it("preserves a draft edited while its earlier submission is in flight", async () => {
    let confirm!: (value: unknown) => void;
    service.ingress.mockImplementation(() => new Promise((resolve) => { confirm = resolve; }));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "  original  "));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    await act(async () => typeValue(textarea, "new text while sending"));
    await act(async () => { confirm({ message_id: "a".repeat(32) }); await flush(); });
    expect(textarea.value).toBe("new text while sending");
    await act(async () => root.unmount()); element.remove();
  });
  it("preserves the draft when submission is not confirmed", async () => {
    service.ingress.mockRejectedValue(new Error("network unavailable"));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "  keep this draft  "));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe("  keep this draft  ");
    expect(element.textContent).toContain("Message submission could not be confirmed");
    await act(async () => root.unmount()); element.remove();
  });
  it.each(["queue", "steer"] as const)("clears a whitespace-padded %s after acceptance", async (mode) => {
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "first request"));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe("");
    await act(async () => typeValue(textarea, "  later details  \n"));
    await act(async () => {
      if (mode === "queue") textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      else element.querySelector<HTMLButtonElement>(".conversation-send-alternative")!.click();
      await flush();
    });
    const sender = mode === "queue" ? service.followup : service.steer;
    expect(sender).toHaveBeenCalledWith(expect.objectContaining({ prompt: "later details", active_message_id: "a".repeat(32) }));
    expect(textarea.value).toBe("");
    await act(async () => root.unmount()); element.remove();
  });

  it.each([false, true])("reconciles an uncertain queued send without erasing later edits: %s", async (edited) => {
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    service.followup.mockRejectedValue(new Error("reply lost"));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "first request"));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    await act(async () => typeValue(textarea, "  queued message  "));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe("  queued message  ");
    const id = service.followup.mock.calls[0][0].request_id;
    if (edited) await act(async () => typeValue(textarea, "newer draft"));
    service.followupList.mockResolvedValue({ items: [{ id, prompt: "queued message", status: "queued", can_remove: true }] });
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 950)); });
    expect(textarea.value).toBe(edited ? "newer draft" : "");
    expect(service.followup).toHaveBeenCalledTimes(1);
    await act(async () => root.unmount()); element.remove();
  });

  it("binds the selected one of two identical failed requests, preserving and clearing draft context safely", async () => {
    const first = "c".repeat(32); const second = "d".repeat(32);
    window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([first, second].map((messageId) => ({ messageId, failureCode: "time_limit", dismissed: false }))));
    service.history.mockResolvedValue({ messages: [first, second].map((id, index) => ({ id, role: "user", source: "desktop", content: "Same request", created_at: new Date(Date.now() + index).toISOString() })) });
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "Existing draft"));
    const buttons = [...element.querySelectorAll("button")].filter((button) => button.textContent === "Reference this failed request in draft");
    expect(buttons).toHaveLength(2);
    await act(async () => { buttons[1].click(); await flush(); });
    expect(textarea.value).toContain("Existing draft"); expect(element.textContent).toContain(`ID: ${second}`);
    expect(service.ingress).not.toHaveBeenCalled();
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith(expect.objectContaining({ failed_request_message_id: second }));
    await act(async () => root.unmount()); element.remove();
  });

  it("lets the user remove a failed-request reference without erasing the draft", async () => {
    const id = "f".repeat(32); window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([{ messageId: id, failureCode: "time_limit", dismissed: false }]));
    service.history.mockResolvedValue({ messages: [{ id, role: "user", source: "desktop", content: "Previous request", created_at: new Date().toISOString() }] });
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Reference this failed request in draft")!.click(); await flush(); });
    const savedDraft = textarea.value;
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Remove reference")!.click(); });
    expect(textarea.value).toBe(savedDraft); expect(document.activeElement).toBe(textarea); expect(element.textContent).not.toContain(`ID: ${id}`);
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith({ prompt: savedDraft.trim() });
    await act(async () => root.unmount()); element.remove();
  });

  it("preserves the draft and allows removal when a failed reference was rejected before enqueue", async () => {
    const id = "1".repeat(32); window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([{ messageId: id, failureCode: "time_limit", dismissed: false }]));
    service.history.mockResolvedValue({ messages: [{ id, role: "user", source: "desktop", content: "Previous request", created_at: new Date().toISOString() }] });
    service.ingress.mockRejectedValueOnce(new Error("ValueError: owner_failure_reference_unavailable")).mockResolvedValueOnce({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Reference this failed request in draft")!.click(); await flush(); });
    const draft = textarea.value;
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe(draft); expect(element.textContent).toContain("Your draft was not sent");
    expect(element.textContent).not.toContain("Message submission could not be confirmed");
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)); });
    expect(document.activeElement?.textContent).toBe("Remove reference");
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Remove reference")!.click(); });
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenLastCalledWith({ prompt: draft.trim() });
    await act(async () => root.unmount()); element.remove();
  });

  it("removes a failed-request reference if its quoted text is deleted", async () => {
    const id = "e".repeat(32); window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([{ messageId: id, failureCode: "time_limit", dismissed: false }]));
    service.history.mockResolvedValue({ messages: [{ id, role: "user", source: "desktop", content: "Same request", created_at: new Date().toISOString() }] });
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    const button = [...element.querySelectorAll("button")].find((item) => item.textContent === "Reference this failed request in draft")!;
    await act(async () => { button.click(); await flush(); });
    await act(async () => typeValue(textarea, "A different question"));
    expect(element.textContent).not.toContain(`ID: ${id}`);
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith({ prompt: "A different question" });
    await act(async () => root.unmount()); element.remove();
  });

});
